"""Replace the single arm of finished single-vs-RSI runs with the one-shot single run.

Before 2026-10-20 the "single" arm iterated under a fixed improvement workflow (gated rounds until the round or
time budget ran out). The single run is now one model request whose programs are frozen as returned
(code_agents.request_one_shot). For each run this script:

  request  moves <run>/single to <run>/fixed_workflow (kept for reference) and makes the one request into
           <run>/single, reusing the old arm's round-1 evaluations of the same deterministic starting programs.
           No solver runs, so it can run on the login node.
  rerequest  re-requests single runs whose one answer was cut off by the run's output cap (stopReason
           max_tokens), with the model's own maximum; the capped attempt stays in <run>/single_capped_<N>k.
  restate  re-requests single runs whose answer exceeded the program-length limit, with the v2 prompt that
           states the limit; the rejected attempt stays in <run>/single_overlimit_v1.
  repair   single run with one rule repair: if the single answer was rejected before execution or cut off,
           the rejection is sent back once and one corrected answer is accepted, into <run>/single_repair.
  score-repair  (GPU node) scores <run>/single_repair on the public cases.
  score    (GPU node, after sourcing scripts/env.sh) scores the frozen single-run programs on the public cases,
           writes single/report.json, updates the manifest and per-run tables, and scores the sealed splits the
           run's RSI arm was scored on. Old sealed summaries keep the fixed workflow under "fixed_workflow".

    python scripts/redo_single_one_shot.py request --runs runs/code-agents-pdebench2h-*-20261017
    python scripts/redo_single_one_shot.py score --runs ... [--core 53]
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
import platform
import shutil
import sys
import threading
from pathlib import Path

from pde_rsi.code_agents import (
    ONE_SHOT_PROMPT_VERSION, CodeProtocol, CodeVersion, finish_one_shot, request_one_shot, write_summary, write_tables,
)
from pde_rsi.codegen import SANDBOX_VERSION, program_digest
from pde_rsi.evaluator import EVALUATOR_VERSION
from pde_rsi.experiment import write_json
from pde_rsi.model_backend import ModelBackend, ModelConfig
from pde_rsi.runner import Budget
from pde_rsi.storage import project_path

ROOT = project_path(".")
PACKAGE = ROOT/"datasets/huggingface/pde-problems"
LOADING = threading.Lock()  # task loading (dataset readers, tqdm) is not thread-safe


def protocol_for(manifest):
    name = manifest.get("task_protocol") or "code"
    settings = manifest.get("task_protocol_metadata") or {}
    modular = settings.get("harness") == "v2-modular"
    if name == "pdebench":
        from pde_rsi.pdebench import PDEBenchProtocol
        return PDEBenchProtocol(settings.get("noise_fraction", 0.01), bool(settings.get("hide_time_dependence")),
                                modular=modular)
    if name == "inventory":
        from pde_rsi.inventory import InventoryProtocol
        return InventoryProtocol(settings.get("noise_fraction", 0.01), modular=modular)
    if name in ("identification", "student"):
        from pde_rsi.identification import IdentificationProtocol, StudentProtocol
        return StudentProtocol() if name == "student" else IdentificationProtocol()
    return CodeProtocol()


def load(run):
    """Manifest, protocol, tasks, and starting programs, checked against what the run recorded."""
    manifest = json.loads((run/"manifest.json").read_text())
    if manifest.get("kind") != "code_writing_single_vs_rsi" or "rsi" not in manifest.get("arms", {}):
        raise ValueError("not a finished code-writing single-vs-RSI run")
    protocol = protocol_for(manifest)
    from pde_rsi.tasks import FAMILIES
    families = tuple(manifest.get("families") or FAMILIES)
    tasks = protocol.load_tasks(ROOT, PACKAGE, families)
    ids = {split: [t.task_id for t in items] for split, items in tasks.items()}
    if ids != manifest["task_ids"]:
        raise ValueError("public tasks differ from the run's recorded task IDs")
    starting = CodeVersion(model=manifest["model_config"]["model"],
                           program=protocol.starting_program(manifest["start"], families))
    if program_digest(starting.program) != manifest["starting_program"]:
        raise ValueError("starting programs differ from the run's recorded digest")
    return manifest, protocol, tasks, starting


def request(run):
    with LOADING:
        manifest, protocol, tasks, starting = load(run)
    if (run/"single/one_shot.json").exists():
        return f"{run.name}: already requested"
    if (run/"fixed_workflow").exists():
        raise ValueError("fixed_workflow exists but single/one_shot.json does not; inspect by hand")
    old = run/"single"
    evolution_feedback = json.loads((old/"evolution-001.json").read_text())
    validation_feedback = json.loads((old/"initial_validation.json").read_text())
    old.rename(run/"fixed_workflow")
    manifest["arms"]["fixed_workflow"] = manifest["arms"].pop("single")
    manifest["single_arm"] = "one_shot_requested"
    write_json(run/"manifest.json", manifest)
    config = ModelConfig(**manifest["model_config"])
    request_one_shot(run/"single", starting, tasks["evolution"], tasks["development"], ModelBackend(config),
                     protocol, evolution_feedback, validation_feedback,
                     model_timeout=manifest["configuration"].get("model_timeout_seconds", 900),
                     max_input_chars=manifest["configuration"].get("max_input_characters_per_request", 500000))
    record = json.loads((run/"single/one_shot.json").read_text())
    return f"{run.name}: {record['status']} {record['message'] or ''}"


def rerequest(run, max_output_tokens, model_timeout):
    """Re-request a single run whose one answer was cut off by the harness output cap (stopReason max_tokens),
    with a larger cap. The capped attempt is kept as <run>/single_capped_<N>k."""
    with LOADING:
        manifest, protocol, tasks, starting = load(run)
    record = json.loads((run/"single/one_shot.json").read_text())
    if "incomplete_response" not in (record["message"] or ""):
        return f"{run.name}: not truncated ({record['status']}), left as is"
    config = ModelConfig(**{**manifest["model_config"], "max_output_tokens": max_output_tokens})
    if config.max_output_tokens <= manifest["model_config"]["max_output_tokens"]:
        raise ValueError("the new output cap must exceed the run's")
    capped = run/f"single_capped_{manifest['model_config']['max_output_tokens']//1000}k"
    old = run/"fixed_workflow" if (run/"fixed_workflow").exists() else run/"single"  # v2 runs have no fixed_workflow
    evolution_feedback = json.loads((old/"evolution-001.json").read_text())
    validation_feedback = json.loads((old/"initial_validation.json").read_text())
    (run/"single").rename(capped)
    manifest.setdefault("single_capped_attempts", []).append(
        {"directory": capped.name, "max_output_tokens": manifest["model_config"]["max_output_tokens"],
         "status": record["status"], "message": record["message"]})
    manifest["single_arm"] = "one_shot_requested"
    write_json(run/"manifest.json", manifest)
    request_one_shot(run/"single", starting, tasks["evolution"], tasks["development"], ModelBackend(config),
                     protocol, evolution_feedback, validation_feedback, model_timeout=model_timeout,
                     max_input_chars=manifest["configuration"].get("max_input_characters_per_request", 500000))
    record = json.loads((run/"single/one_shot.json").read_text())
    return f"{run.name}: {record['status']} {record['message'] or ''}"


def restate(run, model_timeout):
    """Re-request a single run whose answer was rejected for exceeding the program-length limit, with the v2
    prompt that states the limit up front. The rejected attempt is kept as <run>/single_overlimit_v1."""
    with LOADING:
        manifest, protocol, tasks, starting = load(run)
    record = json.loads((run/"single/one_shot.json").read_text())
    if "exceeds" not in (record["message"] or "") or record.get("prompt_version") == ONE_SHOT_PROMPT_VERSION:
        return f"{run.name}: not an over-long v1 answer, left as is"
    config = ModelConfig(**{**manifest["model_config"], "max_output_tokens": record["max_output_tokens"]})
    old = run/"fixed_workflow"
    evolution_feedback = json.loads((old/"evolution-001.json").read_text())
    validation_feedback = json.loads((old/"initial_validation.json").read_text())
    (run/"single").rename(run/"single_overlimit_v1")
    manifest.setdefault("single_superseded_attempts", []).append(
        {"directory": "single_overlimit_v1", "prompt_version": record.get("prompt_version", "v1"),
         "status": record["status"], "message": record["message"]})
    manifest["single_arm"] = "one_shot_requested"
    write_json(run/"manifest.json", manifest)
    request_one_shot(run/"single", starting, tasks["evolution"], tasks["development"], ModelBackend(config),
                     protocol, evolution_feedback, validation_feedback, model_timeout=model_timeout,
                     max_input_chars=manifest["configuration"].get("max_input_characters_per_request", 500000))
    record = json.loads((run/"single/one_shot.json").read_text())
    return f"{run.name}: {record['status']} {record['message'] or ''}"


def sealed(run, manifest, arm_directory):
    """Score the frozen single-run program on the sealed splits that RSI was scored on."""
    from pde_rsi import inventory
    from pde_rsi.pdebench import PROGRAM_KEY, make_cases, run_case
    settings = manifest.get("task_protocol_metadata") or {}
    if manifest.get("task_protocol") == "inventory":
        splits = [("sealed_test", lambda: inventory.make_cases("inventory_test", settings.get("noise_fraction", 0.01)),
                   inventory.run_case)]
    elif manifest.get("task_protocol") == "pdebench":
        splits = [(folder, (lambda s=split: make_cases(s, settings.get("noise_fraction", 0.01),
                                                       settings.get("hide_time_dependence", False))), run_case)
                  for split, folder in (("test", "sealed_test"), ("test2", "sealed_test2"))]
    else:
        return
    from pde_rsi.modular import load_program
    code = load_program(arm_directory/"programs/frozen")
    for folder, cases, runner in splits:
        if not (run/"rsi"/folder/"results.json").exists():
            continue
        target = arm_directory/folder
        if target.exists():
            shutil.rmtree(target)
        (target/"solutions").mkdir(parents=True)
        items = cases()
        budget = Budget(len(items), 3600, max_cpu_seconds=3600)
        results = [runner(case, code, budget, target/"solutions"/f"{case.case_id}.npz") for case in items]
        write_json(target/"results.json", results)
        summary_path = run/f"{folder}_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if "fixed_workflow" not in summary and "single" in summary and not summary["single"].get("one_shot"):
            summary["fixed_workflow"] = summary.pop("single")
        summary["single"] = {"cases": len(results), "passes": sum(r["passed"] for r in results),
                             "mean_relative_l2": sum(r["relative_l2"] for r in results)/len(results),
                             "class_accuracy": sum(bool(r["family_correct"]) for r in results)/len(results),
                             "executed": sum(r["status"] == "ok" for r in results), "one_shot": True}
        write_json(summary_path, summary)
        print(run.name, folder, summary["single"], flush=True)


def repair(run, model_timeout):
    """Single run with one rule repair (code_agents.repair_one_shot) into <run>/single_repair; the strict single
    run in <run>/single stays as it is. Only runs whose single answer was rejected or cut off are repaired."""
    from pde_rsi.code_agents import repair_one_shot
    with LOADING:
        manifest, protocol, tasks, starting = load(run)
    record = json.loads((run/"single/one_shot.json").read_text())
    if record["status"] == "ok":
        return f"{run.name}: single answer valid, no repair needed"
    if (run/"single_repair").exists():
        return f"{run.name}: already repaired"
    config = ModelConfig(**{**manifest["model_config"], "max_output_tokens": record["max_output_tokens"]})
    repair_one_shot(run/"single", run/"single_repair", starting, tasks["evolution"], tasks["development"],
                    ModelBackend(config), protocol, model_timeout=model_timeout)
    record = json.loads((run/"single_repair/one_shot.json").read_text())
    return f"{run.name}: {record['status']} {record['message'] or ''}"


def score_repair(run):
    """Score <run>/single_repair on the public cases (report.json); the manifest gains arms["single_repair"]."""
    manifest, protocol, tasks, starting = load(run)
    directory = run/"single_repair"
    record = json.loads((directory/"one_shot.json").read_text())
    configuration = manifest["configuration"]
    budget = Budget(len(tasks["evolution"])+len(tasks["development"]), configuration["max_wall_seconds_per_arm"],
                    max_cpu_seconds=configuration["max_cpu_seconds_per_arm"])
    for name in ("solutions", "frozen_results.json", "solve_calls.jsonl", "report.json"):
        path = directory/name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    report, _, _ = finish_one_shot(directory, starting, tasks["evolution"], tasks["development"], budget, protocol,
                                   record["max_output_tokens"])
    report["repair"] = True
    write_json(directory/"report.json", report)
    manifest["arms"]["single_repair"] = report
    write_json(run/"manifest.json", manifest)
    value = report["frozen_public"]
    return f"{run.name}: repair {record['status']} public {round(value['pass_rate']*value['count'])}/{value['count']}"


def score(run):
    manifest, protocol, tasks, starting = load(run)
    if not (run/"single/one_shot.json").exists():
        raise ValueError("request the single run first")
    evolution, development = tasks["evolution"], tasks["development"]
    configuration = manifest["configuration"]
    budget = Budget(len(evolution)+len(development), configuration["max_wall_seconds_per_arm"],
                    max_cpu_seconds=configuration["max_cpu_seconds_per_arm"])
    for name in ("solutions", "frozen_results.json", "solve_calls.jsonl", "report.json"):
        path = run/"single"/name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    record = json.loads((run/"single/one_shot.json").read_text())
    report, version, frozen = finish_one_shot(run/"single", starting, evolution, development, budget, protocol,
                                              record.get("max_output_tokens",
                                                         manifest["model_config"].get("max_output_tokens")))
    manifest["arms"] = {"single": report, "rsi": manifest["arms"]["rsi"],
                        **({"fixed_workflow": manifest["arms"]["fixed_workflow"]}
                           if "fixed_workflow" in manifest["arms"] else {})}
    manifest["single_arm"] = "one_shot"
    manifest["single_arm_scored"] = {"utc": datetime.now(timezone.utc).isoformat(), "node": platform.node(),
                                     "cores": sorted(os.sched_getaffinity(0)), "sandbox_version": SANDBOX_VERSION,
                                     "evaluator_version": EVALUATOR_VERSION,
                                     "suite_evaluator_version": (frozen[0].get("evaluator_version") if frozen else None)}
    manifest["single_arm_note"] = ("single = one model request frozen as returned (2026-10-20); fixed_workflow = the "
                                   "earlier iterated fixed-workflow arm, kept for reference")
    rsi_version = CodeVersion(**json.loads((run/"rsi/frozen_agent.json").read_text())["agent"])
    arms = {"single": (report, version, frozen),
            "rsi": (manifest["arms"]["rsi"], rsi_version, json.loads((run/"rsi/frozen_results.json").read_text()))}
    manifest["comparison"] = write_tables(run, arms)
    write_json(run/"manifest.json", manifest)
    write_summary(run, {**manifest, "arms": {k: manifest["arms"][k] for k in ("single", "rsi")}})
    sealed(run, manifest, run/"single")
    value = report["frozen_public"]
    return f"{run.name}: {report['one_shot_status']} public {round(value['pass_rate']*value['count'])}/{value['count']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("request", "rerequest", "restate", "repair", "score", "score-repair"))
    parser.add_argument("--max-output-tokens", type=int, default=128000,
                        help="rerequest: output cap for answers cut off at the run's cap (Opus 5.5 maximum: 128000)")
    parser.add_argument("--model-timeout", type=float, default=2400, help="rerequest: seconds per request")
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--parallel", type=int, default=6, help="request: concurrent model requests")
    parser.add_argument("--core", type=int, default=None, help="score: pin to this CPU core")
    args = parser.parse_args()
    runs = [project_path(r) for r in args.runs]
    if args.core is not None:
        os.sched_setaffinity(0, {args.core})

    def guarded(work, run):
        try:
            return work(run)
        except Exception as error:  # report and continue with the other runs
            return f"{run.name}: SKIPPED {type(error).__name__}: {error}"
    if args.stage in ("request", "rerequest", "restate", "repair"):
        work = {"request": request,
                "rerequest": lambda run: rerequest(run, args.max_output_tokens, args.model_timeout),
                "restate": lambda run: restate(run, args.model_timeout),
                "repair": lambda run: repair(run, args.model_timeout)}[args.stage]
        with ThreadPoolExecutor(args.parallel) as pool:
            for line in pool.map(lambda r: guarded(work, r), runs):
                print(line, flush=True)
    else:
        for run in runs:
            print(guarded(score if args.stage == "score" else score_repair, run), flush=True)


if __name__ == "__main__":
    sys.exit(main())
