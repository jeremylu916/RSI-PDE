"""Three-arm evolution, frozen final evaluation, and paired continuation probes."""

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy

from .evaluator import EVALUATOR_VERSION, acceptance_gate, summarize
from .policy import AgentVersion, ImprovementPolicy, propose
from .runner import Budget, BudgetExceeded, run_suite
from .tasks import FAMILIES, load_tasks
from .storage import project_path, storage_environment


def write_json(path, data):
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def append_event(path, data):
    with project_path(path).open("a") as stream:
        stream.write(json.dumps(data, sort_keys=True, allow_nan=False) + "\n")


def selected_tasks(root, split, limit_per_family=0):
    root = project_path(root)
    path = project_path(Path(root)/("evaluator_private" if split == "sealed" else "benchmarks")/f"{split}.jsonl")
    expected = json.loads((Path(root)/"benchmarks/manifest.json").read_text())["files"][split]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
        raise ValueError(f"Task file differs from benchmark manifest: {split}")
    tasks = load_tasks(path)
    if len(tasks) != expected["cases"] or any(t.split != split for t in tasks):
        raise ValueError(f"Task count or split differs from benchmark manifest: {split}")
    if limit_per_family:
        tasks = [task for family in FAMILIES for task in [t for t in tasks if t.family == family][:limit_per_family]]
    if set(t.family for t in tasks) != set(FAMILIES):
        raise ValueError("Experiments require coverage of all six families")
    return tasks


def provenance(root):
    root = project_path(root)
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((root/"src").rglob("*.py"))}
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__,
            "python_executable": sys.executable, "python_prefix": sys.prefix,
            "storage_paths": storage_environment(),
            "environment_lock_sha256": (hashlib.sha256((root/"configs/conda-explicit.txt").read_bytes()).hexdigest()
                                         if (root/"configs/conda-explicit.txt").is_file() else None),
            "source_sha256": source_hashes, "evaluator_version": EVALUATOR_VERSION,
            "benchmark_manifest": json.loads((root/"benchmarks/manifest.json").read_text())}


def evolve_arm(directory, version, evolution, development, generations, recursive, budget, reserve_final=0,
               single_session=False):
    directory = project_path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    budget.ledger_path = directory/"solve_calls.jsonl"
    budget.phase = "initial_development"
    feedback = run_suite(development, version, budget)
    write_json(directory/"initial_development.json", feedback)
    write_json(directory/"generation-000.json", {"version_id": version.version_id, "agent": version.to_dict()})
    initial_summary = summarize(feedback)
    attempted = accepted = 0
    starting_generation = version.generation
    session_state = (copy.deepcopy(version.improvement), copy.deepcopy(version.memory), version.parent_id)
    for iteration in range(1, generations+1):
        try:
            budget.reserve(len(evolution)+len(development)+reserve_final)
        except BudgetExceeded:
            append_event(directory/"events.jsonl", {"iteration": iteration, "event": "budget_exhausted"})
            break
        try:
            budget.phase = f"evolution_{iteration:03d}"
            evolution_feedback = summarize(run_suite(evolution, version, budget))
            candidate = propose(version, evolution_feedback, recursive=recursive)
            if single_session:
                # The same fixed numerical search runs inside one session. Commit
                # only solver settings; policy, reusable memory, and lineage stay fixed.
                candidate.generation = starting_generation
                candidate.improvement, candidate.memory, candidate.parent_id = copy.deepcopy(session_state)
            write_json(directory/f"proposal-{iteration:03d}.json", {"version_id": candidate.version_id, "agent": candidate.to_dict()})
            budget.phase = f"development_{iteration:03d}"
            candidate_feedback = run_suite(development, candidate, budget)
        except BudgetExceeded:
            append_event(directory/"events.jsonl", {"iteration": iteration, "event": "wall_budget_exhausted"})
            break
        attempted += 1
        gate = acceptance_gate(feedback, candidate_feedback)
        write_json(directory/f"development-{iteration:03d}.json", candidate_feedback)
        event = {"iteration": iteration, "parent_id": version.version_id, "candidate_id": candidate.version_id,
                 "gate": gate, "evolution_summary": evolution_feedback,
                 "candidate_development_summary": summarize(candidate_feedback)}
        if gate["accepted"]:
            version, feedback = candidate, candidate_feedback
            accepted += 1
        event["active_version_id"] = version.version_id
        append_event(directory/"events.jsonl", event)
        write_json(directory/f"generation-{iteration:03d}.json", {"version_id": version.version_id, "agent": version.to_dict()})
    snapshot = {"version_id": version.version_id, "agent": version.to_dict()}
    write_json(directory/"frozen_agent.json", snapshot)
    report = {"initial_development": initial_summary, "frozen_development": summarize(feedback),
              "frozen_version_id": version.version_id, "accepted_generations": version.generation,
              "accepted_updates_in_run": version.generation-starting_generation,
              "accepted_candidates": accepted, "single_session": single_session,
              "attempted_updates": attempted, "budget": budget.report()}
    write_json(directory/"report.json", report)
    return report


def compare(root, output, generations=4, max_calls=240, wall_seconds=600, limit_per_family=0,
            single_mode="session", cpu_seconds=None):
    root, output = project_path(root), project_path(output)
    if single_mode not in ("session", "static"):
        raise ValueError("single_mode must be session or static")
    evolution = selected_tasks(root, "evolution", limit_per_family)
    development = selected_tasks(root, "development", limit_per_family)
    required_calls = 2*len(development)+(len(evolution)+len(development))*generations
    if max_calls < required_calls:
        raise ValueError(f"Budget must cover all requested updates and final reservation: {required_calls} calls")
    output.mkdir(parents=True, exist_ok=False)
    metadata = provenance(root)
    metadata.update({"kind": "three_arm_comparison", "frozen": True, "final_evaluated": False,
                     "configuration": {"generations": generations, "max_solver_calls": max_calls,
                                       "max_wall_seconds": wall_seconds, "limit_per_family": limit_per_family,
                                       "max_cpu_seconds": wall_seconds if cpu_seconds is None else cpu_seconds,
                                       "single_mode": single_mode},
                     "task_ids": {"evolution": [t.task_id for t in evolution], "development": [t.task_id for t in development]},
                     "interpretation": "deterministic numerical controls; equal resource ceilings and candidate slots; CPU/grid costs can differ; no LLM RSI claim",
                     "arms": {}})
    write_json(output/"manifest.json", metadata)
    for arm in ("single", "fixed", "rsi"):
        budget = Budget(max_calls, wall_seconds, max_cpu_seconds=cpu_seconds)
        metadata["arms"][arm] = evolve_arm(output/arm, AgentVersion(), evolution, development,
                                           generations=0 if arm == "single" and single_mode == "static" else generations,
                                           recursive=arm == "rsi", budget=budget,
                                           reserve_final=len(development), single_session=arm == "single")
        write_json(output/"manifest.json", metadata)
    reports = metadata["arms"].values()
    metadata["matched_actual_solve_calls"] = len({r["budget"]["solver_calls"] for r in reports}) == 1
    metadata["complete_requested_searches"] = all(
        r["attempted_updates"] == (0 if arm == "single" and single_mode == "static" else generations)
        for arm, r in metadata["arms"].items())
    metadata["all_within_resource_ceilings"] = all(r["budget"]["within_resource_ceilings"] for r in reports)
    write_json(output/"manifest.json", metadata)
    write_comparison_table(output/"development_comparison.csv", metadata["arms"])
    return metadata


def write_comparison_table(path, reports, key="frozen_development"):
    lines = ["arm,accepted_generations,accepted_candidates,attempted_updates,pass_rate,macro_relative_l2,solver_calls,elapsed_wall_seconds,total_cpu_seconds"]
    for arm, report in reports.items():
        metric = report[key]
        lines.append(f"{arm},{report['accepted_generations']},{report.get('accepted_candidates', 0)},{report['attempted_updates']},"
                     f"{metric['pass_rate']:.8g},{metric['macro_relative_l2']:.8g},"
                     f"{report['budget']['solver_calls']},{report['budget']['elapsed_wall_seconds']:.8g},"
                     f"{report['budget'].get('total_cpu_seconds', 0):.8g}")
    project_path(path).write_text("\n".join(lines)+"\n")


def finalize(root, run_directory):
    """Explicit, once-per-run evaluation. Freezes are never updated using final feedback.

    This is protocol separation for trusted code, not a security boundary against a
    malicious agent sharing the same Unix user. A production evaluator needs another
    Unix account/service and a distinct, inaccessible sealed dataset.
    """
    root, run_directory = project_path(root), project_path(run_directory)
    if (run_directory/"FINAL_CONSUMED.json").exists():
        raise FileExistsError("This run has already consumed its final evaluation")
    metadata = json.loads((run_directory/"manifest.json").read_text())
    if metadata["kind"] != "three_arm_comparison" or not metadata["frozen"] or len(metadata["arms"]) != 3:
        raise ValueError("Final evaluation requires a complete frozen three-arm comparison")
    current_provenance = provenance(root)
    if current_provenance["source_sha256"] != metadata["source_sha256"]:
        raise ValueError("Source changed after freezing; rerun comparison before final evaluation")
    for field in ("python", "numpy", "scipy", "environment_lock_sha256"):
        if current_provenance.get(field) != metadata.get(field):
            raise ValueError(f"Runtime environment changed after freezing: {field}")
    if json.loads((root/"benchmarks/manifest.json").read_text()) != metadata["benchmark_manifest"]:
        raise ValueError("Benchmark manifest changed after freezing")
    configuration = metadata["configuration"]
    tasks = selected_tasks(root, "sealed", configuration["limit_per_family"])
    if hashlib.sha256((root/"evaluator_private/sealed.jsonl").read_bytes()).hexdigest() != metadata["benchmark_manifest"]["files"]["sealed"]["sha256"]:
        raise ValueError("Sealed task file changed after freezing")
    snapshots = {}
    for arm, report in metadata["arms"].items():
        record = json.loads((run_directory/arm/"frozen_agent.json").read_text())
        version = AgentVersion.from_dict(record["agent"])
        if version.version_id != report["frozen_version_id"] or record["version_id"] != version.version_id:
            raise ValueError("Frozen agent snapshot changed")
        snapshots[arm] = version
        budget_report = report["budget"]
        if (budget_report["solver_calls"]+len(tasks) > budget_report["max_solver_calls"] or
                budget_report["elapsed_wall_seconds"] >= budget_report["max_wall_seconds"] or
                budget_report.get("total_cpu_seconds", 0) >= budget_report.get("max_cpu_seconds", float("inf"))):
            raise BudgetExceeded("Reserved final budget unavailable")
    # Exclusive creation records consumption even if execution fails partway through.
    with (run_directory/"FINAL_CONSUMED.json").open("x") as stream:
        json.dump({"time_utc": datetime.now(timezone.utc).isoformat(), "task_ids": [t.task_id for t in tasks]}, stream)
    for arm, version in snapshots.items():
        report = metadata["arms"][arm]
        previous = report["budget"]
        budget = Budget(previous["max_solver_calls"]-previous["solver_calls"],
                        previous["max_wall_seconds"]-previous["elapsed_wall_seconds"],
                        max_cpu_seconds=(previous.get("max_cpu_seconds", previous["max_wall_seconds"])
                                         -previous.get("total_cpu_seconds", 0)),
                        ledger_path=run_directory/arm/"solve_calls.jsonl", phase="final")
        budget.solver_calls = previous["solver_calls"]
        budget.max_solver_calls = previous["max_solver_calls"]
        results = run_suite(tasks, version, budget)
        write_json(run_directory/arm/"final_results.json", results)
        report["final"] = summarize(results)
        current = budget.report()
        for field in ("elapsed_wall_seconds", "child_wall_seconds", "parent_cpu_seconds", "child_cpu_seconds", "total_cpu_seconds"):
            report["budget"][field] = report["budget"].get(field, 0)+current[field]
        report["budget"]["solver_calls"] = current["solver_calls"]
        report["budget"]["within_resource_ceilings"] = current["within_resource_ceilings"]
        write_json(run_directory/arm/"report.json", report)
    metadata["final_evaluated"] = True
    metadata["all_within_resource_ceilings"] = all(
        report["budget"]["within_resource_ceilings"] for report in metadata["arms"].values())
    write_json(run_directory/"manifest.json", metadata)
    write_comparison_table(run_directory/"final_comparison.csv", metadata["arms"], key="final")
    return metadata


def continuation_probe(root, snapshot_path, output, generations=2, max_calls=120, wall_seconds=300, limit_per_family=0,
                       cpu_seconds=None):
    """Start from identical solver+memory; vary only the improvement policy.

    These are development/evolution measurements and must be labeled as such.
    Repeated probes are selection. Sealed data are never loaded by this function.
    """
    root, output = project_path(root), project_path(output)
    starting_record = json.loads(project_path(snapshot_path).read_text())
    starting = AgentVersion.from_dict(starting_record["agent"])
    if starting.version_id != starting_record["version_id"]:
        raise ValueError("Invalid source snapshot hash")
    evolution = selected_tasks(root, "evolution", limit_per_family)
    development = selected_tasks(root, "development", limit_per_family)
    if max_calls < len(development)+(len(evolution)+len(development))*generations:
        raise ValueError("Continuation budget must cover all requested paired updates")
    output.mkdir(parents=True, exist_ok=False)
    metadata = provenance(root)
    metadata.update({"kind": "continuation_probe", "starting_version_id": starting.version_id,
                     "configuration": {"generations": generations, "max_solver_calls": max_calls,
                                       "max_wall_seconds": wall_seconds, "limit_per_family": limit_per_family,
                                       "max_cpu_seconds": wall_seconds if cpu_seconds is None else cpu_seconds},
                     "interpretation": "paired development probe; evidence of future policy behavior, not sealed-test RSI evidence",
                     "arms": {}})
    write_json(output/"starting_agent.json", starting_record)
    for arm in ("initial_policy", "evolved_policy"):
        version = copy.deepcopy(starting)
        if arm == "initial_policy":
            version.improvement = ImprovementPolicy()
        # Hold both policy definitions fixed during the continuation. Each one drives
        # its own future solver proposals; recursively modifying again would confound
        # the comparison of the accepted meta-policy to the initial meta-policy.
        metadata["arms"][arm] = evolve_arm(output/arm, version, evolution, development,
                                           generations, recursive=False,
                                           budget=Budget(max_calls, wall_seconds, max_cpu_seconds=cpu_seconds))
    reports = metadata["arms"]
    metadata["matched_actual_solve_calls"] = len({r["budget"]["solver_calls"] for r in reports.values()}) == 1
    metadata["complete_paired_continuations"] = all(r["attempted_updates"] == generations for r in reports.values())
    metadata["future_error_reduction"] = {
        arm: report["initial_development"]["macro_relative_l2"]-report["frozen_development"]["macro_relative_l2"]
        for arm, report in reports.items()
    }
    metadata["evolved_policy_advantage"] = (metadata["future_error_reduction"]["evolved_policy"]-
                                            metadata["future_error_reduction"]["initial_policy"])
    write_json(output/"manifest.json", metadata)
    write_comparison_table(output/"continuation_comparison.csv", reports)
    return metadata
