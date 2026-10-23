#!/usr/bin/env python
"""Validate and summarize completed public agent runs; never call a model or solver.

Run after sourcing scripts/env.sh. All paths and artifacts stay inside PDE-RSI.
"""

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from pde_rsi.evaluator import EVALUATOR_VERSION, evaluate
from pde_rsi.solvers import Solution
from pde_rsi.storage import PROJECT_ROOT, doctor, project_path
from pde_rsi.tasks import FAMILIES, load_tasks


METRICS = ("absolute_l2", "relative_l2", "final_absolute_l2", "final_relative_l2")


def read_json(path):
    return json.loads(project_path(path).read_text())


def write_json(path, value):
    project_path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows):
    with project_path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def snapshot_valid(snapshot):
    expected = hashlib.sha256(json.dumps(snapshot["agent"], sort_keys=True).encode()).hexdigest()[:20]
    return snapshot["version_id"] == expected


def token_totals(records):
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    return {
        field: sum(row[field] for row in records) if all(type(row.get(field)) is int for row in records) else None
        for field in fields
    }


def audit_model_calls(directory):
    requests = sorted(directory.glob("model-call-*/request.json"))
    attempted = completed = 0
    usage = []
    routes = set()
    for path in requests:
        request = read_json(path)
        if request.get("url"):
            routes.add(request["url"])
        response_path = path.parent / "response.json"
        if not response_path.is_file():
            raise ValueError(f"Unfinished model request: {path.parent}")
        response = read_json(response_path)
        attempted += int(response.get("call_attempted", True))
        if "error" not in response:
            completed += 1
        raw = response.get("usage", {})
        usage.append({
            key: raw.get(key, raw.get(provider_key))
            for key, provider_key in (("prompt_tokens", "inputTokens"),
                                      ("completion_tokens", "outputTokens"),
                                      ("total_tokens", "totalTokens"))
        })
    return {"request_artifacts": len(requests), "attempted_calls": attempted,
            "response_artifacts": len(requests), "provider_responses": completed,
            "token_totals": token_totals(usage), "routes": sorted(routes)}


class Checks:
    def __init__(self):
        self.records = []

    def add(self, name, passed, detail=None):
        record = {"check": name, "passed": bool(passed)}
        if detail is not None:
            record["detail"] = detail
        self.records.append(record)

    @property
    def passed(self):
        return all(record["passed"] for record in self.records)


def collect_arm(label, run, name, report, manifest, tasks, checks, rounds, verify_npz):
    directory = project_path(run / name)
    prefix = f"{label}/{name}"
    results = read_json(directory / "frozen_results.json")
    initial = read_json(directory / "initial_agent.json")
    frozen = read_json(directory / "frozen_agent.json")
    ids = [row["task_id"] for row in results]
    expected_ids = set(tasks)
    checks.add(prefix + "/unique_public_coverage", len(ids) == 48 and len(set(ids)) == 48 and set(ids) == expected_ids)
    checks.add(prefix + "/only_public_splits", all(
        row["task_id"] in tasks and row["split"] == tasks[row["task_id"]].split
        and row["family"] == tasks[row["task_id"]].family for row in results
    ))
    checks.add(prefix + "/all_outputs_successful", len(results) == 48 and all(row["status"] == "ok" for row in results))
    checks.add(prefix + "/evaluator_version", all(row.get("evaluator_version") == EVALUATOR_VERSION for row in results))
    checks.add(prefix + "/finite_l2", all(
        isinstance(row.get(metric), (int, float)) and math.isfinite(row[metric]) and row[metric] >= 0
        for row in results for metric in METRICS
    ))
    checks.add(prefix + "/snapshot_hashes", snapshot_valid(initial) and snapshot_valid(frozen)
               and frozen["version_id"] == report["frozen_version_id"])
    checks.add(prefix + "/complete_rounds", report["requested_rounds"] == rounds and report["completed_rounds"] == rounds)
    budget = report["budget"]
    expected_calls = 72 + 48 * rounds
    checks.add(prefix + "/solver_slots", budget["solver_calls"] == expected_calls,
               {"expected": expected_calls, "actual": budget["solver_calls"]})
    checks.add(prefix + "/model_slots", budget["model_calls"] == rounds,
               {"expected": rounds, "actual": budget["model_calls"]})
    checks.add(prefix + "/resource_ceilings", budget["within_resource_ceilings"]
               and budget["solver_calls"] <= budget["max_solver_calls"]
               and budget["model_calls"] <= budget["max_model_calls"]
               and budget["total_cpu_seconds"] <= budget["max_cpu_seconds"]
               and budget["elapsed_wall_seconds"] <= budget["max_wall_seconds"])
    calls = [json.loads(line) for line in project_path(directory / "solve_calls.jsonl").read_text().splitlines()]
    checks.add(prefix + "/solver_ledger", len(calls) == budget["solver_calls"]
               and [row["call"] for row in calls] == list(range(1, len(calls) + 1))
               and all(row["task_id"] in tasks and row["split"] == tasks[row["task_id"]].split for row in calls))
    frozen_calls = [row for row in calls if row["phase"] == "frozen_public_evaluation"]
    checks.add(prefix + "/frozen_ledger_coverage", len(frozen_calls) == 48
               and {row["task_id"] for row in frozen_calls} == expected_ids)
    audit = audit_model_calls(directory)
    checks.add(prefix + "/model_audit", audit["attempted_calls"] == budget["model_calls"]
               and audit["request_artifacts"] == rounds and audit["provider_responses"] == rounds)
    checks.add(prefix + "/token_audit", audit["token_totals"] == token_totals(budget["model_usage"]))
    checks.add(prefix + "/output_token_cap", all(
        type(row.get("completion_tokens")) is int
        and row["completion_tokens"] <= budget["max_output_tokens_per_request"]
        for row in budget["model_usage"]
    ))
    if name == "single":
        checks.add(prefix + "/fixed_policy_and_memory", all(
            frozen["agent"][key] == initial["agent"][key]
            for key in ("improvement_workflow", "memory", "generation", "parent_id", "model")
        ))
    if verify_npz:
        mismatches = []
        for row in results:
            if row["task_id"] not in tasks or row["status"] != "ok":
                continue
            path = project_path(directory / "solutions" / f"{row['task_id']}.npz")
            if not path.is_file():
                mismatches.append({"task_id": row["task_id"], "field": "missing_npz"})
                continue
            solution = Solution.load_npz(path)
            repeated = evaluate(tasks[row["task_id"]], solution)
            for field in METRICS:
                if not math.isclose(repeated[field], row[field], rel_tol=5e-12, abs_tol=1e-14):
                    mismatches.append({"task_id": row["task_id"], "field": field})
            for field in ("nx", "nt", "accuracy_pass", "constraints_pass"):
                if repeated[field] != row[field]:
                    mismatches.append({"task_id": row["task_id"], "field": field})
        checks.add(prefix + "/npz_l2_replay", not mismatches, mismatches)
    return {"label": label, "name": name, "run": str(run), "report": report,
            "results": results, "initial": initial, "frozen": frozen,
            "audit": audit, "model": manifest["model_config"]["model"]}


def summary_row(arm):
    rows, report, audit = arm["results"], arm["report"], arm["audit"]
    budget = report["budget"]
    return {
        "model": arm["label"], "model_id": arm["model"], "arm": arm["name"],
        "cases": len(rows), "passes": sum(row["passed"] for row in rows),
        "train_passes": sum(row["passed"] for row in rows if row["split"] == "evolution"),
        "validation_passes": sum(row["passed"] for row in rows if row["split"] == "development"),
        **{"mean_" + metric: statistics.mean(row[metric] for row in rows) for metric in METRICS},
        "accepted_candidates": report["accepted_candidates"],
        "requested_rounds": report["requested_rounds"], "completed_rounds": report["completed_rounds"],
        "solver_call_allowance": budget["max_solver_calls"], "actual_solver_calls": budget["solver_calls"],
        "model_call_allowance": budget["max_model_calls"], "actual_model_calls": budget["model_calls"],
        "cpu_ceiling_seconds": budget["max_cpu_seconds"], "actual_cpu_seconds": budget["total_cpu_seconds"],
        "wall_ceiling_seconds": budget["max_wall_seconds"], "actual_wall_seconds": budget["elapsed_wall_seconds"],
        "model_wall_seconds": budget["model_wall_seconds"], **audit["token_totals"],
    }


def baseline_rows():
    paths = [PROJECT_ROOT / f"runs/agent-build-baseline-{split}-20261014/results.json"
             for split in ("evolution", "development")]
    if not all(path.is_file() for path in paths):
        return None
    return [row for path in paths for row in read_json(path)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt-run", type=Path, required=True)
    parser.add_argument("--claude-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-npz-verification", action="store_true",
                        help="Skip numerical replay of saved arrays; never invokes a solver")
    args = parser.parse_args()
    doctor()
    output = project_path(args.output)
    public = [task for split in ("evolution", "development")
              for task in load_tasks(project_path(PROJECT_ROOT / "benchmarks" / f"{split}.jsonl"))]
    tasks = {task.task_id: task for task in public}
    if len(public) != 48 or len(tasks) != 48:
        raise ValueError("Expected exactly 48 unique public tasks")
    hashes = {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted((PROJECT_ROOT / "src").rglob("*.py"))}
    package_hash = hashlib.sha256(project_path(PROJECT_ROOT / "datasets/huggingface/pde-problems/manifest.json").read_bytes()).hexdigest()
    checks, mains, continuations, manifests = Checks(), [], [], {}
    for label, raw in (("GPT-OSS", args.gpt_run), ("Claude", args.claude_run)):
        run = project_path(raw)
        manifest = read_json(run / "manifest.json")
        if set(manifest.get("arms", {})) != {"single", "rsi"} or "continuation" not in manifest:
            raise ValueError(f"Run is not complete yet: {run}")
        manifests[label] = manifest
        checks.add(label + "/source_hashes_current", manifest["source_sha256"] == hashes)
        checks.add(label + "/dataset_hash_current", manifest["dataset_manifest_sha256"] == package_hash)
        checks.add(label + "/evaluator_current", manifest["evaluator_version"] == EVALUATOR_VERSION)
        checks.add(label + "/no_sealed_usage_declared", manifest["sealed_cases_used"] is False)
        checks.add(label + "/task_manifest", set(manifest["task_ids"]) == {"evolution", "development"}
                   and all(len(ids) == 24 and len(set(ids)) == 24
                           and set(ids) == {task.task_id for task in public if task.split == split}
                           for split, ids in manifest["task_ids"].items()))
        configuration = manifest["configuration"]
        arms = []
        for name in ("single", "rsi"):
            report = read_json(run / name / "report.json")
            checks.add(label + "/" + name + "/manifest_report", report == manifest["arms"][name])
            arm = collect_arm(label, run, name, report, manifest, tasks, checks,
                              configuration["rounds"], not args.skip_npz_verification)
            mains.append(arm)
            arms.append(arm)
        checks.add(label + "/identical_initial_agents", arms[0]["initial"] == arms[1]["initial"])
        checks.add(label + "/main_matched_solver_and_model_calls", all(
            arms[0]["report"]["budget"][field] == arms[1]["report"]["budget"][field]
            for field in ("solver_calls", "model_calls", "max_solver_calls", "max_model_calls",
                          "max_cpu_seconds", "max_wall_seconds", "max_output_tokens_per_request",
                          "max_input_characters_per_request")
        ))
        branches = []
        for name in ("initial_policy", "evolved_policy"):
            report = read_json(run / "continuation" / name / "report.json")
            checks.add(label + "/continuation/" + name + "/manifest_report", report == manifest["continuation"][name])
            arm = collect_arm(label, run / "continuation", name, report, manifest, tasks, checks,
                              configuration["probe_rounds"], not args.skip_npz_verification)
            branches.append(arm)
            continuations.append(arm)
        before, after = (copy.deepcopy(arm["initial"]["agent"]) for arm in branches)
        initial_workflow, evolved_workflow = before.pop("improvement_workflow"), after.pop("improvement_workflow")
        checks.add(label + "/continuation/same_initial_solver_memory_model", before == after)
        checks.add(label + "/continuation/evolved_starts_from_rsi", branches[1]["initial"] == arms[1]["frozen"])
        checks.add(label + "/continuation/initial_workflow_control", initial_workflow == arms[0]["initial"]["agent"]["improvement_workflow"])
        checks.add(label + "/continuation/matched_call_and_resource_allowances", all(
            branches[0]["report"]["budget"][field] == branches[1]["report"]["budget"][field]
            for field in ("solver_calls", "model_calls", "max_solver_calls", "max_model_calls",
                          "max_cpu_seconds", "max_wall_seconds", "max_output_tokens_per_request")
        ))
        for branch in branches:
            checks.add(label + "/continuation/" + branch["name"] + "/workflow_memory_fixed", all(
                branch["initial"]["agent"][field] == branch["frozen"]["agent"][field]
                for field in ("improvement_workflow", "memory", "generation", "parent_id", "model")
            ))
        manifest["reporting_distinct_continuation_policies"] = initial_workflow != evolved_workflow
        resolved = {model for arm in arms + branches for model in arm["report"]["resolved_models"]}
        checks.add(label + "/resolved_identity_consistent_when_available", len(resolved) <= 1)
    checks.add("cross_model/main_equal_allowances", len({tuple(
        arm["report"]["budget"][field] for field in (
            "max_solver_calls", "max_model_calls", "max_cpu_seconds", "max_wall_seconds",
            "max_output_tokens_per_request", "max_input_characters_per_request")
    ) for arm in mains}) == 1)
    baseline = baseline_rows()
    if baseline is not None:
        checks.add("baseline/complete_public_successful_coverage", len(baseline) == 48
                   and {row["task_id"] for row in baseline} == set(tasks)
                   and all(row["status"] == "ok" and row.get("evaluator_version") == EVALUATOR_VERSION for row in baseline))
    aborted_notes = []
    for aborted_path in sorted((PROJECT_ROOT / "runs").glob("*agents*/attempt_status.json")):
        aborted = read_json(aborted_path)
        solver_calls = aborted.get("solver_calls")
        if solver_calls is None:
            solver_calls = sum(aborted.get("solver_calls_by_arm", {}).values())
        aborted_notes.append({
            "path": str(project_path(aborted_path)), "status": aborted.get("status"),
            "comparison_eligible": aborted.get("comparison_eligible"), "solver_calls": solver_calls,
            "attempted_requests": len(aborted.get("attempted_requests", [])),
            "completed_responses": len(aborted.get("completed_response_usage", [])),
            "completed_response_usage": aborted.get("completed_response_usage", []),
            "reason": aborted.get("reason"), "included_in_comparison_budgets": False,
        })
    totals = {
        "main": {"solver_calls": sum(arm["report"]["budget"]["solver_calls"] for arm in mains),
                 "model_calls": sum(arm["report"]["budget"]["model_calls"] for arm in mains),
                 "cpu_seconds": sum(arm["report"]["budget"]["total_cpu_seconds"] for arm in mains),
                 "wall_seconds": sum(arm["report"]["budget"]["elapsed_wall_seconds"] for arm in mains),
                 **token_totals([usage for arm in mains for usage in arm["report"]["budget"]["model_usage"]])},
        "continuations": {"solver_calls": sum(arm["report"]["budget"]["solver_calls"] for arm in continuations),
                          "model_calls": sum(arm["report"]["budget"]["model_calls"] for arm in continuations),
                          "cpu_seconds": sum(arm["report"]["budget"]["total_cpu_seconds"] for arm in continuations),
                          "wall_seconds": sum(arm["report"]["budget"]["elapsed_wall_seconds"] for arm in continuations),
                          **token_totals([usage for arm in continuations for usage in arm["report"]["budget"]["model_usage"]])},
    }
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "all_checks_passed": checks.passed,
        "checks": checks.records, "npz_replay_performed": not args.skip_npz_verification,
        "npz_replay_scope": "Accuracy, IC/BC, and output grids; execution resources come from original metered runs",
        "evaluator_version": EVALUATOR_VERSION, "public_task_ids": sorted(tasks), "sealed_cases_read": False,
        "runs": {label: {"path": str(project_path(args.gpt_run if label == "GPT-OSS" else args.claude_run)),
                         "model": value["model_config"]["model"], "protocol": value["model_config"]["protocol"],
                         "endpoint": value["model_config"]["base_url"], "source_sha256": value["source_sha256"],
                         "distinct_continuation_policies": value["reporting_distinct_continuation_policies"]}
                 for label, value in manifests.items()},
        "usage_totals": totals, "aborted_setup_pilots": aborted_notes,
        "remote_inference_compute": "Unmeasured; compare model-call and token allowances and observed token usage",
    }
    write_json(output / "protocol_checks.json", protocol)
    if not checks.passed:
        failures = [record["check"] for record in checks.records if not record["passed"]]
        project_path(output / "summary.md").write_text("Report validation failed. See protocol_checks.json.\n\n"
                                                        + "\n".join("- " + name for name in failures) + "\n")
        raise SystemExit("Protocol checks failed; comparison means were not exported.")
    summaries = [summary_row(arm) for arm in mains]
    write_csv(output / "comparison.csv", summaries)
    cases, families = [], []
    for arm in mains:
        for row in arm["results"]:
            requested = arm["frozen"]["agent"]["solver"][row["family"]]
            cases.append({"model": arm["label"], "model_id": arm["model"], "arm": arm["name"],
                          "task_id": row["task_id"], "family": row["family"], "split": row["split"],
                          "status": row["status"], **{metric: row[metric] for metric in METRICS},
                          "passed": row["passed"], "requested_nx": requested["nx"], "requested_nt": requested["nt"],
                          "actual_nx": row["nx"], "actual_nt": row["nt"],
                          "grading": requested["grading"], "advection": requested["advection"],
                          "method": row["method"], "wall_seconds": row["wall_seconds"],
                          "cpu_seconds": row["cpu_seconds"], "peak_memory_mb": row["peak_memory_mb"]})
        for family in FAMILIES:
            rows = [row for row in arm["results"] if row["family"] == family]
            families.append({"model": arm["label"], "model_id": arm["model"], "arm": arm["name"],
                             "family": family, "cases": len(rows), "passes": sum(row["passed"] for row in rows),
                             "train_passes": sum(row["passed"] for row in rows if row["split"] == "evolution"),
                             "validation_passes": sum(row["passed"] for row in rows if row["split"] == "development"),
                             **{"mean_" + metric: statistics.mean(row[metric] for row in rows) for metric in METRICS}})
    write_csv(output / "per_case_l2.csv", cases)
    write_csv(output / "per_family_l2.csv", families)
    text = [
        "Four model/arm comparisons use all 48 public manufactured cases: 24 train/evolution and 24 validation/development.",
        "All source, coverage, accounting, and continuation checks passed. "
        + ("Saved solution arrays reproduce the reported L2 metrics." if not args.skip_npz_verification else "Saved-array numerical replay was skipped."),
        "", "| Model | Arm | Passes | Absolute L2 | Relative L2 | Final absolute L2 | Final relative L2 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        text.append(f"| {row['model']} | {row['arm']} | {row['passes']}/48 | "
                    + " | ".join(f"{row['mean_' + metric]:.6g}" for metric in METRICS) + " |")
    text += ["", "Numbers are arithmetic means over 48 cases. Space-time norms use the declared 65 spatial and 17 temporal evaluation nodes; "
             "Poisson uses a spatial norm. Per-family and split details are in the CSV artifacts.", "",
             "| Model | Arm | Solver calls / allowance | Model calls / allowance | CPU seconds / ceiling | Wall seconds / ceiling | Input / output tokens |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summaries:
        text.append(f"| {row['model']} | {row['arm']} | {row['actual_solver_calls']}/{row['solver_call_allowance']} | "
                    f"{row['actual_model_calls']}/{row['model_call_allowance']} | {row['actual_cpu_seconds']:.2f}/{row['cpu_ceiling_seconds']:g} | "
                    f"{row['actual_wall_seconds']:.2f}/{row['wall_ceiling_seconds']:g} | {row['prompt_tokens']} / {row['completion_tokens']} |")
    text += ["", "CPU/GPU limits describe local solver/controller resources: one CPU thread and zero local GPUs. "
             "Remote inference compute is unmeasured. The arms receive equal solver, model-call, output-token, input-character, CPU, and wall ceilings; "
             "their actual numerical grid and token costs can differ. Poisson ignores requested nt; Burgers may increase nt to satisfy its CFL bound. "
             "per_case_l2.csv records both requested and actual grids.", "",
             "| Model | Initial-policy future gain | Evolved-policy future gain | Evolved minus initial | Distinct workflows |",
             "| --- | ---: | ---: | ---: | --- |"]
    for label, manifest in manifests.items():
        gains = {
            name: report["initial_validation"]["macro_relative_l2"] - report["selected_validation"]["macro_relative_l2"]
            for name, report in manifest["continuation"].items()
        }
        text.append(f"| {label} | {gains['initial_policy']:.6g} | {gains['evolved_policy']:.6g} | "
                    f"{gains['evolved_policy'] - gains['initial_policy']:.6g} | {manifest['reporting_distinct_continuation_policies']} |")
    text += ["", "Continuation branches start from identical solver settings, memory, and model, changing only the workflow instruction. "
             "The instruction and memory remain fixed during continuation. These are exploratory future gains on repeatedly queried development cases. "
             "One paired continuation and this small manufactured dataset do not establish scientific RSI or unseen-task generalization.", ""]
    if baseline is not None:
        text += ["The initial numerical baseline, evaluated separately from agent budgets:", "",
                 "| Passes | Absolute L2 | Relative L2 | Final absolute L2 | Final relative L2 |",
                 "| ---: | ---: | ---: | ---: | ---: |",
                 f"| {sum(row['passed'] for row in baseline)}/48 | "
                 + " | ".join(f"{statistics.mean(row[metric] for row in baseline):.6g}" for metric in METRICS) + " |", ""]
    text += [f"Main totals: {totals['main']['solver_calls']} solver calls, {totals['main']['model_calls']} model calls," 
             f"{totals['main']['cpu_seconds']:.2f} local CPU seconds. Continuation totals: "
             f"{totals['continuations']['solver_calls']} solver calls, {totals['continuations']['model_calls']} model calls,"
             f"{totals['continuations']['cpu_seconds']:.2f} local CPU seconds.", ""]
    for aborted_note in aborted_notes:
        text += [f"Aborted setup run {Path(aborted_note['path']).parent.name}: "
                 f"{aborted_note['attempted_requests']} attempted requests, "
                 f"{aborted_note['completed_responses']} completed responses, and {aborted_note['solver_calls']} solver calls. "
                 "Its artifacts and response usage are retained separately and excluded from comparison and continuation budgets.", ""]
    for label, manifest in manifests.items():
        config = manifest["model_config"]
        text.append(f"{label} source: {project_path(args.gpt_run if label == 'GPT-OSS' else args.claude_run)}. "
                    f"Requested model: {config['model']}; route: {config['base_url']}; protocol: {config['protocol']}.")
    project_path(output / "summary.md").write_text("\n".join(text) + "\n")
    print(json.dumps({"output": str(output), "all_checks_passed": True,
                      "main_rows": len(summaries), "per_case_rows": len(cases),
                      "per_family_rows": len(families)}, indent=2))


if __name__ == "__main__":
    main()
