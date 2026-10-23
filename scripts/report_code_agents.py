"""Consolidate code-writing agent comparisons and re-verify saved solutions.

Usage: bash scripts/run.sh is not required; run with the project interpreter after
sourcing scripts/env.sh:
    python scripts/report_code_agents.py --runs runs/code-agents-* --output runs/code-agent-results
"""

import argparse
import csv
import json
from pathlib import Path

from pde_rsi.agent_runs import METRICS
from pde_rsi.evaluator import EVALUATOR_VERSION, evaluate
from pde_rsi.experiment import selected_tasks
from pde_rsi.solvers import Solution
from pde_rsi.storage import PROJECT_ROOT, project_path
from pde_rsi.tasks import load_group_tasks, load_wave_tasks

# Runs before the rename recorded the classical start as "seed".
START_LABELS = {"seed": "classical", "classical": "classical", "scratch": "scratch"}
MODEL_NAMES = {"openai.gpt-oss-120b-1:0": "GPT-OSS", "us.anthropic.claude-sonnet-4-5-20250929-v1:0": "Claude",
               "us.anthropic.claude-opus-5-5": "Claude Opus 5.5", "us.openai.gpt-6.1-sol": "GPT-6.1",
               "us.openai.gpt-5.5": "GPT-5.5"}


def tokens(directory):
    """Input/output tokens from the raw provider responses, counting prompt-cache input as input."""
    prompt = completion = 0
    for path in directory.glob("model-call-*/response.json"):
        usage = json.loads(path.read_text()).get("usage") or {}
        if "inputTokens" in usage:
            prompt += sum(usage.get(k) or 0 for k in ("inputTokens", "cacheReadInputTokens", "cacheWriteInputTokens"))
            completion += usage.get("outputTokens") or 0
        else:
            prompt += usage.get("prompt_tokens") or 0
            completion += usage.get("completion_tokens") or 0
    return prompt, completion


def verify_solutions(directory, results, tasks, grid="fine", evaluator=None):
    """Re-evaluate every saved frozen NPZ on the grid the run was scored with; compare with the recorded metrics."""
    checked = mismatched = 0
    for row in results:
        if row["status"] != "ok":
            continue
        name = row.get("case_id", row["task_id"])  # identification runs save solutions by case ID
        solution = Solution.load_npz(directory/"solutions"/f"{name}.npz")
        metrics = (evaluator(tasks[row["task_id"]], solution) if evaluator is not None
                   else evaluate(tasks[row["task_id"]], solution, grid=grid))
        checked += 1
        if any(abs(metrics[key]-row[key]) > 1e-12*max(1.0, abs(row[key])) for key in METRICS):
            mismatched += 1
    return checked, mismatched


def per_round(directory, report):
    """Active validation score and resources after each round, for the score-vs-round curve."""
    events = [json.loads(line) for line in (directory/"events.jsonl").read_text().splitlines()] \
        if (directory/"events.jsonl").exists() else []
    calls = [json.loads(line) for line in (directory/"solve_calls.jsonl").read_text().splitlines()]
    active = report["initial_validation"]["macro_relative_l2"]
    active_pass = report["initial_validation"]["pass_rate"]
    rows = [{"round": 0, "active_validation_macro_relative_l2": active, "active_validation_pass_rate": active_pass,
             "decision": "start", "solves": sum(c["phase"] == "initial_validation" for c in calls),
             "solve_cpu_seconds": sum(c["cpu_seconds"] for c in calls if c["phase"] == "initial_validation"),
             "solve_wall_seconds": sum(c["wall_seconds"] for c in calls if c["phase"] == "initial_validation"),
             "input_tokens": 0, "output_tokens": 0}]
    last = max([e.get("iteration") or 0 for e in events] or [0])
    started = (directory/"initial_validation.json").stat().st_mtime
    rows[0]["elapsed_minutes"] = 0.0
    if report.get("one_shot"):  # the single run: one request, frozen as returned
        frozen = report["selected_validation"]
        usage = (report["budget"]["model_usage"] or [{}])[-1]
        rows.append({"round": 1, "elapsed_minutes": report["budget"]["model_wall_seconds"]/60,
                     "active_validation_macro_relative_l2": frozen["macro_relative_l2"],
                     "active_validation_pass_rate": frozen["pass_rate"], "decision": report["one_shot_status"],
                     "solves": 0, "solve_cpu_seconds": 0.0, "solve_wall_seconds": 0.0,
                     "input_tokens": usage.get("prompt_tokens") or 0,
                     "output_tokens": usage.get("completion_tokens") or 0})
        return rows
    # Time-budgeted arms allow up to max_rounds; report only the rounds that ran.
    for iteration in range(1, min(report["requested_rounds"], max(last, 1))+1):
        round_events = [e for e in events if e.get("iteration") == iteration]
        decision = "not run"
        usage = {}
        for event in round_events:
            if "gate" in event:
                decision = "accepted" if event["gate"]["accepted"] else "rejected"
                usage = event.get("model_usage") or {}
                if event["gate"]["accepted"]:
                    active = event["candidate_validation"]["macro_relative_l2"]
                    active_pass = event["candidate_validation"]["pass_rate"]
            else:
                decision = event["event"]
                usage = event.get("usage") or {}
        phases = {f"evolution_{iteration:03d}", f"candidate_validation_{iteration:03d}"}
        selected = [c for c in calls if c["phase"] in phases]
        marker = directory/f"candidate-validation-{iteration:03d}.json"
        evolution = directory/f"evolution-{iteration:03d}.json"
        stamp = marker if marker.exists() else evolution if evolution.exists() else None
        rows.append({"round": iteration, "elapsed_minutes": (stamp.stat().st_mtime-started)/60 if stamp else None,
                     "active_validation_macro_relative_l2": active,
                     "active_validation_pass_rate": active_pass, "decision": decision, "solves": len(selected),
                     "solve_cpu_seconds": sum(c["cpu_seconds"] for c in selected),
                     "solve_wall_seconds": sum(c["wall_seconds"] for c in selected),
                     "input_tokens": usage.get("prompt_tokens") or 0,
                     "output_tokens": usage.get("completion_tokens") or 0})
    return rows


def sandbox_solve_seconds(directory):
    """In-sandbox solve() time over every evaluation an arm ran; a CPU proxy for single-threaded code."""
    total = 0.0
    for path in directory.glob("*.json"):
        if path.name.startswith(("initial_validation", "evolution-", "self-test-", "candidate-validation-",
                                 "frozen_results")):
            total += sum(row.get("runtime_seconds") or 0.0 for row in json.loads(path.read_text()))
    return total


def identification_scores(directory):
    """Class accuracy of the frozen agent on identification runs (empty for ordinary runs)."""
    frozen = json.loads((directory/"frozen_results.json").read_text())
    if not frozen or "family_correct" not in frozen[0]:
        return {}
    scores = {}
    for split in ("evolution", "development"):
        rows = [r for r in frozen if r["split"] == split]
        scores[f"class_accuracy_{split}"] = sum(bool(r["family_correct"]) for r in rows)/len(rows)
    methods = [r.get("method") for r in frozen]
    if any(methods):
        scores["pinn_fraction"] = sum(m == "pinn" for m in methods)/len(methods)
    return scores


def fmt(value):
    return "n/a" if value is None else f"{value:.4g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    tasks = {t.task_id: t for split in ("evolution", "development")
             for t in selected_tasks(PROJECT_ROOT, split)+load_wave_tasks(PROJECT_ROOT, split)
             +load_group_tasks(PROJECT_ROOT, split)}
    # Identification v2 / student truths are generated per case (two per family covers both protocols).
    from pde_rsi.identification import make_cases
    tasks.update({case.task_id: case.truth for split in ("evolution", "development")
                  for case in make_cases(PROJECT_ROOT, split)})
    from pde_rsi import pdebench
    tasks.update({case.task_id: case.truth for split in ("evolution", "development")
                  for case in pdebench.make_cases(split)})
    rows, family_rows, continuation_rows, round_rows, checks = [], [], [], [], {}
    for run in map(project_path, args.runs):
        manifest = json.loads((run/"manifest.json").read_text())
        model = MODEL_NAMES.get(manifest["model_config"]["model"], manifest["model_config"]["model"])
        start = START_LABELS[manifest["start"]]
        reference = manifest["starting_reference"]
        families = "+".join(manifest.get("families", ["1d"]))
        rows.append({"families": families, "start": start, "model": model, "arm": "starting programs",
                     "cases": reference["count"],
                     "passes": round(reference["pass_rate"]*reference["count"]),
                     **{key: reference["mean_"+key] for key in METRICS}})
        run_checks = {key: manifest.get(key) for key in (
            "matched_actual_solver_calls", "matched_actual_model_calls", "resolved_model_identity_consistent",
            "all_requested_rounds_complete", "all_within_resource_ceilings", "continuation_matched_actual_calls",
            "continuation_distinct_workflows", "sealed_cases_used")}
        run_checks["sandbox_isolated"] = manifest["sandbox"]["isolated"]
        # "single" is the one-shot single run; an earlier iterated fixed-workflow arm is not reported.
        for arm, report in ((a, manifest["arms"][a]) for a in ("single", "rsi")):
            value = report["frozen_public"]
            prompt, completion = tokens(run/arm)
            rows.append({"families": families, "start": start, "model": model, "arm": arm, "cases": value["count"],
                         "passes": round(value["pass_rate"]*value["count"]),
                         **{key: value["mean_"+key] for key in METRICS},
                         "accepted": report["accepted_candidates"], "rounds": report["requested_rounds"],
                         "completed_rounds": report["completed_rounds"],
                         "changed_families": " ".join(report.get("changed_from_start", report.get("changed_from_seed", []))),
                         "solver_calls": report["budget"]["solver_calls"],
                         "model_calls": report["budget"]["model_calls"],
                         "cpu_seconds": report["budget"]["total_cpu_seconds"],
                         # Before sandbox bwrap-v3-cpu, sandboxed CPU was not metered (recorded as ~0).
                         "cpu_metering_valid": "sandbox_cpu_seconds" in report["budget"],
                         "sandbox_solve_seconds": sandbox_solve_seconds(run/arm),
                         "wall_seconds": report["budget"]["elapsed_wall_seconds"],
                         "input_tokens": prompt, "output_tokens": completion,
                         **identification_scores(run/arm)})
            for family, item in value["families"].items():
                family_rows.append({"families": families, "start": start, "model": model, "arm": arm, "family": family,
                                    "pass_rate": item["pass_rate"],
                                    **{"mean_"+key: item["mean_"+key] for key in METRICS}})
            for item in per_round(run/arm, report):
                round_rows.append({"families": families, "start": start, "model": model, "arm": arm, **item})
            results = json.loads((run/arm/"frozen_results.json").read_text())
            run_tasks = tasks
            if manifest.get("task_protocol") == "pdebench":  # score against the run's own posing of the cases
                settings = manifest["task_protocol_metadata"]
                run_tasks = {c.task_id: c.truth for split in ("evolution", "development") for c in pdebench.make_cases(
                    split, settings.get("noise_fraction", 0.01), settings.get("hide_time_dependence", False))}
            run_checks[f"{arm}_npz_checked"], run_checks[f"{arm}_npz_mismatched"] = verify_solutions(
                run/arm, results, run_tasks, "fine" if manifest["evaluator_version"] == EVALUATOR_VERSION else "legacy",
                pdebench.evaluate if manifest.get("task_protocol") == "pdebench" else None)
        for name, report in manifest.get("continuation", {}).items():
            before = report["initial_validation"]["macro_relative_l2"]
            after = report["selected_validation"]["macro_relative_l2"]
            continuation_rows.append({"families": families, "start": start, "model": model, "branch": name, "initial_macro": before,
                                      "selected_macro": after, "future_gain": before-after,
                                      "accepted": report["accepted_candidates"], "rounds": report["requested_rounds"],
                                      "advantage": manifest["continuation_policy_advantage"]})
        checks[run.name] = run_checks
    for name, records in (("comparison.csv", rows), ("per_family_l2.csv", family_rows),
                          ("continuation.csv", continuation_rows), ("per_round.csv", round_rows)):
        with (output/name).open("w", newline="") as stream:
            fields = list(dict.fromkeys(key for record in records for key in record))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    (output/"protocol_checks.json").write_text(json.dumps(checks, indent=2)+"\n")
    lines = ["| Start | Model | Arm | Passes | Absolute L2 | Relative L2 | Final absolute L2 | Final relative L2 | Accepted / rounds |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        accepted = f"{row['accepted']}/{row['rounds']}" if "accepted" in row else "—"
        lines.append(f"| {row['start']} | {row['model']} | {row['arm']} | {row['passes']}/{row['cases']} | "
                     + " | ".join(fmt(row[key]) for key in METRICS) + f" | {accepted} |")
    lines += ["", "| Start | Model | Branch | Initial macro rel. L2 | Selected | Future gain | Accepted |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for row in continuation_rows:
        lines.append(f"| {row['start']} | {row['model']} | {row['branch']} | {fmt(row['initial_macro'])} | "
                     f"{fmt(row['selected_macro'])} | {fmt(row['future_gain'])} | {row['accepted']}/{row['rounds']} |")
    (output/"summary.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"output": str(output), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
