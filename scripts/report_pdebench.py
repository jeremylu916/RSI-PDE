"""Tables for pdebench runs: per tex case, per split, and the sealed test (single run vs RSI).

Run after scripts/report_code_agents.py (which writes the shared CSVs and verifies the NPZs) and
scripts/pdebench_sealed_test.py:
    python scripts/report_pdebench.py runs/code-agent-results-pdebench-20261017
"""

import argparse
import json
import math
from pathlib import Path

from pde_rsi.pdebench import CLASS, TITLES

MODELS = {"us.anthropic.claude-sonnet-4-5-20250929-v1:0": "Sonnet 4.5", "openai.gpt-oss-120b-1:0": "GPT-OSS",
          "us.openai.gpt-5.5": "GPT-5.5", "us.openai.gpt-6.1-sol": "GPT-6.1", "us.anthropic.claude-opus-5-5": "Opus 5.5"}
ORDER = list(MODELS.values())
PROBLEMS = ["1a", "1b", "1c", "2a", "2b", "2c", "3a", "3b", "3c", "3d", "4a", "4b", "4c", "5", "6", "7", "8",
            "9", "10", "11"]
SPLIT_LABELS = {"evolution": "training", "development": "validation", "test": "sealed test"}


def problem_of(row):
    return row.get("problem") or row["task_id"].split("-")[1]


def cell_value(row):
    if row is None:
        return None
    return row["relative_l2"] if row["status"] == "ok" and math.isfinite(row["relative_l2"]) else None


def fmt(row):
    if row is None:
        return "—"
    value = cell_value(row)
    text = f"{value:.1e}" if value is not None else row["status"].replace("_", " ")
    mark = "" if row["family_correct"] else " ✗"
    return ("**" + text + "**" if row["passed"] else text) + mark


def load_runs(results, run_dirs=None):
    runs = {}
    names = [d.name for d in run_dirs] if run_dirs else json.loads((results/"protocol_checks.json").read_text())
    for name in names:
        run = results.parent/name
        manifest = json.loads((run/"manifest.json").read_text())
        if manifest.get("task_protocol") != "pdebench":
            continue
        model = MODELS.get(manifest["model_config"]["model"], manifest["model_config"]["model"])
        arms = {}
        for arm in ("single", "rsi"):
            rows = json.loads((run/arm/"frozen_results.json").read_text())
            sealed = run/arm/"sealed_test/results.json"
            rows += json.loads(sealed.read_text()) if sealed.exists() else []
            arms[arm] = {(r["split"], problem_of(r), r["task_id"]): r for r in rows}
            arms[arm]["_report"] = manifest["arms"][arm]
        arms["_manifest"] = manifest
        runs[model] = arms
    return runs


def summary(rows):
    rows = list(rows)
    if not rows:
        return None
    passes = sum(r["passed"] for r in rows)
    mean = sum(r["relative_l2"] for r in rows)/len(rows)
    finite = [cell_value(r) for r in rows if cell_value(r) is not None]
    median = sorted(finite)[len(finite)//2] if finite else None
    geometric = math.exp(sum(math.log(max(v, 1e-300)) for v in finite)/len(finite)) if finite else None
    accuracy = sum(bool(r["family_correct"]) for r in rows)/len(rows)
    pinn = sum((r.get("method") or "") == "pinn" for r in rows)
    return {"n": len(rows), "passes": passes, "pass_rate": passes/len(rows), "mean": mean, "median": median,
            "geometric_mean": geometric, "failed_runs": len(rows)-len(finite), "accuracy": accuracy, "pinn": pinn}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--runs", type=Path, nargs="*", help="run directories (default: protocol_checks.json)")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.results, args.runs)
    models = [m for m in ORDER if m in runs]
    lines = ["# pde_benchmarks.tex: single run vs RSI", "",
             "Overview: pass rate (both relative L2 errors <= 2e-3, exact IC/BC, within limits) with the median and",
             "geometric-mean relative L2 over executed cases; failed runs (crash/invalid output) are counted separately.",
             "Per-case tables: each cell is the relative space-time L2 error of the final (frozen) agent; **bold** = passed",
             "(both errors <= 2e-3, IC/BC exact, within 90 s and 16 GB); ✗ = wrong equation class identified.",
             "Training and validation cases were visible to the improvement loop (training with labels);",
             "sealed-test cases were scored only after freezing.", ""]
    lines += ["## Overview", "",
              "| Model | Arm | Split | Pass rate | Median rel. L2 | Geo. mean rel. L2 | Failed runs | Class accuracy | PINN solves | Accepted / rounds |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    overview = {}
    for model in models:
        for arm in ("single", "rsi"):
            report = runs[model][arm]["_report"]
            for split in ("evolution", "development", "test"):
                rows = [r for key, r in runs[model][arm].items() if key != "_report" and key[0] == split]
                item = summary(rows)
                if item is None:
                    continue
                overview[(model, arm, split)] = item
                median = "—" if item["median"] is None else f"{item['median']:.1e}"
                geometric = "—" if item["geometric_mean"] is None else f"{item['geometric_mean']:.1e}"
                # Time-budgeted arms have no fixed round count: show the gate decisions they reached.
                timed = runs[model]["_manifest"]["configuration"].get("time_budget_seconds_per_arm")
                rounds = report["completed_rounds"] if timed else report["requested_rounds"]
                decisions = "one request" if report.get("one_shot") else f"{report['accepted_candidates']}/{rounds}"
                lines.append(f"| {model} | {'single run' if arm == 'single' else 'RSI'} | {SPLIT_LABELS[split]} | "
                             f"**{item['pass_rate']:.0%}** ({item['passes']}/{item['n']}) | {median} | {geometric} | "
                             f"{item['failed_runs']} | {item['accuracy']:.0%} | {item['pinn']} | "
                             f"{decisions} |")
    for split in ("evolution", "development", "test"):
        lines += ["", f"## Per case: {SPLIT_LABELS[split]}", "",
                  "| Case | Class | " + " | ".join(f"{m} single | {m} RSI" for m in models) + " |",
                  "| --- | --- | " + "---: | "*(2*len(models)) + ""]
        keys = sorted({key for model in models for arm in ("single", "rsi") for key in runs[model][arm]
                       if key != "_report" and key[0] == split}, key=lambda k: (PROBLEMS.index(k[1]), k[2]))
        for key in keys:
            cells = [fmt(runs[m][arm].get(key)) for m in models for arm in ("single", "rsi")]
            lines.append(f"| {key[1]} {TITLES[key[1]]} | {CLASS[key[1]]} | " + " | ".join(cells) + " |")
    continuation = [(m, runs[m]["_manifest"]) for m in models if "continuation" in runs[m]["_manifest"]]
    if continuation:
        lines += ["", "## Continuation test (frozen RSI agent, 2 more rounds)", "",
                  "| Model | Initial-workflow gain | Evolved-workflow gain | Advantage |", "| --- | ---: | ---: | ---: |"]
        for model, manifest in continuation:
            gains = {name: item["initial_validation"]["macro_relative_l2"]-item["selected_validation"]["macro_relative_l2"]
                     for name, item in manifest["continuation"].items()}
            lines.append(f"| {model} | {gains['initial_policy']:.3g} | {gains['evolved_policy']:.3g} | "
                         f"{manifest['continuation_policy_advantage']:.3g} |")
    (args.results/"pdebench_tables.md").write_text("\n".join(lines)+"\n")
    (args.results/"pdebench_overview.json").write_text(json.dumps(
        [{"model": m, "arm": a, "split": s, **v} for (m, a, s), v in overview.items()], indent=2)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
