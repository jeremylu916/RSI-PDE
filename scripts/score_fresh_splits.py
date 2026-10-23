"""Score frozen solvers on the fresh sealed splits of 2026-10-20 (pdebench test3, inventory inventory_test2).

These splits were created after test, test2, and inventory_test had been used to diagnose the harness, to compare
harness v2 with every earlier configuration on cases no development decision has seen. Works for harness runs
(arms single and rsi; modular or single-module programs) and native-agent runs (final/retained_solver.py).
Results: <run>/<arm>/sealed_<split>/results.json and <run>/sealed_<split>_summary.json for harness runs;
<run>/final/sealed_<split>_results.json for native runs. Run on a GPU node after sourcing scripts/env.sh:
    python scripts/score_fresh_splits.py --runs runs/matched-*-20261018 [--core 15]
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from pde_rsi import inventory, pdebench
from pde_rsi.modular import load_program
from pde_rsi.runner import Budget
from pde_rsi.storage import project_path

FRESH = {"pdebench": "test3", "inventory": "inventory_test2"}


def cases_and_runner(suite, settings):
    noise = settings.get("noise_fraction", 0.01)
    if suite == "inventory":
        return inventory.make_cases("inventory_test2", noise), inventory.run_case
    return pdebench.make_cases("test3", noise, settings.get("hide_time_dependence", False)), pdebench.run_case


def score(code, cases, runner, target):
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    budget = Budget(len(cases), 7200, max_cpu_seconds=7200)
    return [runner(case, code, budget, target/f"{case.case_id}.npz") for case in cases]


def summary(rows):
    return {"cases": len(rows), "passes": sum(r["passed"] for r in rows),
            "mean_relative_l2": sum(min(r["relative_l2"], 1e6) for r in rows)/len(rows),
            "class_accuracy": sum(bool(r.get("family_correct")) for r in rows)/len(rows),
            "executed": sum(r["status"] == "ok" for r in rows)}


def score_run(run, force, arms=("single", "rsi")):
    manifest = json.loads((run/"manifest.json").read_text())
    if manifest.get("kind") == "native_coding_agent":
        suite, settings = manifest.get("suite", "pdebench"), manifest.get("settings") or {}  # pdebench runs predate "suite"
        split = FRESH[suite]
        out = run/"final"/f"sealed_{split}_results.json"
        if out.exists() and not force:
            return f"{run.name}: done"
        cases, runner = cases_and_runner(suite, settings)
        rows = score((run/"final/retained_solver.py").read_text(), cases, runner, run/"final/solutions"/split)
        out.write_text(json.dumps(rows, indent=1, default=float)+"\n")
        return f"{run.name}: native {summary(rows)}"
    suite, settings = manifest.get("task_protocol"), manifest.get("task_protocol_metadata") or {}
    if suite not in FRESH:
        raise ValueError(f"{run.name} is not a pdebench or inventory run")
    split = FRESH[suite]
    cases, runner = cases_and_runner(suite, settings)
    summary_path = run/f"sealed_{split}_summary.json"
    totals = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for arm in arms:
        if not (run/arm/"programs/frozen").exists():
            continue
        target = run/arm/f"sealed_{split}"
        if (target/"results.json").exists() and not force:
            rows = json.loads((target/"results.json").read_text())
        else:
            rows = score(load_program(run/arm/"programs/frozen"), cases, runner, target/"solutions")
            (target/"results.json").write_text(json.dumps(rows, indent=1, default=float)+"\n")
        totals[arm] = summary(rows)
    (run/f"sealed_{split}_summary.json").write_text(json.dumps(totals, indent=2)+"\n")
    return f"{run.name}: {totals}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--core", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--arms", nargs="+", default=["single", "rsi"], help="harness arms to score")
    args = parser.parse_args()
    if args.core is not None:
        os.sched_setaffinity(0, {args.core})
    for run in map(project_path, args.runs):
        try:
            print(score_run(run, args.force, args.arms), flush=True)
        except Exception as error:  # report and continue with the other runs
            print(f"{run.name}: SKIPPED {type(error).__name__}: {error}", flush=True)


if __name__ == "__main__":
    main()
