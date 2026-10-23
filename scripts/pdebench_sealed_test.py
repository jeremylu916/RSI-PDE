"""Score the frozen single-run and RSI programs of pdebench runs on the sealed test cases.

The test split (tex reserved profile and harder literature variants) is never shown to agents and never
enters selection or memory: this script runs only after an arm is frozen and writes its results next to
the arm. Run on a GPU node after sourcing scripts/env.sh:
    python scripts/pdebench_sealed_test.py --runs runs/code-agents-pdebench-*-20261017

Writes <run>/<arm>/sealed_test/{results.json, solutions/<case_id>.npz} and <run>/sealed_test_summary.json.
"""

import argparse
import json
import shutil
from pathlib import Path

from pde_rsi.modular import load_program
from pde_rsi.pdebench import make_cases, run_case
from pde_rsi import inventory
from pde_rsi.runner import Budget
from pde_rsi.storage import project_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--force", action="store_true", help="re-run arms that already have sealed results")
    parser.add_argument("--split", choices=("test", "test2", "test3"), default="test",
                        help="test: the original sealed split; test2: the fresh split of 2026-10-18; test3: the fresh "
                             "split of 2026-10-20 (or use scripts/score_fresh_splits.py)")
    args = parser.parse_args()
    for run in map(project_path, args.runs):
        manifest = json.loads((run/"manifest.json").read_text())
        settings = manifest.get("task_protocol_metadata", {})
        suite = manifest.get("task_protocol")
        if suite == "inventory":  # the inventory suite has one sealed split
            cases, runner, folder = inventory.make_cases("inventory_test", settings.get("noise_fraction", 0.01)), inventory.run_case, "sealed_test"
        elif suite == "pdebench":
            # sealed cases are posed exactly like the run's public cases (noise level, hidden time dependence)
            cases = make_cases(args.split, settings.get("noise_fraction", 0.01), settings.get("hide_time_dependence", False))
            runner, folder = run_case, {"test": "sealed_test", "test2": "sealed_test2", "test3": "sealed_test3"}[args.split]
        else:
            raise SystemExit(f"{run} is not a pdebench or inventory run")
        summary = {}
        for arm in ("single", "rsi"):
            if arm not in manifest.get("arms", {}):
                raise SystemExit(f"{run}: arm {arm} has not finished")
            target = run/arm/folder
            if (target/"results.json").exists() and not args.force:
                results = json.loads((target/"results.json").read_text())
            else:
                if target.exists():
                    shutil.rmtree(target)
                (target/"solutions").mkdir(parents=True)
                code = load_program(run/arm/"programs/frozen")  # assembles modular (harness v2) programs
                budget = Budget(len(cases), 3600, max_cpu_seconds=3600)
                results = [runner(case, code, budget, target/"solutions"/f"{case.case_id}.npz") for case in cases]
                (target/"results.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
            summary[arm] = {"cases": len(results), "passes": sum(r["passed"] for r in results),
                            "mean_relative_l2": sum(r["relative_l2"] for r in results)/len(results),
                            "class_accuracy": sum(bool(r["family_correct"]) for r in results)/len(results),
                            "executed": sum(r["status"] == "ok" for r in results)}
            print(run.name, arm, summary[arm], flush=True)
        (run/f"{folder}_summary.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    main()
