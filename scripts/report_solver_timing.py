"""Agent coding efficiency: how long each frozen solver takes, next to how often it passes.

Usage (after scripts/time_frozen_solvers.py):
    python scripts/report_solver_timing.py runs/solver-timing-pdebench-20261017
Writes solver_timing.md and solver_timing_per_case.csv into the timing directory. Times are medians over
the repeats of the in-sandbox solve() call (the agent's code only) and of the whole sandboxed call
(adds start-up, PyTorch import, and output transfer).
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

from pde_rsi.pdebench import TITLES

LABELS = {"sonnet45": "Sonnet 4.5", "gpt-oss": "GPT-OSS", "gpt55": "GPT-5.5", "gpt61": "GPT-6.1", "opus55": "Opus 5.5"}
ORDER = ["1a", "1b", "1c", "2a", "2b", "2c", "3a", "3b", "3c", "3d", "4a", "4b", "4c", "5", "6", "7", "8", "9", "10", "11"]
SPLITS = {"evolution": "training", "development": "validation", "test": "sealed test"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timing", type=Path)
    args = parser.parse_args()
    environment = json.loads((args.timing/"environment.json").read_text())
    programs = {}
    for path in sorted(args.timing.glob("*-*.json")):
        if path.name in ("environment.json", "summary.json"):
            continue
        model, _, arm = path.stem.rpartition("-")
        per_case = {}
        for row in json.loads(path.read_text()):
            per_case.setdefault(row["case_id"], []).append(row)
        programs[(model, arm)] = per_case
    with (args.timing/"solver_timing_per_case.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "arm", "case_id", "split", "tex_case", "median_solve_seconds", "min_solve_seconds",
                         "max_solve_seconds", "median_call_seconds", "peak_memory_mb", "status", "passed", "relative_l2"])
        for (model, arm), per_case in programs.items():
            for case_id, rows in sorted(per_case.items()):
                solves = [r["solve_seconds"] for r in rows]
                writer.writerow([LABELS.get(model, model), arm, case_id, SPLITS[rows[0]["split"]], rows[0]["tex_case"],
                                 f"{statistics.median(solves):.3f}", f"{min(solves):.3f}", f"{max(solves):.3f}",
                                 f"{statistics.median(r['call_wall_seconds'] for r in rows):.3f}",
                                 f"{max(r['peak_memory_mb'] for r in rows):.0f}", rows[0]["status"], rows[0]["passed"],
                                 rows[0]["relative_l2"]])
    models = [m for m in LABELS if any(k[0] == m for k in programs)]
    lines = [f"Node {environment['node']} ({environment['cpu_model']}); each program pinned to its own core, all ten "
             f"run concurrently; {environment['repeats']} repeats of {environment['cases_per_repeat']} cases.", "",
             "| Model | Arm | Passed (of 49) | Total solve time | Median per case | Slowest case | Total incl. start-up | Max repeat spread |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model in models:
        for arm in ("single", "rsi"):
            per_case = programs.get((model, arm))
            if per_case is None:
                continue
            ok = {cid: rows for cid, rows in per_case.items() if rows[0]["status"] == "ok"}
            medians = {cid: statistics.median(r["solve_seconds"] for r in rows) for cid, rows in ok.items()}
            calls = sum(statistics.median(r["call_wall_seconds"] for r in rows) for rows in per_case.values())
            spread = max((max(r["solve_seconds"] for r in rows)-min(r["solve_seconds"] for r in rows)
                          for rows in ok.values()), default=0.0)
            slowest = max(medians, key=medians.get) if medians else None
            passes = sum(rows[0]["passed"] for rows in per_case.values())
            lines.append(f"| {LABELS[model]} | {'single run' if arm == 'single' else 'RSI'} | {passes} | "
                         + (f"{sum(medians.values()):.1f} s | {statistics.median(medians.values()):.2f} s | "
                            f"{medians[slowest]:.1f} s ({per_case[slowest][0]['tex_case']}) | "
                            if medians else "— | — | — | ")
                         + f"{calls:.0f} s | {spread:.2f} s |")
    lines += ["", "Median solve() seconds per tex case (sum over that case's splits), single / RSI:", "",
              "| tex case | " + " | ".join(LABELS[m] for m in models) + " |", "| --- | " + "---: | "*len(models)]
    for problem in ORDER:
        cells = []
        for model in models:
            parts = []
            for arm in ("single", "rsi"):
                per_case = programs.get((model, arm), {})
                rows = [rows for rows in per_case.values() if rows[0]["tex_case"] == problem and rows[0]["status"] == "ok"]
                parts.append(f"{sum(statistics.median(r['solve_seconds'] for r in items) for items in rows):.2f}"
                             if rows else "fail")
            cells.append(" / ".join(parts))
        lines.append(f"| {problem} {TITLES[problem]} | " + " | ".join(cells) + " |")
    (args.timing/"solver_timing.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
