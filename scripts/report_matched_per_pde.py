"""Per tex PDE results of the pdebench matched study: single run vs RSI vs native agent, 3 runs each.

Cells give cases passed, summed over the 3 runs: sealed (original test + fresh test2) and all splits (training,
validation, test, test2). Also writes the long CSV of every case.
Usage (after sourcing scripts/env.sh):
    python scripts/report_matched_per_pde.py --output runs/matched-study-results-20261018
"""

import argparse
import csv
import json
from pathlib import Path

from pde_rsi.pdebench import TITLES
from pde_rsi.storage import PROJECT_ROOT, project_path

WAVES = (("opus55", "Opus 5.5", "claude-code", "Claude Code"), ("gpt55", "GPT-5.5", "codex", "Codex"))
CONFIGS = (("single", "single run"), ("rsi", "RSI"), ("native", "native"))
ORDER = ["1a", "1b", "1c", "2a", "2b", "2c", "3a", "3b", "3c", "3d", "4a", "4b", "4c", "5", "6", "7", "8", "9", "10", "11"]


def load(wave, agent, key, run):
    runs = PROJECT_ROOT/"runs"
    if key == "native":
        final = runs/f"matched-{wave}-{agent}-r{run}-20261018"/"final"
        names = ("frozen_results.json", "sealed_test_results.json", "sealed_test2_results.json")
        return [row for name in names for row in json.loads((final/name).read_text())]
    arm = runs/f"matched-{wave}-harness-r{run}-20261018"/key
    names = ("frozen_results.json", "sealed_test/results.json", "sealed_test2/results.json")
    return [row for name in names for row in json.loads((arm/name).read_text())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = project_path(args.output)
    records = []
    for wave, model, agent, agent_label in WAVES:
        for key, label in CONFIGS:
            for run in (1, 2, 3):
                for row in load(wave, agent, key, run):
                    records.append({"model": model, "config": agent_label if key == "native" else label, "run": run,
                                    "problem": row["task_id"].split("-")[1], "split": row["split"],
                                    "status": row["status"], "passed": row["passed"],
                                    "relative_l2": row["relative_l2"], "final_relative_l2": row["final_relative_l2"],
                                    "family_correct": row.get("family_correct"), "solve_seconds": row.get("runtime_seconds")})
    with (output/"per_pde_long.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    columns = [(m, c) for m in dict.fromkeys(r["model"] for r in records)
               for c in dict.fromkeys(r["config"] for r in records if r["model"] == m)]
    lines = ["Per tex PDE, summed over 3 runs per configuration. Cell: sealed passed / sealed cases (original test + "
             "fresh test2); in parentheses, passed / cases over all splits (training, validation, test, test2).", "",
             "| tex PDE | " + " | ".join(f"{m} {c}" for m, c in columns) + " |", "| --- | " + "---: | "*len(columns)]
    sealed_splits = ("test", "test2")
    for problem in ORDER:
        cells = []
        for model, config in columns:
            rows = [r for r in records if (r["model"], r["config"], r["problem"]) == (model, config, problem)]
            sealed = [r for r in rows if r["split"] in sealed_splits]
            cells.append(f"{sum(r['passed'] for r in sealed)}/{len(sealed)} ({sum(r['passed'] for r in rows)}/{len(rows)})")
        lines.append(f"| {problem} {TITLES[problem].split(' (')[0]} | " + " | ".join(cells) + " |")
    totals = []
    for model, config in columns:
        rows = [r for r in records if (r["model"], r["config"]) == (model, config)]
        sealed = [r for r in rows if r["split"] in sealed_splits]
        totals.append(f"**{sum(r['passed'] for r in sealed)}/{len(sealed)}** ({sum(r['passed'] for r in rows)}/{len(rows)})")
    lines.append("| **All** | " + " | ".join(totals) + " |")
    (output/"per_pde.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
