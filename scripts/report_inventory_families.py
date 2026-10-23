"""Per-family results of the inventory matched study: single run vs RSI vs native agent, 3 runs each.

For each of the 12 inventory problems, cells give sealed cases passed over the 3 runs (one sealed case per family
and run) and the median sealed relative L2, plus training + validation passes. Also writes the long CSV.
Usage (after sourcing scripts/env.sh):
    python scripts/report_inventory_families.py --output runs/matched-inv-study-results-20261020
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

from pde_rsi.storage import PROJECT_ROOT, project_path

WAVES = (("opus55", "Opus 5.5", "claude-code", "Claude Code"), ("gpt55", "GPT-5.5", "codex", "Codex"))
CONFIGS = (("single", "single run"), ("rsi", "RSI"), ("native", "native"))


def load(wave, agent, key, run):
    runs = PROJECT_ROOT/"runs"
    if key == "native":
        final = runs/f"matched-inv-{wave}-{agent}-r{run}-20261020"/"final"
        return (json.loads((final/"frozen_results.json").read_text()),
                json.loads((final/"sealed_inventory_test_results.json").read_text()))
    arm = runs/f"matched-inv-{wave}-harness-r{run}-20261020"/key
    return (json.loads((arm/"frozen_results.json").read_text()),
            json.loads((arm/"sealed_test/results.json").read_text()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records, problems = [], []
    for wave, model, agent, agent_label in WAVES:
        for key, label in CONFIGS:
            for run in (1, 2, 3):
                public, sealed = load(wave, agent, key, run)
                for row in public+sealed:
                    problem = row.get("problem") or row["task_id"].split("-")[1]
                    if problem not in problems:
                        problems.append(problem)
                    records.append({"model": model, "config": agent_label if key == "native" else label, "run": run,
                                    "problem": problem, "split": row["split"], "status": row["status"],
                                    "passed": row["passed"], "relative_l2": row["relative_l2"],
                                    "final_relative_l2": row["final_relative_l2"],
                                    "family_correct": row.get("family_correct"), "method": row.get("method"),
                                    "solve_seconds": row.get("runtime_seconds")})
    with (output/"per_family_long.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    columns = [(model, config) for model in dict.fromkeys(r["model"] for r in records)
               for config in dict.fromkeys(r["config"] for r in records if r["model"] == model)]
    lines = ["Per inventory problem, 3 runs per configuration. Cell: sealed passed (of 3), median sealed relative L2; "
             "in parentheses, training + validation passed (of 6). ✗ marks a run whose sealed case got the wrong "
             "equation class.", "",
             "| Problem | " + " | ".join(f"{m} {c}" for m, c in columns) + " |",
             "| --- | " + "---: | "*len(columns)]
    for problem in problems:
        cells = []
        for model, config in columns:
            rows = [r for r in records if (r["model"], r["config"], r["problem"]) == (model, config, problem)]
            sealed = [r for r in rows if r["split"] == "inventory_test"]
            public = [r for r in rows if r["split"] != "inventory_test"]
            median = statistics.median(min(r["relative_l2"], 1e6) for r in sealed)
            wrong = "✗"*sum(r["family_correct"] is False for r in sealed)
            cells.append(f"{sum(r['passed'] for r in sealed)}/3, {median:.1e}{wrong} ({sum(r['passed'] for r in public)}/6)")
        lines.append(f"| {problem} | " + " | ".join(cells) + " |")
    totals = []
    for model, config in columns:
        rows = [r for r in records if (r["model"], r["config"]) == (model, config)]
        totals.append(f"**{sum(r['passed'] for r in rows if r['split'] == 'inventory_test')}/36** "
                      f"({sum(r['passed'] for r in rows if r['split'] != 'inventory_test')}/72)")
    lines.append("| **All** | " + " | ".join(totals) + " |")
    (output/"per_family.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
