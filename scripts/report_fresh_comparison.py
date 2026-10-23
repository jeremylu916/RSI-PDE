"""Harness v2 vs every earlier configuration on the fresh sealed splits of 2026-10-20.

Configurations per model (3 runs each): single run and RSI of harness v1 (matched studies), the native coding
agent, and single run and RSI of harness v2 (scripts/run_v2_study.sh). Every frozen solver is scored on the same
fresh split (pdebench test3: 20 cases; inventory inventory_test2: 12 cases) by scripts/score_fresh_splits.py.
Usage (after sourcing scripts/env.sh):
    python scripts/report_fresh_comparison.py --output runs/v2-fresh-comparison-20261020
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

from pde_rsi.costs import load_prices, response_cost
from pde_rsi.storage import PROJECT_ROOT, project_path

RUNS = PROJECT_ROOT/"runs"
WAVES = (("opus55", "Opus 5.5", "us.anthropic.claude-opus-5-5", "claude-code", "Claude Code"),
         ("gpt55", "GPT-5.5", "us.openai.gpt-5.5", "codex", "Codex"))
SUITES = {"pdebench": ("", "20261018", "test3", 20), "inventory": ("inv-", "20261020", "inventory_test2", 12)}


def configurations(suite, wave, agent, agent_label):
    tag, stamp, split, _ = SUITES[suite]
    return [("v1 single run", lambda r: RUNS/f"matched-{tag}{wave}-harness-r{r}-{stamp}", "single"),
            ("v1 single run + repair", lambda r: RUNS/f"matched-{tag}{wave}-harness-r{r}-{stamp}", "single_repair"),
            ("v1 RSI", lambda r: RUNS/f"matched-{tag}{wave}-harness-r{r}-{stamp}", "rsi"),
            (agent_label, lambda r: RUNS/f"matched-{tag}{wave}-{agent}-r{r}-{stamp}", "native"),
            ("v2 single run", lambda r: RUNS/f"v2-{tag}{wave}-harness-r{r}-20261020", "single"),
            ("v2 single run + repair", lambda r: RUNS/f"v2-{tag}{wave}-harness-r{r}-20261020", "single_repair"),
            ("v2 RSI", lambda r: RUNS/f"v2-{tag}{wave}-harness-r{r}-20261020", "rsi")]


def load(run, arm, split, model, prices):
    if arm == "native":
        path = run/"final"/f"sealed_{split}_results.json"
        if not path.exists():
            return None
        return json.loads(path.read_text()), json.loads((run/"manifest.json").read_text())["model_cost_usd"]
    if arm == "single_repair" and not (run/arm).exists():
        arm = "single"  # the single answer was valid, so the repaired single run is the strict one
    path = run/arm/f"sealed_{split}"/"results.json"
    if not path.exists():
        return None
    cost = sum(response_cost(p, model, prices) or 0 for p in (run/arm).glob("model-call-*/response.json"))
    if arm == "single_repair":  # the repaired run also paid for the rejected first answer
        cost += sum(response_cost(p, model, prices) or 0 for p in (run/"single").glob("model-call-*/response.json"))
    return json.loads(path.read_text()), cost


def problem_of(row):
    return row.get("problem") or row["task_id"].split("-")[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    lines, records = [], []
    for suite, (_, _, split, cases) in SUITES.items():
        summary = [f"## {suite}: fresh sealed split `{split}` ({cases} cases per run)", "",
                   "| Model | Configuration | Runs | Passed per run (mean ± sd) | Per run | Cost per run |",
                   "| --- | --- | ---: | ---: | --- | ---: |"]
        per_problem, columns = {}, []
        for wave, model, model_id, agent, agent_label in WAVES:
            for label, run_of, arm in configurations(suite, wave, agent, agent_label):
                items = [x for x in (load(run_of(r), arm, split, model_id, prices) for r in (1, 2, 3)) if x is not None]
                if not items:
                    continue
                columns.append((model, label))
                passes = [sum(r["passed"] for r in rows) for rows, _ in items]
                spread = f"{statistics.mean(passes):.1f}" + (f" ± {statistics.stdev(passes):.1f}" if len(passes) > 1 else "")
                summary.append(f"| {model} | {label} | {len(items)} | {spread} | {' / '.join(map(str, passes))} | "
                               f"${statistics.mean(c for _, c in items):.2f} |")
                for run, (rows, cost) in enumerate(items, 1):
                    for row in rows:
                        per_problem.setdefault(problem_of(row), {}).setdefault((model, label), []).append(row["passed"])
                        records.append({"suite": suite, "model": model, "configuration": label, "run": run,
                                        "problem": problem_of(row), "passed": row["passed"],
                                        "relative_l2": row["relative_l2"], "final_relative_l2": row["final_relative_l2"],
                                        "family_correct": row.get("family_correct"), "status": row["status"],
                                        "solve_seconds": row.get("runtime_seconds")})
        table = ["", "Per problem: sealed cases passed, summed over the runs.", "",
                 "| Problem | " + " | ".join(f"{m} {c}" for m, c in columns) + " |", "| --- | " + "---: | "*len(columns)]
        for problem, cells in per_problem.items():
            table.append(f"| {problem} | " + " | ".join(
                f"{sum(cells.get(c, []))}/{len(cells.get(c, []))}" for c in columns) + " |")
        lines += summary + table + [""]
    (output/"fresh_comparison.md").write_text(
        "# Harness v2 vs earlier configurations on the fresh sealed splits (2026-10-20)\n\n"+"\n".join(lines)+"\n")
    if records:
        with (output/"fresh_comparison_long.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
