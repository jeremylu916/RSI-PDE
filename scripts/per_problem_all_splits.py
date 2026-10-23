"""Per tex case (training, validation, sealed test) relative L2 of each model's single run and RSI agent.

Usage (after sourcing scripts/env.sh):
    python scripts/per_problem_all_splits.py runs/code-agent-results-pdebench-20261017 --stamp 20261017 --prefix pdebench
Writes <results>/per_problem_all_splits.md.
"""

import argparse
import json
from pathlib import Path

from pde_rsi.pdebench import SPLITS, TITLES, make_cases

RUNS = [("Sonnet 4.5", "claude"), ("GPT-OSS", "gpt-oss"), ("GPT-5.5", "gpt55"), ("GPT-6.1", "gpt61"),
        ("Opus 5.5", "opus55")]
ORDER = ["1a", "1b", "1c", "2a", "2b", "2c", "3a", "3b", "3c", "3d", "4a", "4b", "4c", "5", "6", "7", "8", "9", "10", "11"]
SPLIT_LABELS = {"evolution": "train", "development": "val", "test": "test"}
PUBLIC_AND_SEALED = ("evolution", "development", "test")


def cell(row):
    if row["status"] != "ok":
        return "fail"
    value = f"{row['relative_l2']:.1e}"
    return (f"**{value}**" if row["passed"] else value) + ("" if row["family_correct"] else "✗")


def variant(case):
    s, p = case.truth.spec, case.truth.problem
    if s["profile"]:
        return {"sine": "sine", "polynomial": "poly", "mixed_sine": "mixed sine"}[s["profile"]]
    labels = {"1b": lambda: "D=1", "1c": lambda: f"α={s['sharpness']:g}", "2b": lambda: f"L={s['b']:g}, D={s['D']:g}",
              "2c": lambda: f"γ={s['gamma']:.3g}, D={s['D']:g}", "3b": lambda: f"Pe={s['v']*s['b']/s['D']:g}",
              "3c": lambda: f"Pe={s['v']*s['b']/s['D']:g}", "3d": lambda: f"v={s['v']:g}, D={s['D']:g}",
              "4b": lambda: "D=0.01/π", "4c": lambda: f"D={s['D']:g}", "6": lambda: f"α={s['alpha']:g}, p={s['p']:g}",
              "9": lambda: f"α={s['alpha']:g}, p={s['p']:g}", "10": lambda: f"c={s['c']:g}, p={s['p']:g}",
              "11": lambda: f"c={s['c']:g}, p={s['p']:g}"}
    return labels[p]()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--prefix", default="pdebench", help="run name infix: runs/code-agents-<prefix>-<model>-<stamp>")
    parser.add_argument("--stamp", default="20261017")
    args = parser.parse_args()
    results = {}
    for label, key in RUNS:
        for arm in ("single", "rsi"):
            run = Path(f"runs/code-agents-{args.prefix}-{key}-{args.stamp}")/arm
            rows = json.loads((run/"frozen_results.json").read_text())+json.loads((run/"sealed_test/results.json").read_text())
            for row in rows:
                results[(label, arm, row["task_id"])] = row
    cases = [case for split in PUBLIC_AND_SEALED for case in make_cases(split)]
    cases.sort(key=lambda case: (ORDER.index(case.truth.problem), SPLITS.index(case.split)))
    lines = ["| tex case | Split | Variant | " + " | ".join(f"{m} S / R" for m, _ in RUNS) + " |",
             "|---|---|---|" + "---|"*len(RUNS)]
    previous = None
    for case in cases:
        problem = case.truth.problem
        name = f"**{problem}** {TITLES[problem].split(' (')[0]}" if problem != previous else ""
        previous = problem
        lines.append(f"| {name} | {SPLIT_LABELS[case.split]} | {variant(case)} | " + " | ".join(
            f"{cell(results[(m, 'single', case.task_id)])} / {cell(results[(m, 'rsi', case.task_id)])}" for m, _ in RUNS) + " |")
    for split in PUBLIC_AND_SEALED:
        subset = [case for case in cases if case.split == split]
        lines.append(f"| **Passed** | {SPLIT_LABELS[split]} ({len(subset)}) | | " + " | ".join(
            f"{sum(results[(m, 'single', c.task_id)]['passed'] for c in subset)} / "
            f"{sum(results[(m, 'rsi', c.task_id)]['passed'] for c in subset)}" for m, _ in RUNS) + " |")
    (args.results/"per_problem_all_splits.md").write_text(
        "Relative space-time L2 of the frozen agents, single run (S, one model request) / RSI (R). Bold = passed; "
        "✗ = wrong equation class; fail = crashed or invalid output.\n\n"+"\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
