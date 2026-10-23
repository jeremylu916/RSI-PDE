"""Diagnostic, outside the protocol: score a single-run answer that the static policy rejected, as written.

The harness rejects programs over MAX_CODE_CHARS before running them, so a rejected single run scores 0 without
saying anything about its solver. This script runs the rejected program unchanged in the same sandbox (the
sandbox's own import restrictions still apply) on the public cases and the fresh sealed split, to measure what
the answer would have scored. Results are written to <attempt>/diagnostic_as_written/ and are never used as a
single-run result. Run on a GPU node after sourcing scripts/env.sh:
    python scripts/diagnostic_score_rejected.py --attempts runs/matched-inv-opus55-harness-r1-20261020/single_overlimit_v1 --core 56
"""

import argparse
import json
import os
from pathlib import Path

from pde_rsi import inventory
from pde_rsi.runner import Budget
from pde_rsi.storage import project_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=Path, nargs="+", required=True)
    parser.add_argument("--core", type=int, default=None)
    args = parser.parse_args()
    if args.core is not None:
        os.sched_setaffinity(0, {args.core})
    for attempt in map(project_path, args.attempts):
        proposal = json.loads((attempt/"proposal-001-1.json").read_text())["proposal"]
        code = next(u["code"] for u in proposal["program_updates"] if u["family"] == inventory.PROGRAM_KEY)
        target = attempt/"diagnostic_as_written"
        target.mkdir(exist_ok=True)
        summary = {"note": "outside the protocol: a program the static policy rejected, run as written",
                   "rejection": json.loads((attempt/"one_shot.json").read_text())["message"], "characters": len(code)}
        for split in ("evolution", "development", "inventory_test2"):
            cases = inventory.make_cases(split)
            budget = Budget(len(cases), 7200, max_cpu_seconds=7200)
            (target/split).mkdir(exist_ok=True)
            rows = [inventory.run_case(case, code, budget, target/split/f"{case.case_id}.npz") for case in cases]
            (target/f"{split}.json").write_text(json.dumps(rows, indent=1, default=float)+"\n")
            summary[split] = {"cases": len(rows), "passes": sum(r["passed"] for r in rows),
                              "executed": sum(r["status"] == "ok" for r in rows)}
        (target/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(attempt, json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
