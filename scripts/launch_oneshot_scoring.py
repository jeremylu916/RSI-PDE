"""Launch `redo_single_one_shot.py score` on a GPU node, one process per dedicated core.

Runs are spread over the cores by the solver time their earlier arms needed (longest first, greedy), so the
shards finish together. Run on the login node:
    python scripts/launch_oneshot_scoring.py --node h013 --cores 15,17,19 --runs $(cat runs/oneshot-redo-runs.txt)
Each shard logs to runs/oneshot-score-<core>.log.
"""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path("/depot/lin491/data/Binghang_project/PDE-RSI")


def expected_seconds(run):
    arm = run/"fixed_workflow" if (run/"fixed_workflow").exists() else run/"rsi"
    total = 0.0
    for name in ("frozen_results.json", "sealed_test/results.json", "sealed_test2/results.json"):
        if (arm/name).exists():
            total += sum(r.get("wall_seconds") or 0.0 for r in json.loads((arm/name).read_text()))
    return total+60


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", required=True)
    parser.add_argument("--cores", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cores = [int(c) for c in args.cores.split(",")]
    shards = {c: [] for c in cores}
    load = {c: 0.0 for c in cores}
    for run in sorted((ROOT/r if not r.startswith("/") else Path(r) for r in args.runs), key=expected_seconds,
                      reverse=True):
        core = min(load, key=load.get)
        shards[core].append(str(run.relative_to(ROOT)))
        load[core] += expected_seconds(run)
    for core, runs in shards.items():
        if not runs:
            continue
        print(f"core {core}: {len(runs)} runs, about {load[core]/60:.0f} min")
        command = (f"cd {ROOT} && source scripts/env.sh && setsid nohup python scripts/redo_single_one_shot.py score "
                   f"--core {core} --runs {' '.join(runs)} > runs/oneshot-score-{core}.log 2>&1 < /dev/null &")
        if not args.dry_run:
            subprocess.run(["timeout", "20", "ssh", "-n", "-f", "-o", "BatchMode=yes", args.node, command],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
