"""Time every frozen pdebench solver under controlled conditions (agent coding efficiency).

Each frozen program (model x arm) runs in its own process pinned to one dedicated CPU core, all programs
at the same time on one node, on all 49 cases (training, validation, sealed test), `--repeats` times.
Every solve goes through the same GPU sandbox as the experiments. Records the in-sandbox solve() time, the
whole sandboxed call (start-up, PyTorch import, and output transfer), peak memory, and the result, plus the
node, CPU model, core, and load average.
Run on a GPU node after sourcing scripts/env.sh:
    python scripts/time_frozen_solvers.py --runs runs/code-agents-pdebench-*-20261017 \
        --output runs/solver-timing-pdebench-20261017
"""

import argparse
import json
import multiprocessing
import os
import platform
import statistics
from pathlib import Path

from pde_rsi.modular import load_program
from pde_rsi.pdebench import SPLITS, make_cases, run_case
from pde_rsi.runner import Budget
from pde_rsi.storage import project_path

MODELS = {"us.anthropic.claude-sonnet-4-5-20250929-v1:0": "sonnet45", "openai.gpt-oss-120b-1:0": "gpt-oss",
          "us.openai.gpt-5.5": "gpt55", "us.openai.gpt-6.1-sol": "gpt61", "us.anthropic.claude-opus-5-5": "opus55"}


def cpu_model():
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def worker(job):
    label, code_path, core, repeats, output = job
    os.sched_setaffinity(0, {core})
    code = load_program(Path(code_path))  # a programs/frozen directory; modular programs are assembled
    cases = [case for split in SPLITS for case in make_cases(split)]
    rows = []
    for repeat in range(repeats):
        for case in cases:
            load = os.getloadavg()[0]
            result = run_case(case, code, Budget(1, 3600, max_cpu_seconds=3600))
            rows.append({"program": label, "repeat": repeat, "case_id": case.case_id, "split": case.split,
                         "tex_case": case.truth.problem, "status": result["status"], "passed": result["passed"],
                         "relative_l2": result["relative_l2"], "solve_seconds": result["runtime_seconds"],
                         "call_wall_seconds": result["wall_seconds"], "cpu_seconds": result["cpu_seconds"],
                         "peak_memory_mb": result["peak_memory_mb"], "core": core, "load_1min": load})
    (Path(output)/f"{label}.json").write_text(json.dumps(rows, indent=1, allow_nan=False)+"\n")
    return label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cores = sorted(os.sched_getaffinity(0))
    jobs = []
    for run in map(project_path, sorted(args.runs)):
        model = MODELS[json.loads((run/"manifest.json").read_text())["model_config"]["model"]]
        for arm in ("single", "rsi"):
            jobs.append((f"{model}-{arm}", str(run/arm/"programs/frozen")))
    if len(jobs) > len(cores):
        raise SystemExit(f"Need {len(jobs)} dedicated cores, have {len(cores)}")
    environment = {"node": platform.node(), "cpu_model": cpu_model(), "allowed_cores": cores,
                   "assignment": {label: core for (label, _), core in zip(jobs, cores)},
                   "repeats": args.repeats, "cases_per_repeat": sum(len(make_cases(s)) for s in SPLITS),
                   "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "default"), "sandbox": "GPU sandbox, 1 thread"}
    (output/"environment.json").write_text(json.dumps(environment, indent=2)+"\n")
    context = multiprocessing.get_context("fork")
    with context.Pool(len(jobs)) as pool:
        for label in pool.imap_unordered(worker, [(label, path, core, args.repeats, str(output))
                                                  for (label, path), core in zip(jobs, cores)]):
            print("done", label, flush=True)
    summary = {}
    for label, _ in jobs:
        rows = json.loads((output/f"{label}.json").read_text())
        per_case = {}
        for row in rows:
            per_case.setdefault(row["case_id"], []).append(row)
        medians = {cid: statistics.median(r["solve_seconds"] for r in items) for cid, items in per_case.items()}
        walls = {cid: statistics.median(r["call_wall_seconds"] for r in items) for cid, items in per_case.items()}
        spread = [max(r["solve_seconds"] for r in items)-min(r["solve_seconds"] for r in items)
                  for items in per_case.values()]
        summary[label] = {"cases": len(per_case), "total_median_solve_seconds": sum(medians.values()),
                          "median_case_solve_seconds": statistics.median(medians.values()),
                          "max_case_solve_seconds": max(medians.values()),
                          "total_median_call_seconds": sum(walls.values()),
                          "max_repeat_spread_seconds": max(spread),
                          "passes": sum(items[0]["passed"] for items in per_case.values())}
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
