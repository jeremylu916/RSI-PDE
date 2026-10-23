"""RSI-PDEbench evaluator: score a solver program on one split of one track, one sandboxed solve per case.

Public splits (bench-train, bench-val) can be scored by anyone with the repository, e.g. to check a solver before a
run; the hidden split (bench-test) needs evaluator_private/bench (its secret, inputs and references).
    rsi-pde-bench-eval --track tex --split bench-val solver.py [--cores 0,1,2,3]
"""

import argparse
import json
import multiprocessing
import os

from ..runner import Budget

CASES, CORE = {}, None


def cases(track, split):
    from .. import inventory as inv
    from .. import pdebench as pb
    from .. import pdebench_ext as px
    if (track, split) not in CASES:
        CASES[(track, split)] = {"tex": pb, "ext": px, "inventory": inv}[track].make_cases(split)
    return CASES[(track, split)]


def _init(cores):
    global CORE
    CORE = cores[(multiprocessing.current_process()._identity[0]-1) % len(cores)]
    os.sched_setaffinity(0, {CORE})


def solve_case(job):
    """One case: the sandboxed solve and the project's evaluator, reduced to the scored fields. The static policy
    is not re-checked: entrants' runners check every submission (harness v2 checks each part of a modular
    program, whose assembled module may exceed one part's 30,000 characters), and the sandbox isolates any code."""
    from .. import inventory as inv
    from .. import pdebench as pb
    from .. import pdebench_ext as px
    track, split, index, code = job
    case = cases(track, split)[index]
    runner = {"tex": pb.run_case, "ext": px.run_case, "inventory": inv.run_case}[track]
    try:
        row = runner(case, code, Budget(1, 3600, max_cpu_seconds=3600))
    except Exception as error:  # noqa: BLE001 - a crash on the harness side counts as a failed case
        row = {"status": "harness_error", "passed": False, "relative_l2": 1e6, "message": f"{type(error).__name__}: {error}"}
    keep = ("status", "passed", "relative_l2", "final_relative_l2", "family_correct", "identified_family", "message")
    return {"case_id": case.case_id, **{k: row.get(k) for k in keep if k in row}}


def score_programs(track, split, codes, cores):
    """{code: per-case rows} for several programs, all solves spread over the given cores."""
    count = len(cases(track, split))
    jobs = [(track, split, i, code) for code in codes for i in range(count)]
    if not jobs:
        return {}
    with multiprocessing.get_context("fork").Pool(len(cores), initializer=_init, initargs=(list(cores),)) as pool:
        rows = pool.map(solve_case, jobs, chunksize=1)
    return {code: rows[k*count:(k+1)*count] for k, code in enumerate(codes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("solver")
    parser.add_argument("--track", choices=("tex", "ext", "inventory"), required=True)
    parser.add_argument("--split", default="bench-val", choices=("bench-train", "bench-val", "bench-test"))
    parser.add_argument("--cores", default=None, help="comma-separated cores (default: all allowed)")
    args = parser.parse_args()
    cores = [int(c) for c in args.cores.split(",")] if args.cores else sorted(os.sched_getaffinity(0))
    code = open(args.solver).read()
    rows = score_programs(args.track, args.split, [code], cores)[code]
    print(json.dumps({"track": args.track, "split": args.split, "cases": len(rows),
                      "passed": sum(bool(r["passed"]) for r in rows), "rows": rows}, indent=1, default=float))


if __name__ == "__main__":
    main()
