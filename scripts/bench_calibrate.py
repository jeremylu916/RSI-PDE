"""RSI-PDE Bench step 2: difficulty calibration of the 1001 expanded cases with fixed (non-LLM) probe solvers.

Each probe is a frozen solver program from an earlier run (harness v2 RSI, Claude Code, Codex, code-protocol RSI).
No model is called. Every probe solves every case of its track in the sandbox, and data/bench/calibration/
<track>/<probe>.jsonl gets one row per case (resumable). scripts/bench_report_calibration.py turns the rows into
per-template solve rates and difficulty tiers.
Run on a GPU node after sourcing scripts/env.sh, one worker per core:
    python scripts/bench_calibrate.py --tracks tex inventory manufactured --cores 42,44,46 [--shard 0/2] [--probes ...]
"""

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

from pde_rsi import expanded as X
from pde_rsi import inventory as inv
from pde_rsi import pdebench as pb
from pde_rsi import pdebench_ext as px
from pde_rsi.codegen import run_program_task
from pde_rsi.modular import load_program
from pde_rsi.runner import Budget
from pde_rsi.storage import PROJECT_ROOT

OUT = PROJECT_ROOT/"data/bench/calibration"
RUNS = PROJECT_ROOT/"runs"
# track -> probe label -> program source (a harness program directory or a native agent's retained solver)
PROBES = {
    "tex": {"v2-rsi-opus55": "v2-opus55-harness-r1-20261020/rsi/programs/frozen",
            "v2-rsi-gpt55": "v2-gpt55-harness-r1-20261020/rsi/programs/frozen",
            "claude-code-opus55": "matched-opus55-claude-code-r1-20261018/final/retained_solver.py",
            "codex-gpt55": "matched-gpt55-codex-r1-20261018/final/retained_solver.py"},
    "inventory": {"v2-rsi-opus55": "v2-inv-opus55-harness-r1-20261020/rsi/programs/frozen",
                  "v2-rsi-gpt55": "v2-inv-gpt55-harness-r1-20261020/rsi/programs/frozen",
                  "claude-code-opus55": "matched-inv-opus55-claude-code-r1-20261020/final/retained_solver.py",
                  "codex-gpt55": "matched-inv-gpt55-codex-r1-20261020/final/retained_solver.py"},
    # code protocol: one program per family, from the per-family-set runs
    "manufactured": {"rsi-opus55": ("code-agents-fine-1d-scratch-opus55-20261016", "code-agents-fine-wave2d-scratch-opus55-20261016",
                                    "code-agents-fine-group-scratch-opus55-20261016"),
                     "rsi-gpt55": ("code-agents-fine-1d-scratch-gpt55-20261016", "code-agents-fine-wave2d-scratch-gpt55-20261016",
                                   "code-agents-fine-group-scratch-gpt55-20261016")},
}
PROBES["ext"] = {}  # no frozen ext solvers exist yet; the step-5 single-run programs are added with --probe-dir


def load_probe(track, source):
    if track == "manufactured":
        program = {}
        for run in source:
            for path in sorted((RUNS/run/"rsi/programs/frozen").glob("*.py")):
                program[path.stem] = path.read_text()
        return program
    path = RUNS/source if not Path(source).is_absolute() else Path(source)
    return path.read_text() if path.suffix == ".py" else load_program(path)


def cases(track):
    if track.endswith("-cal"):  # the bench's public calibration draw of the hardened templates
        module = {"tex-cal": pb, "inventory-cal": inv}[track]
        return module.make_cases("bench-calibration")
    if track == "tex":
        return X.tex_cases()
    if track == "ext":
        return X.ext_cases()
    if track == "inventory":
        return X.inventory_cases()
    return X.manufactured_tasks()


def solve_one(job):
    track, probe, index, core = job
    os.sched_setaffinity(0, {core})
    case = CASES[track][index]
    code = PROGRAMS[(track, probe)]
    budget = Budget(1, 3600, max_cpu_seconds=3600)
    started = time.time()
    try:
        if track in ("tex", "tex-cal"):
            row = pb.run_case(case, code, budget)
        elif track == "ext":
            row = px.run_case(case, code, budget)
        elif track in ("inventory", "inventory-cal"):
            row = inv.run_case(case, code, budget)
        else:
            if case.family not in code:
                raise KeyError(f"probe has no program for {case.family}")
            row = run_program_task(case, code[case.family], budget)
    except Exception as error:  # a crashing probe is a failed case, not a failed calibration
        row = {"status": "harness_error", "passed": False, "relative_l2": None, "error": f"{type(error).__name__}: {error}"}
    case_id = getattr(case, "case_id", None) or case.task_id
    keep = ("status", "passed", "relative_l2", "final_time_relative_l2", "family_correct", "identified_family",
            "boundary_error", "initial_error", "error")
    return track, probe, {"case_id": case_id, **{k: row.get(k) for k in keep if k in row},
                          "seconds": round(time.time()-started, 2)}


CASES, PROGRAMS = {}, {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", nargs="+", default=["tex", "inventory", "manufactured"],
                        choices=("tex", "ext", "inventory", "manufactured", "tex-cal", "inventory-cal"))
    parser.add_argument("--probes", nargs="*", default=None, help="probe labels (default: all of each track)")
    parser.add_argument("--probe-dir", nargs="*", default=[], metavar="TRACK:LABEL:PATH",
                        help="extra probes, e.g. ext:single-opus55:runs/<run>/single/programs/frozen")
    parser.add_argument("--cores", required=True, help="comma-separated OS cores, one worker each")
    parser.add_argument("--limit", type=int, default=None, help="smoke test: only the first N solves")
    parser.add_argument("--shard", default="0/1", help="i/n: this process takes every n-th solve starting at i")
    args = parser.parse_args()
    for item in args.probe_dir:
        track, label, path = item.split(":", 2)
        PROBES[track][label] = str(Path(path).resolve())
    cores = [int(c) for c in args.cores.split(",")]
    jobs = []
    for track in args.tracks:
        CASES[track] = cases(track)
        for probe, source in PROBES[track.removesuffix("-cal")].items():
            if args.probes and probe not in args.probes:
                continue
            PROGRAMS[(track, probe)] = load_probe(track.removesuffix("-cal"), source)
            path = OUT/track/f"{probe}.jsonl"
            done = {json.loads(line)["case_id"] for line in path.read_text().splitlines()} if path.exists() else set()
            jobs += [(track, probe, i) for i, c in enumerate(CASES[track])
                     if (getattr(c, "case_id", None) or c.task_id) not in done]
    # interleave probes and tracks so slow cases spread over the workers
    jobs.sort(key=lambda j: (j[2], j[0], j[1]))
    index, total = map(int, args.shard.split("/"))
    jobs = jobs[index::total][:args.limit]
    print(f"{len(jobs)} solves on {len(cores)} cores", flush=True)
    with multiprocessing.get_context("fork").Pool(len(cores), initializer=_claim_core, initargs=(cores,)) as pool:
        for count, (track, probe, row) in enumerate(pool.imap_unordered(_run, jobs), 1):
            path = OUT/track/f"{probe}.jsonl"  # appends from several processes: one short line per write
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as handle:
                handle.write(json.dumps(row, default=float)+"\n")
            if count % 50 == 0 or count == len(jobs):
                print(f"{count}/{len(jobs)} {time.strftime('%H:%M:%S')}", flush=True)


CORE = None


def _claim_core(cores):
    global CORE
    CORE = cores[(multiprocessing.current_process()._identity[0]-1) % len(cores)]


def _run(job):
    return solve_one((*job, CORE))


if __name__ == "__main__":
    main()
