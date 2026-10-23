"""Build the references of the 1001-case expanded release that have no closed form.

    --suite ext        Problems 12b and 14b variants (Fourier-spectral ETDRK4; a few minutes in total)
    --suite inventory  the 240 inventory variants (several CPU-hours; run on a compute node, one worker per core)

References go where each suite's evaluator reads them (data/pdebench_ext/references, data/inventory/references), and
every one is checked against a coarser solve; data/expanded/reference_report-<suite>-<host>-<shard>.json records the
errors. Inventory jobs claim their case in data/expanded/claims, so several nodes can run this at once.
Run after sourcing scripts/env.sh, e.g.
    python scripts/build_expanded_references.py --suite inventory --workers 14 [--shard 0/2]
"""

import argparse
import importlib.util
import json
import multiprocessing
import os
import socket
import time
from pathlib import Path

import numpy as np

from pde_rsi import expanded as X
from pde_rsi import pdebench_ext as px
from pde_rsi.storage import PROJECT_ROOT

REPORTS = PROJECT_ROOT/"data/expanded"  # reference_report-<suite>-<shard>.json, one file per shard


def load_inventory_builder():
    spec = importlib.util.spec_from_file_location("build_inventory_references",
                                                  PROJECT_ROOT/"scripts/build_inventory_references.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ext_job(case):
    builder = importlib.util.spec_from_file_location("ext_refs", PROJECT_ROOT/"scripts/build_pdebench_ext_references.py")
    module = importlib.util.module_from_spec(builder)
    builder.loader.exec_module(module)
    started = time.time()
    problem = case.truth.spec["problem"]
    fine = module.solve(case, *module.SETTINGS[problem]["fine"])
    coarse = module.solve(case, *module.SETTINGS[problem]["coarse"])
    px.reference_path(case.truth).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(px.reference_path(case.truth), u=fine)
    return {"task_id": case.task_id, "coarse_vs_fine_relative_l2": float(np.sqrt(np.sum((fine-coarse)**2)/np.sum(fine**2))),
            "final_time_relative_l2": float(np.linalg.norm(fine[-1]-coarse[-1])/np.linalg.norm(fine[-1])),
            "seconds": time.time()-started}


def inventory_job(case):
    module = load_inventory_builder()
    row = module.build_task(case.truth, case.truth.spec["family_key"], case.truth.split)
    return {k: row[k] for k in ("task_id", "coarse_vs_fine_relative_l2", "seconds")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("ext", "inventory"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard", default="0/1", help="i/n: build every n-th missing reference starting at i")
    parser.add_argument("--source", choices=("expanded", "bench"), default="expanded",
                        help="bench: the RSI-PDE Bench splits (hidden-test references go to evaluator_private/)")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if args.source == "bench":
        from pde_rsi import inventory as inv
        from pde_rsi.bench import splits as S
        track = "ext" if args.suite == "ext" else "inventory"
        wrap = px.ExtCase if track == "ext" else inv.InvCase
        cases = [wrap(t.task_id, t) for split in ("train", "val", "test", "calibration") for t in S.tasks(track, split)]
    if args.suite == "ext":
        cases = [c for c in (cases if args.source == "bench" else X.ext_cases())
                 if c.truth.spec["problem"] in ("12b", "14b")]
        done = lambda c: px.reference_path(c.truth).exists()  # noqa: E731
        job = ext_job
    else:
        from pde_rsi import inventory as inv
        cases = cases if args.source == "bench" else X.inventory_cases()
        done = lambda c: inv.reference_path(c.truth).exists()  # noqa: E731
        job = inventory_job
    index, total = map(int, args.shard.split("/"))
    pending = [c for c in cases if not done(c)][index::total]
    # longest first, so the slow families do not finish last
    weight = {"kolmogorov": 25, "ns": 6, "adr": 5, "allen_cahn": 3, "elliptic4d": 1.3, "burgers_gauss": 1.2, "14b": 1}
    pending.sort(key=lambda c: -weight.get(c.truth.spec.get("family_key", c.truth.spec.get("problem")), 0.5))
    print(f"{len(pending)} references to build", flush=True)
    reports = REPORTS if args.source == "expanded" else PROJECT_ROOT/"evaluator_private/bench"  # hidden task ids
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports/f"reference_report-{args.source}-{args.suite}-{socket.gethostname().split('.')[0]}-{index}of{total}.json"
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        for row in pool.imap_unordered(job, pending):
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            report[row["task_id"]] = row
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
