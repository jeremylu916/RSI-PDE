"""Build the evaluator-only references of the pdebench-ext cases without a closed form (12b and 14b).

Each reference is solved on a fine grid and again on a coarser one; the coarse-vs-fine relative L2 estimates the
reference error, which must stay well below the 2e-3 pass tolerance. Writes data/pdebench_ext/references/<task_id>.npz
(u on the evaluation grid) and data/pdebench_ext/reference_report.json. Run after sourcing scripts/env.sh:
    python scripts/build_pdebench_ext_references.py
"""

import json
import time

import numpy as np

from pde_rsi import pdebench_ext as px
from pde_rsi import pdebench_ext_reference as ref

SETTINGS = {"12b": {"fine": (512, 120), "coarse": (256, 60)}, "14b": {"fine": (256, 100), "coarse": (128, 50)}}


def solve(case, n, substeps):
    axes, spec = px.evaluation_axes(case.truth), case.truth.spec
    if spec["problem"] == "12b":
        return ref.kuramoto_sivashinsky(spec, n, axes["t"], axes["x"], substeps)
    return ref.cahn_hilliard(spec, n, axes["t"], axes["x"], substeps)


def main():
    px.REFERENCES.mkdir(parents=True, exist_ok=True)
    report_path = px.REFERENCES.parent/"reference_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for split in px.SPLITS:
        for case in px.make_cases(split):
            problem = case.truth.spec["problem"]
            if problem not in SETTINGS:
                continue
            started = time.time()
            fine = solve(case, *SETTINGS[problem]["fine"])
            coarse = solve(case, *SETTINGS[problem]["coarse"])
            np.savez_compressed(px.reference_path(case.truth), u=fine)
            row = {"task_id": case.task_id, "fine": SETTINGS[problem]["fine"], "coarse": SETTINGS[problem]["coarse"],
                   "coarse_vs_fine_relative_l2": float(np.sqrt(np.sum((fine-coarse)**2)/np.sum(fine**2))),
                   "final_time_relative_l2": float(np.linalg.norm(fine[-1]-coarse[-1])/np.linalg.norm(fine[-1])),
                   "max_abs": float(np.abs(fine).max()), "seconds": time.time()-started}
            report[case.task_id] = row
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
