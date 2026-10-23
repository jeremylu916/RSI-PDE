"""Build the evaluator-only references of the inventory suite and measure their accuracy.

For every case: the reference solution on the evaluation grid (data/inventory/references/<task_id>.npz), a
second solve at lower resolution to estimate the reference error (self-convergence), and 256 noise-free samples
at random evaluation nodes (time-dependent cases: t > 0) for the agents' noisy data. Writes
data/inventory/reference_report.json; per-family pass tolerances are set from it (>= 5x the estimated error).
Run on a compute node after sourcing scripts/env.sh:
    python scripts/build_inventory_references.py --workers 12
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import time

import numpy as np

from pde_rsi import equations_inventory as eq
from pde_rsi import inventory_reference as ref
from pde_rsi.inventory import FAMILIES, N_POINTS, REFERENCES, reference_path, SEALED_SPLITS, SPLITS, DATA_DIR, InvTask, evaluation_axes, make_spec

# (fine, coarse) resolution per kind
RESOLUTION = {"burgers_gauss": (2048, 1024), "burgers_random": (1024, 512), "darcy": (8, 4), "poisson": (1020, 510),
              "helmholtz": (1020, 510), "ns": (256, 128), "kolmogorov": (512, 256), "allen_cahn": (256, 128),
              "burgers2d": (256, 128), "adr": (256, 128), "elliptic4d": (63, 21)}


def solve(spec, axes, resolution):
    kind = spec["kind"]
    if kind in ("burgers_gauss", "burgers_random"):
        return ref.burgers_periodic(spec, resolution, axes["x"], axes["t"], substeps=40 if resolution > 1024 else 20)
    if kind == "convdiff":
        tt, xx = np.meshgrid(axes["t"], axes["x"], indexing="ij")
        return eq.convdiff_exact(spec, xx, tt)
    if kind == "darcy":
        return ref.darcy(spec, resolution, axes["x"], axes["y"])
    if kind in ("poisson", "helmholtz"):
        return ref.poisson_helmholtz(spec, resolution, axes["x"], axes["y"])
    if kind in ("ns", "kolmogorov"):
        return ref.vorticity(spec, resolution, list(axes["t"]), 128, dt=1e-4)
    if kind in ("allen_cahn", "burgers2d", "adr"):
        return ref.scalar_periodic(spec, resolution, list(axes["t"]), 128, dt=5e-4)
    raise ValueError(kind)


def elliptic(spec, fine, coarse):
    points_c, u_c = ref.elliptic4d(spec, coarse)
    _, u_f = ref.elliptic4d(spec, fine)
    # the coarse grid's cell centres coincide with fine cells (3i+1 in every coordinate)
    centres_f, ball_f = ref.elliptic4d_grid(fine)
    index = -np.ones((fine,)*4, dtype=np.int64)
    index[np.broadcast_to(ball_f[..., None], (fine,)*4)] = np.arange(len(u_f))
    factor = fine//coarse
    centres_c, ball_c = ref.elliptic4d_grid(coarse)
    sel = np.argwhere(np.broadcast_to(ball_c[..., None], (coarse,)*4))*factor+factor//2
    return points_c, u_f[index[tuple(sel.T)]], u_c


def build(job):
    family, split = job
    spec = make_spec(family, split)
    return build_task(InvTask(f"inv-{family}-{split}", FAMILIES[family][0], split, spec), family, split)


def claim(task_id):
    """Create data/expanded/claims/<task_id> atomically; False if a process (on any node) already claimed it.
    Claims are kept after the build: to rebuild a reference, delete its claim file together with it."""
    claims = DATA_DIR.parent/"expanded/claims"
    claims.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(claims/task_id, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return False
    return True


def build_task(task, family, split):
    """Reference, coarse check and noise-free samples of one task; written to REFERENCES/<task_id>.npz."""
    started = time.time()
    if task.task_id.startswith(("x-", "bench-")) and not claim(task.task_id):  # another node builds it
        return {"task_id": task.task_id, "family": family, "split": split, "skipped": True,
                "coarse_vs_fine_relative_l2": None, "seconds": 0.0}
    spec = task.spec
    fine, coarse = RESOLUTION.get(spec["kind"], (None, None))
    axes = evaluation_axes(task)
    rng = np.random.default_rng(int(hashlib.sha256(f"samples:{task.task_id}".encode()).hexdigest()[:8], 16))
    out = {}
    if spec["kind"] == "elliptic4d":
        points, u, u_coarse = elliptic(spec, fine, coarse)
        out.update(points=points, u=u)
        pick = rng.choice(len(u), N_POINTS, replace=False)
        out.update(sample_points=points[pick], sample_u=u[pick])
    else:
        u = solve(spec, axes, fine)
        u_coarse = solve(spec, axes, coarse) if coarse else u
        out.update(u=u, **{f"axis_{k}": v for k, v in axes.items()})
        if task.layout == "2d_steady":
            iy, ix = rng.integers(1, len(axes["y"])-1, N_POINTS), rng.integers(1, len(axes["x"])-1, N_POINTS)
            out.update(sample_x=axes["x"][ix], sample_y=axes["y"][iy], sample_u=u[iy, ix])
        elif task.layout == "2d_periodic_time":
            it = rng.integers(1, len(axes["t"]), N_POINTS)
            iy, ix = rng.integers(0, len(axes["y"]), N_POINTS), rng.integers(0, len(axes["x"]), N_POINTS)
            out.update(sample_x=axes["x"][ix], sample_y=axes["y"][iy], sample_t=axes["t"][it], sample_u=u[it, iy, ix])
        else:
            it, ix = rng.integers(1, len(axes["t"]), N_POINTS), rng.integers(1, len(axes["x"])-1, N_POINTS)
            out.update(sample_x=axes["x"][ix], sample_t=axes["t"][it], sample_u=u[it, ix])
    path = reference_path(task)  # hidden RSI-PDE Bench references go to evaluator_private/
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    error = float(np.linalg.norm(u_coarse-u)/np.linalg.norm(u)) if coarse else 0.0
    return {"task_id": task.task_id, "family": family, "split": split, "fine": fine, "coarse": coarse,
            "coarse_vs_fine_relative_l2": error, "seconds": time.time()-started,
            "solution_rms": float(np.sqrt(np.mean(u**2)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--families", nargs="*", default=list(FAMILIES))
    parser.add_argument("--splits", nargs="*", default=list(SPLITS), choices=list(SPLITS+SEALED_SPLITS[1:]))
    args = parser.parse_args()
    jobs = [(f, s) for f in args.families for s in args.splits]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    report_path = DATA_DIR/"reference_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        for row in pool.imap_unordered(build, jobs):
            report[row["task_id"]] = row
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
