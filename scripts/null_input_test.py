"""Null-input test: do frozen pdebench solvers notice when the measurements are random numbers?

Each null case is a public pdebench case (evolution or development split) whose 256 measurements u are replaced
by random numbers (NULL_TYPES):
- gaussian: i.i.d. Gaussian numbers with the clean samples' RMS;
- uniform: i.i.d. uniform numbers over the range of the real measurements;
- shuffled: the real measurements permuted across the sample points (same values, no relation to the points);
- smooth: a random low-wavenumber field in the case's coordinates, scaled to the clean RMS, plus Gaussian noise
  of the stated noise_std. It looks like noisy samples of a smooth solution but solves no library equation
  with the case's forcing, initial, and boundary data.
Everything else is unchanged: sample points, domain, the stated noise_std (1 % of the clean RMS), and the
forcing, initial, and boundary callables of the real case. So the data claim to be 1 %-noise measurements of a
library equation, but carry no information about it.

The solver contract has no "not a PDE" answer, so a solver can only show detection indirectly. For every solve
this script records the claimed class and parameters, the error of the returned solution against the real
case's exact solution, and the data misfit: the RMS of (returned solution at the sample points - data) divided
by noise_std. For data that really come from the reported equation the misfit is about 1; the pre-registered
consistency check flags a case as "not this PDE" when the misfit exceeds MISFIT_THRESHOLD. Each program is run
on the real cases too, which gives the check's false-alarm rate.

Solves run on the CPU (gpu=False); the matched-study solvers are all numerical. Each (null type, program) pair
is one job, written to <output>/<null type>/programs/<program>/results.json. Run on a GPU node after sourcing
scripts/env.sh, one shard per dedicated core:
    taskset -c 84 python scripts/null_input_test.py --output runs/null-input-test-20261018 \
        --null-types uniform shuffled smooth --shard 0/14
    python scripts/null_input_test.py --output runs/null-input-test-20261018 --summarize
--yield-to-study stops a shard before the next solve once the matched study's GPT-5.5 sealed scoring has
started (needed only on h016 cores 108/110, which that scoring uses).
"""

import argparse
from dataclasses import dataclass
import glob
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pde_rsi.pdebench import N_POINTS, PROTOCOL_VERSION, BenchCase, make_cases, run_case
from pde_rsi.runner import Budget
from pde_rsi.storage import project_path

MISFIT_THRESHOLD = 2.0  # residual RMS / noise_std; fixed before any null result was seen
SPLITS = ("evolution", "development")
NULL_TYPES = ("gaussian", "uniform", "shuffled", "smooth")
SMOOTH_MODES = 6  # random product-of-sinusoids terms in the smooth field
SMOOTH_MAX_WAVENUMBER = 3  # half-periods per axis over the domain
STOP_FILES = "runs/matched-sealed-*-gpt55-20261018.log"  # the matched study needs cores 108/110 again


@dataclass(frozen=True)
class NullCase(BenchCase):
    """A real case whose measurements are replaced by random numbers of kind null_type (see NULL_TYPES)."""
    null_type: str = "gaussian"

    def data(self):
        real = BenchCase.data(self)
        seed = int(hashlib.sha256(f"{PROTOCOL_VERSION}:null-{self.null_type}:{self.task_id}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        clean_rms = real["noise_std"]/self.noise_fraction
        if self.null_type == "gaussian":
            u = clean_rms*rng.standard_normal(N_POINTS)
        elif self.null_type == "uniform":
            u = rng.uniform(real["u"].min(), real["u"].max(), N_POINTS)
        elif self.null_type == "shuffled":
            u = rng.permutation(real["u"])
        elif self.null_type == "smooth":
            u = self._smooth_field(real, rng)
            u = clean_rms*u/np.sqrt(np.mean(u**2))+real["noise_std"]*rng.standard_normal(N_POINTS)
        else:
            raise ValueError(f"null_type must be one of {NULL_TYPES}")
        return {**real, "u": u}

    def _smooth_field(self, real, rng):
        s = self.truth.spec
        bounds = {"x": (s["a"], s["b"]), "y": (s["a"], s["b"]), "t": (0.0, s["T"])}
        field = np.zeros(N_POINTS)
        for _ in range(SMOOTH_MODES):
            term = np.full(N_POINTS, rng.standard_normal())
            for axis in self.coordinates():
                low, high = bounds[axis]
                k = rng.integers(0, SMOOTH_MAX_WAVENUMBER+1)
                term *= np.cos(k*np.pi*(real[axis]-low)/(high-low)+rng.uniform(0, 2*np.pi))
            field += term
        return field


def programs():
    """Label -> frozen program path for the matched-study Opus runs of 2026-10-18."""
    found = {}
    for r in (1, 2, 3):
        for arm in ("single", "rsi"):
            found[f"harness-r{r}-{arm}"] = f"runs/matched-opus55-harness-r{r}-20261018/{arm}/programs/frozen/identify.py"
        found[f"claude-code-r{r}"] = f"runs/matched-opus55-claude-code-r{r}-20261018/final/retained_solver.py"
    return found


def misfit(case, solution_path):
    """RMS of (returned solution at the sample points - data) / noise_std, or None without a solution."""
    path = project_path(solution_path)
    if not path.exists():
        return None
    data = case.data()
    with np.load(path) as archive:
        x, t, u = archive["x"], archive["t"], archive["u"]
        y = archive["y"] if "y" in archive.files else None
    dimension = case.truth.dimension
    if dimension == 0:
        predicted = np.interp(data["t"], t, u.reshape(len(t), -1)[:, 0])
    elif case.truth.steady:
        predicted = np.interp(data["x"], x, u.reshape(-1, len(x))[-1])
    elif dimension == 1:
        grid = RegularGridInterpolator((t, x), u.reshape(len(t), len(x)), bounds_error=False, fill_value=None)
        predicted = grid(np.stack([data["t"], data["x"]], axis=-1))
    else:
        grid = RegularGridInterpolator((t, y, x), u.reshape(len(t), len(y), len(x)), bounds_error=False,
                                       fill_value=None)
        predicted = grid(np.stack([data["t"], data["y"], data["x"]], axis=-1))
    value = float(np.sqrt(np.mean((predicted-data["u"])**2))/data["noise_std"])
    return value if math.isfinite(value) else None


def run_program(null_type, label, path, output, yield_to_study=False):
    target = output/null_type/"programs"/label
    if (target/"results.json").exists():
        print(f"{null_type} {label}: done earlier", flush=True)
        return
    code = project_path(path).read_text()
    real = [case for split in SPLITS for case in make_cases(split)]
    cases = [("real", case) for case in real]+[("null", NullCase(c.case_id, c.truth, c.noise_fraction, null_type))
                                               for c in real]
    budget = Budget(len(cases), 7200, max_cpu_seconds=7200)
    results = []
    for kind, case in cases:
        if yield_to_study and glob.glob(str(project_path(STOP_FILES))):
            sys.exit(f"stopping: the matched study's sealed scoring has started ({STOP_FILES})")
        solution = target/"solutions"/kind/f"{case.case_id}.npz"
        if solution.exists():
            solution.unlink()  # a partial earlier attempt
        result = run_case(case, code, budget, solution, gpu=False)
        record = {"program": label, "null_type": null_type, "kind": kind, "case_id": case.case_id, "split": case.split,
                  "problem": case.truth.problem, "true_family": case.family, "status": result["status"],
                  "message": result.get("message"), "identified_family": result.get("identified_family"),
                  "identified_parameters": result.get("identified_parameters"),
                  "relative_l2_vs_real_truth": result.get("relative_l2"), "passed": result.get("passed"),
                  "wall_seconds": result.get("wall_seconds"), "misfit": misfit(case, solution)}
        results.append(record)
        print(null_type, label, kind, case.case_id, record["status"], record["identified_family"],
              None if record["misfit"] is None else round(record["misfit"], 2), flush=True)
    (target/"results.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")


def flagged(r):  # no finite misfit (crash, no solution) also counts as "not consistent with the claimed PDE"
    return r["misfit"] is None or r["misfit"] > MISFIT_THRESHOLD


def fmt(value):
    return "-" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)


def summarize(output):
    """Per-type summary.{json,md} under <output>/<type>/ and a cross-type summary.md under <output>/."""
    overall, null_types = {}, [t for t in NULL_TYPES if (output/t/"programs").is_dir()]
    if not null_types:
        sys.exit("no results yet")
    for null_type in null_types:
        overall[null_type] = summarize_type(output/null_type, null_type)
    lines = ["# Null-input test: all null types", "",
             f"Consistency check: misfit = RMS(solution at sample points - data) / noise_std > {MISFIT_THRESHOLD}.",
             "Each row pools the 9 frozen Opus solvers of 2026-10-18 on the 32 public cases and their null twins.",
             "", "| Null type | Null: claimed a class | Null: flagged | Real: flagged (false alarm) | "
             "Real misfit max | Null misfit median / min | Null: still passes vs real truth |", "|" + "---|"*7]
    for null_type, s in overall.items():
        lines.append(f"| {null_type} | {s['null_claimed_library_class']}/{s['null_cases']} | "
                     f"{s['null_flagged']}/{s['null_cases']} | {s['real_flagged']}/{s['real_cases']} | "
                     f"{fmt(s['real_misfit_max'])} | {fmt(s['null_misfit_median'])} / {fmt(s['null_misfit_min'])} | "
                     f"{s['null_passed_vs_real_truth']}/{s['null_cases']} |")
    (output/"summary.md").write_text("\n".join(lines)+"\n")
    (output/"summary.json").write_text(json.dumps({"misfit_threshold": MISFIT_THRESHOLD, "overall": overall},
                                                  indent=2)+"\n")
    print("\n".join(lines))


def summarize_type(output, null_type):
    rows = [r for f in sorted((output/"programs").glob("*/results.json")) for r in json.loads(f.read_text())]

    def stats(group):
        null = [r for r in group if r["kind"] == "null"]
        real = [r for r in group if r["kind"] == "real"]
        real_misfit = [r["misfit"] for r in real if r["misfit"] is not None]
        null_misfit = [r["misfit"] for r in null if r["misfit"] is not None]
        return {"null_cases": len(null), "real_cases": len(real),
                "null_claimed_library_class": sum(r["status"] == "ok" and r["identified_family"] is not None
                                                  for r in null),
                "null_failed": sum(r["status"] != "ok" for r in null),
                "null_flagged": sum(map(flagged, null)), "real_flagged": sum(map(flagged, real)),
                "null_same_class_as_template": sum(r["identified_family"] == r["true_family"] for r in null),
                "null_passed_vs_real_truth": sum(bool(r["passed"]) for r in null),
                "real_passed": sum(bool(r["passed"]) for r in real),
                "real_misfit_median": statistics.median(real_misfit) if real_misfit else None,
                "real_misfit_max": max(real_misfit) if real_misfit else None,
                "null_misfit_median": statistics.median(null_misfit) if null_misfit else None,
                "null_misfit_min": min(null_misfit) if null_misfit else None}

    by_program = {p: stats([r for r in rows if r["program"] == p]) for p in sorted({r["program"] for r in rows})}
    summary = {"misfit_threshold": MISFIT_THRESHOLD, "overall": stats(rows), "by_program": by_program}
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    lines = [f"# Null-input test ({null_type} measurements)", "",
             f"Consistency check: misfit = RMS(solution at sample points - data) / noise_std > {MISFIT_THRESHOLD}.",
             "", "| Program | Null: claimed a class | Null: flagged | Real: flagged (false alarm) | "
             "Real misfit median / max | Null misfit median / min | Null: same class as template | "
             "Null: still passes vs real truth |", "|" + "---|"*8]
    for label, s in [*by_program.items(), ("**all**", summary["overall"])]:
        lines.append(f"| {label} | {s['null_claimed_library_class']}/{s['null_cases']} | "
                     f"{s['null_flagged']}/{s['null_cases']} | {s['real_flagged']}/{s['real_cases']} | "
                     f"{fmt(s['real_misfit_median'])} / {fmt(s['real_misfit_max'])} | "
                     f"{fmt(s['null_misfit_median'])} / {fmt(s['null_misfit_min'])} | "
                     f"{s['null_same_class_as_template']}/{s['null_cases']} | "
                     f"{s['null_passed_vs_real_truth']}/{s['null_cases']} |")
    (output/"summary.md").write_text("\n".join(lines)+"\n")
    return summary["overall"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--null-types", nargs="+", choices=NULL_TYPES, default=["gaussian"])
    parser.add_argument("--shard", default="0/1", help="i/n: run every n-th (null type, program) job starting at i")
    parser.add_argument("--programs", nargs="+", help="labels to run (default: all)")
    parser.add_argument("--yield-to-study", action="store_true",
                        help=f"stop before the next solve once {STOP_FILES} exists (h016 cores 108/110 only)")
    parser.add_argument("--limit-cases", type=int, help="smoke test: only the first N real cases and their twins")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    output = project_path(args.output)
    if args.summarize:
        summarize(output)
        return
    if args.limit_cases:
        global SPLITS
        SPLITS = ("evolution",)
        original = make_cases
        globals()["make_cases"] = lambda split: original(split)[:args.limit_cases]
    index, count = map(int, args.shard.split("/"))
    jobs = [(null_type, label, path) for null_type in args.null_types for label, path in programs().items()
            if not args.programs or label in args.programs]
    for null_type, label, path in jobs[index::count]:
        run_program(null_type, label, path, output, args.yield_to_study)


if __name__ == "__main__":
    main()
