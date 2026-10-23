"""pde_benchmarks.tex Problems 12-16 under the student protocol (suite "pdebench-ext").

Problem 12 Kuramoto-Sivashinsky, 13 incompressible Navier-Stokes, 14 Cahn-Hilliard, 15 Maxwell (TE_z, time
domain), 16 time-harmonic electromagnetic (curl-curl). Each problem has two cases (a, b), and every split
(training = evolution, validation = development, sealed test) holds one instance of each case with its own numbers:
manufactured cases draw theirs per split, literature cases use the values in FIXED.

These problems need periodic domains, a fourth-order term and vector fields, so they form their own suite rather
than extending the 20-case pdebench suite (whose splits and results stay unchanged). The task is the same as
there: identify the class and coefficients from 256 noisy samples and black-box inputs, then return the solution.
Exact solutions come from equations_ext.py. Cases 12b and 14b use stored high-resolution references
(scripts/build_pdebench_ext_references.py; evaluator-only).
"""

from dataclasses import dataclass, field
import hashlib
import json
import math

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import RegularGridInterpolator

from . import equations_ext as eq
from .code_agents import CodeProtocol
from .codegen import ALLOWED_MODULES, MAX_CODE_CHARS, MAX_OUTPUT_VALUES, STUB_PROGRAM, run_program_task
from .evaluator import failed_result
from .runner import BudgetExceeded
from .storage import PROJECT_ROOT

VERSION = "pdebench-ext-v1"
EVALUATOR_VERSION = "pdebench-ext-evaluator-v1"
SPLITS = ("evolution", "development", "test")
PREFIX = {"evolution": "et", "development": "ev", "test": "er"}
REFERENCES = PROJECT_ROOT/"data/pdebench_ext/references"
N_POINTS = 256
NOISE_FRACTION = 0.01
TOLERANCE = 2e-3
CONSTRAINT_TOLERANCE = 1e-10
MEMORY_MB = 16384
PROGRAM_KEY = "identify"
SELF_TEST_KINDS = 4

LIBRARY = {
    "kuramoto_sivashinsky": {"equation": "u_t + u*u_x + u_xx + u_xxxx = f(x,t) (periodic in x)", "parameters": {}},
    "navier_stokes_2d": {"equation": "u_t + (u.grad)u + grad p - nu*Lap u = f, div u = 0; velocity (u, v), pressure p "
                                     "(unit density; p determined up to a constant)",
                         "parameters": {"nu": "kinematic viscosity > 0 (Re = 1/nu)"}},
    "cahn_hilliard_2d": {"equation": "u_t = M*Lap(u^3 - u - eps^2*Lap u) + f (periodic)",
                         "parameters": {"M": "mobility > 0", "eps": "interface parameter > 0"}},
    "maxwell_te": {"equation": "E_t = c*curl(H) - J, H_t = -c*curl(E); E = (Ex, Ey), H = Hz, curl(H) = (Hz_y, -Hz_x), "
                               "curl(E) = Ey_x - Ex_y; perfect conductor: tangential E = 0 on the boundary",
                   "parameters": {"c": "wave speed > 0"}},
    "maxwell_time_harmonic": {"equation": "curl curl E - k^2 E = f, E = (Ex, Ey), curl curl E = (w_y, -w_x) with "
                                          "w = Ey_x - Ex_y; perfect conductor: tangential E = 0 on the boundary",
                              "parameters": {"k": "wavenumber > 0"}},
}
# case: (library class, layout, max seconds, title)
CASES = {
    "12a": ("kuramoto_sivashinsky", "1d_periodic_time", 90.0, "Kuramoto-Sivashinsky, manufactured"),
    "12b": ("kuramoto_sivashinsky", "1d_periodic_time", 90.0, "Kuramoto-Sivashinsky, chaotic (Kassam-Trefethen)"),
    "13a": ("navier_stokes_2d", "2d_periodic_time", 300.0, "Navier-Stokes, Taylor-Green vortex"),
    "13b": ("navier_stokes_2d", "2d_steady", 300.0, "Navier-Stokes, Kovasznay flow"),
    "14a": ("cahn_hilliard_2d", "2d_periodic_time", 300.0, "Cahn-Hilliard, manufactured"),
    "14b": ("cahn_hilliard_2d", "2d_periodic_time", 300.0, "Cahn-Hilliard, spinodal decomposition"),
    "15a": ("maxwell_te", "2d_time", 300.0, "Maxwell TE_z, resonant cavity mode"),
    "15b": ("maxwell_te", "2d_time", 300.0, "Maxwell TE_z, current-driven (manufactured)"),
    "16a": ("maxwell_time_harmonic", "2d_steady", 300.0, "Time-harmonic electromagnetic, low frequency"),
    "16b": ("maxwell_time_harmonic", "2d_steady", 300.0, "Time-harmonic electromagnetic, high frequency"),
}
CH_MODES = {
    "evolution": [(0.1, 1, 0, 0.0), (0.08, 0, 2, 0.5), (0.06, 3, 1, 1.0), (0.05, 2, 3, 2.0), (0.04, 5, 2, 0.3)],
    "development": [(0.1, 0, 1, 0.2), (0.07, 2, 1, 1.1), (0.06, 1, 3, 2.2), (0.05, 4, 1, 0.7), (0.04, 3, 4, 1.6)],
    "test": [(0.1, 1, 1, 0.4), (0.07, 3, 0, 1.3), (0.06, 2, 2, 0.1), (0.05, 1, 4, 2.6), (0.04, 5, 3, 0.9)],
}
TWO_PI = 2*math.pi
FIXED = {
    "12b": {s: dict(a=0.0, b=32*math.pi, T=T) for s, T in (("evolution", 30.0), ("development", 60.0), ("test", 150.0))},
    "13a": {"evolution": dict(a=0.0, b=TWO_PI, c=0.0, d=TWO_PI, k=1, nu=0.1, T=1.0),
            "development": dict(a=0.0, b=TWO_PI, c=0.0, d=TWO_PI, k=2, nu=0.05, T=1.0),
            "test": dict(a=0.0, b=TWO_PI, c=0.0, d=TWO_PI, k=3, nu=0.01, T=2.0)},
    "13b": {s: dict(a=-0.5, b=1.0, c=-0.5, d=1.5, nu=1/re) for s, re in (("evolution", 20.0), ("development", 40.0),
                                                                         ("test", 80.0))},
    "14b": {s: dict(a=0.0, b=TWO_PI, c=0.0, d=TWO_PI, M=1.0, eps=eps, T=T, modes=CH_MODES[s])
            for s, eps, T in (("evolution", 0.15, 0.5), ("development", 0.12, 0.4), ("test", 0.1, 0.3))},
    "15a": {"evolution": dict(a=0.0, b=1.0, c=0.0, d=1.0, m=1, n=1, c_speed=1.0, T=1.0),
            "development": dict(a=0.0, b=1.0, c=0.0, d=1.5, m=2, n=1, c_speed=1.5, T=1.0),
            "test": dict(a=0.0, b=2.0, c=0.0, d=1.0, m=3, n=2, c_speed=0.8, T=2.0)},
    # k lies between consecutive cavity eigenvalues pi*sqrt(m^2 + n^2), so each problem is uniquely solvable.
    "16a": {"evolution": dict(A=1.0, B=0.5, p=1, n=1, m=1, q=1, k=2.0),
            "development": dict(A=0.8, B=1.2, p=2, n=1, m=1, q=2, k=1.5),
            "test": dict(A=1.0, B=1.0, p=1, n=2, m=2, q=1, k=2.5)},
    "16b": {"evolution": dict(A=1.0, B=0.7, p=2, n=3, m=3, q=1, k=10.6),
            "development": dict(A=0.9, B=1.1, p=1, n=2, m=2, q=3, k=7.6),
            "test": dict(A=1.2, B=0.8, p=3, n=2, m=1, q=4, k=12.0)},
}


def _rng(*parts):
    return np.random.default_rng(int(hashlib.sha256(":".join([VERSION, *map(str, parts)]).encode()).hexdigest()[:12], 16))


def make_spec(case, split):
    klass, layout, _, _ = CASES[case]
    spec = {"problem": case, "class": klass, "layout": layout, "a": 0.0, "b": 1.0, "c": 0.0, "d": 1.0, "T": None}
    rng = _rng(case, split)
    if case == "12a":
        spec.update(b=float(rng.uniform(2*math.pi, 6*math.pi)), A=float(rng.uniform(0.5, 1.5)),
                    beta=float(rng.uniform(1.0, 2.0)), T=float(rng.uniform(0.5, 1.0)))
    elif case == "14a":
        length = float(rng.uniform(2.0, 4.0))
        spec.update(b=length, d=length, A=float(rng.uniform(0.3, 0.8)), beta=float(rng.uniform(1.0, 2.0)),
                    M=float(rng.uniform(0.5, 1.5)), eps=float(rng.uniform(0.05, 0.15)), T=float(rng.uniform(0.1, 0.3)))
    elif case == "15b":
        spec.update(b=float(rng.uniform(1.0, 2.0)), d=float(rng.uniform(1.0, 2.0)), A=float(rng.uniform(0.5, 1.5)),
                    beta=float(rng.uniform(1.0, 2.0)), c_speed=float(rng.uniform(0.5, 2.0)), T=float(rng.uniform(0.5, 1.0)))
    else:
        spec.update(FIXED[case][split])
    if "c_speed" in spec:  # equations_ext names the wave speed "c"; the domain's y-start stays 0 for Problem 15
        spec["c"] = spec.pop("c_speed")
        spec["c0"] = 0.0
    if case == "12b":
        spec["modes"] = None
    if case in ("14b",):
        spec["modes"] = [list(m) for m in spec["modes"]]
    return spec


def y_range(spec):
    """Problem 15 stores its wave speed in "c", so its y-interval is (c0, d) = (0, d)."""
    return (spec["c0"] if "c0" in spec else spec["c"]), spec["d"]


def true_parameters(spec):
    klass = spec["class"]
    if klass == "navier_stokes_2d":
        return {"nu": spec["nu"]}
    if klass == "cahn_hilliard_2d":
        return {"M": spec["M"], "eps": spec["eps"]}
    if klass == "maxwell_te":
        return {"c": spec["c"]}
    if klass == "maxwell_time_harmonic":
        return {"k": spec["k"]}
    return {}


@dataclass(frozen=True)
class ExtTask:
    task_id: str
    family: str  # library class (used by the gate's per-class grouping)
    split: str
    spec: dict = field(hash=False, compare=False)
    max_seconds: float = 90.0
    memory_mb: int = MEMORY_MB
    relative_l2_tolerance: float = TOLERANCE
    boundary_tolerance: float = CONSTRAINT_TOLERANCE
    initial_tolerance: float = CONSTRAINT_TOLERANCE

    @property
    def layout(self):
        return self.spec["layout"]

    @property
    def components(self):
        return eq.COMPONENTS[eq.problem_number(self.spec)]

    def to_dict(self):
        return {"task_id": self.task_id, "family": self.family, "split": self.split}


def evaluation_axes(task):
    s, layout = task.spec, task.layout
    if layout == "1d_periodic_time":
        return {"x": np.linspace(s["a"], s["b"], 512), "t": np.linspace(0.0, s["T"], 501)}
    y0, y1 = y_range(s)
    axes = {"x": np.linspace(s["a"], s["b"], 129), "y": np.linspace(y0, y1, 129)}
    if layout in ("2d_periodic_time", "2d_time"):
        axes["t"] = np.linspace(0.0, s["T"], 21)
    return axes


HIDDEN_REFERENCES = PROJECT_ROOT/"evaluator_private/bench/references"  # RSI-PDE Bench hidden test


def reference_path(task):
    if task.task_id.startswith("bench-") and task.split == "test":
        return HIDDEN_REFERENCES/f"{task.task_id}.npz"
    return REFERENCES/f"{task.task_id}.npz"


_REFERENCE = {}


def reference(task):
    """Solution on the evaluation grid: [t, x, comp] (1-D) or [t, y, x, comp] (2-D; t = [0] when steady)."""
    if task.task_id in _REFERENCE:
        return _REFERENCE[task.task_id]
    axes, s = evaluation_axes(task), task.spec
    if not eq.has_closed_form(s):
        with np.load(reference_path(task)) as archive:
            u = archive["u"]
        value = u[..., None]
    elif task.layout == "1d_periodic_time":
        tt, xx = np.meshgrid(axes["t"], axes["x"], indexing="ij")
        value = eq.exact(s, xx, tt)
    else:
        t = axes.get("t", np.zeros(1))
        tt, yy, xx = np.meshgrid(t, axes["y"], axes["x"], indexing="ij")
        value = eq.exact(s, xx, tt, yy)
    _REFERENCE[task.task_id] = value
    return value


@dataclass(frozen=True)
class ExtCase:
    case_id: str
    truth: ExtTask
    noise_fraction: float = NOISE_FRACTION

    @property
    def task_id(self):
        return self.truth.task_id

    @property
    def family(self):
        return self.truth.family

    @property
    def split(self):
        return self.truth.split

    @property
    def max_seconds(self):
        return self.truth.max_seconds

    def data(self):
        """n_data noisy samples of every component at random evaluation nodes (t > 0 when time-dependent)."""
        axes, ref = evaluation_axes(self.truth), reference(self.truth)
        rng = np.random.default_rng(int(hashlib.sha256(f"{VERSION}:samples:{self.task_id}".encode()).hexdigest()[:8], 16))
        out = {"x": None, "y": None, "t": None}
        if self.truth.layout == "1d_periodic_time":
            it, ix = rng.integers(1, len(axes["t"]), N_POINTS), rng.integers(0, len(axes["x"]), N_POINTS)
            clean = ref[it, ix]
            out.update(x=axes["x"][ix], t=axes["t"][it])
        else:
            it = (rng.integers(1, len(axes["t"]), N_POINTS) if "t" in axes else np.zeros(N_POINTS, dtype=int))
            iy, ix = rng.integers(0, len(axes["y"]), N_POINTS), rng.integers(0, len(axes["x"]), N_POINTS)
            clean = ref[it, iy, ix]
            out.update(x=axes["x"][ix], y=axes["y"][iy], t=axes["t"][it] if "t" in axes else None)
        noise = np.random.default_rng(int(hashlib.sha256(f"{VERSION}:noise:{self.task_id}".encode()).hexdigest()[:8], 16))
        std = self.noise_fraction*np.sqrt(np.mean(clean**2, axis=0))
        out["u"] = clean+std*noise.standard_normal(clean.shape)
        out["noise_std"] = std.tolist()
        return out

    def public_view(self):
        s, task = self.truth.spec, self.truth
        y0, y1 = y_range(s)
        domain = {"x": [s["a"], s["b"]]}
        if task.layout != "1d_periodic_time":
            domain["y"] = [y0, y1]
        if s.get("T"):
            domain["t"] = [0.0, s["T"]]
        boundary = {"1d_periodic_time": "periodic in x",
                    "2d_periodic_time": "periodic in x and y"}.get(task.layout)
        if boundary is None:
            boundary = ("Dirichlet velocity (u, v) = boundary(x, y) on the four edges; no pressure condition"
                        if task.family == "navier_stokes_2d" else
                        "perfect conductor: tangential E = 0 (Ex = 0 on the bottom and top edges, Ey = 0 on the "
                        "left and right edges)")
        axes = evaluation_axes(task)
        return {"case_id": self.case_id, "split": self.split, "layout": task.layout, "domain": domain,
                "components": list(task.components), "forcing_components": list(eq.FORCING_COMPONENTS[eq.problem_number(s)]),
                "boundary_description": boundary, "n_data": N_POINTS, "noise_std": self.data()["noise_std"],
                "relative_l2_tolerance": task.relative_l2_tolerance, "boundary_tolerance": task.boundary_tolerance,
                "initial_tolerance": task.initial_tolerance, "max_seconds": task.max_seconds, "memory_mb": task.memory_mb,
                "evaluation_grid": {k: len(v) for k, v in axes.items()}}


def make_cases(split, noise_fraction=NOISE_FRACTION):
    if split.startswith("bench-"):  # RSI-PDE Bench splits (src/pde_rsi/bench/splits.py)
        from .bench.splits import bench_cases
        return bench_cases("ext", split, noise_fraction)
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    tasks = []
    for case, (klass, _, seconds, _) in CASES.items():
        tasks.append(ExtTask(f"ext-{case}-{split}", klass, split, make_spec(case, split), max_seconds=seconds))
    tasks.sort(key=lambda t: hashlib.sha256(f"{VERSION}:{t.task_id}".encode()).hexdigest())
    return [ExtCase(f"case-{PREFIX[split]}{index:02d}", task, noise_fraction) for index, task in enumerate(tasks)]


# ---------------------------------------------------------------- evaluator
def _check_axis(axis, low, high, name):
    if axis.ndim != 1 or len(axis) < 3 or not np.all(np.diff(axis) > 0):
        raise ValueError(f"{name} must be strictly increasing with at least 3 points")
    if not np.allclose(np.diff(axis), axis[1]-axis[0], rtol=1e-9, atol=1e-13):
        raise ValueError(f"Expected a uniform {name} grid")
    if not (np.isclose(axis[0], low, atol=1e-12, rtol=0) and np.isclose(axis[-1], high, atol=1e-12, rtol=0)):
        raise ValueError(f"{name} grid must run from {low} to {high} including both ends")


def _norms(error, truth, axes, layout):
    """Space integrals of error^2 and truth^2 (summed over components) for each time, then space-time values."""
    def space(values):
        values = np.sum(values**2, axis=-1)
        if layout == "1d_periodic_time":
            return trapezoid(values, axes["x"], axis=1)
        return trapezoid(trapezoid(values, axes["x"], axis=2), axes["y"], axis=1)
    se, sn = space(error), space(truth)
    if len(se) == 1:
        return se[0], sn[0], se[-1], sn[-1]
    return trapezoid(se, axes["t"]), trapezoid(sn, axes["t"]), se[-1], sn[-1]


def evaluate(task, solution, runtime_seconds=0.0, wall_seconds=0.0, peak_memory_mb=0.0):
    s, layout, components = task.spec, task.layout, task.components
    x, t, u = np.asarray(solution.x, dtype=float), np.asarray(solution.t, dtype=float), np.asarray(solution.u, dtype=float)
    y = None if solution.y is None else np.asarray(solution.y, dtype=float)
    if not np.all(np.isfinite(u)):
        raise ValueError("Non-finite solution")
    ncomp = len(components)
    steady = layout == "2d_steady"
    _check_axis(x, s["a"], s["b"], "x")
    if layout != "1d_periodic_time":
        y0, y1 = y_range(s)
        _check_axis(y, y0, y1, "y")
    if steady:
        t = np.zeros(1)
        u = u[None]
    elif len(t) < 3 or not np.all(np.diff(t) > 0) or abs(t[0]) > 1e-12 or not np.isclose(t[-1], s["T"], atol=1e-12, rtol=0):
        raise ValueError("t must run from 0 to T, strictly increasing, with at least 3 points")
    expected = (len(t), len(x)) if layout == "1d_periodic_time" else (len(t), len(y), len(x))
    if ncomp == 1 and u.shape == expected:
        u = u[..., None]
    if u.shape != expected+(ncomp,):
        raise ValueError(f"u must have shape {expected + ((ncomp,) if ncomp > 1 else ())} "
                         f"(components {list(components)} on the last axis)")
    axes = evaluation_axes(task)
    eval_t = axes.get("t", np.zeros(1))
    if layout == "1d_periodic_time":
        tt, xx = np.meshgrid(eval_t, axes["x"], indexing="ij")
        predicted = np.stack([RegularGridInterpolator((t, x), u[..., j])(np.stack([tt, xx], axis=-1))
                              for j in range(ncomp)], axis=-1)
    else:
        tt, yy, xx = np.meshgrid(eval_t, axes["y"], axes["x"], indexing="ij")
        grid = (y, x) if steady else (t, y, x)
        points = np.stack([yy[0], xx[0]], axis=-1) if steady else np.stack([tt, yy, xx], axis=-1)
        predicted = np.stack([RegularGridInterpolator(grid, (u[0] if steady else u)[..., j])(points)
                              for j in range(ncomp)], axis=-1)
        if steady:
            predicted = predicted[None]
    truth = np.array(reference(task), dtype=float)
    if "p" in components:  # pressure is determined up to a constant: compare mean-free pressures at each time
        j = components.index("p")
        for array in (predicted, truth):
            array[..., j] -= array[..., j].mean(axis=(1, 2), keepdims=True)
    error = predicted-truth
    ie, in_, fe, fn = _norms(error, truth, axes, layout)
    relative_l2 = float(np.sqrt(ie/max(in_, 1e-30)))
    final_l2 = float(np.sqrt(fe/max(fn, 1e-30)))
    # boundary conditions on the returned grid
    if layout in ("1d_periodic_time", "2d_periodic_time"):
        boundary_error = float(np.abs(u[:, ..., 0, :]-u[:, ..., -1, :]).max()) if layout == "1d_periodic_time" else \
            float(max(np.abs(u[:, :, 0]-u[:, :, -1]).max(), np.abs(u[:, 0]-u[:, -1]).max()))
    elif task.family == "navier_stokes_2d":
        yy, xx = np.meshgrid(y, x, indexing="ij")
        edge = np.zeros(yy.shape, dtype=bool)
        edge[[0, -1], :] = edge[:, [0, -1]] = True
        boundary_error = float(np.abs(u[0][edge][:, :2]-eq.boundary(s, xx[edge], yy[edge])).max())
    else:  # perfect conductor: Ex = 0 on the y-edges, Ey = 0 on the x-edges
        boundary_error = float(max(np.abs(u[:, [0, -1], :, 0]).max(), np.abs(u[:, :, [0, -1], 1]).max()))
    if steady:
        initial_error = 0.0
    elif layout == "1d_periodic_time":
        initial_error = float(np.abs(u[0]-eq.initial(s, x)).max())
    else:
        yy, xx = np.meshgrid(y, x, indexing="ij")
        initial_error = float(np.abs(u[0]-eq.initial(s, xx, yy)).max())
    accuracy = relative_l2 <= task.relative_l2_tolerance and final_l2 <= task.relative_l2_tolerance
    constraints = boundary_error <= task.boundary_tolerance and initial_error <= task.initial_tolerance
    resources = wall_seconds <= task.max_seconds and peak_memory_mb <= task.memory_mb
    return {"task_id": task.task_id, "family": task.family, "split": task.split, "status": "ok",
            "relative_l2": relative_l2, "final_relative_l2": final_l2, "absolute_l2": float(np.sqrt(ie)),
            "final_absolute_l2": float(np.sqrt(fe)), "absolute_linf": float(np.abs(error).max()),
            "boundary_error": boundary_error, "initial_error": initial_error, "physical_residual_relative": None,
            "runtime_seconds": runtime_seconds, "wall_seconds": wall_seconds, "peak_memory_mb": peak_memory_mb,
            "accuracy_pass": accuracy, "constraints_pass": constraints, "resources_pass": resources,
            "passed": accuracy and constraints and resources, "method": getattr(solution, "method", None),
            "nx": len(x), "nt": len(t), "evaluator_version": EVALUATOR_VERSION}


def score_model(case, model):
    family = model.get("family") if isinstance(model, dict) else None
    parameters = model.get("parameters") if isinstance(model, dict) else None
    parameters = {k: v for k, v in (parameters if isinstance(parameters, dict) else {}).items()
                  if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}
    correct = family == case.family
    errors = {}
    if correct:
        for name, value in true_parameters(case.truth.spec).items():
            estimate = parameters.get(name)
            errors[name] = None if estimate is None else abs(estimate-value)/max(abs(value), 1e-12)
    method = model.get("method") if isinstance(model, dict) else None
    return {"identified_family": family if isinstance(family, str) else None, "identified_parameters": parameters,
            "true_family": case.family, "family_correct": correct, "parameter_relative_errors": errors,
            "method": method if isinstance(method, str) else None}


def _jsonable(spec):
    return json.loads(json.dumps(spec, default=float))


def run_case(case, code, budget, solution_path=None, gpu=True):
    data = case.data()

    def postprocess(result, extras):
        model = None
        if "model_json" in extras:
            try:
                model = json.loads(str(extras["model_json"]))
            except ValueError:
                model = None
        return {**result, **score_model(case, model), "case_id": case.case_id}

    view = {**case.public_view(), "library": LIBRARY}
    result = run_program_task(
        case.truth, code, budget, solution_path,
        request_overrides={"mode": "pdebench_ext", "gpu": gpu, "problem": view, "hidden": _jsonable(case.truth.spec),
                           "data": {k: (None if data[k] is None else np.asarray(data[k]).tolist())
                                    for k in ("x", "y", "t", "u")}},
        postprocess=postprocess, evaluator=evaluate)
    if "case_id" not in result:
        result.update(score_model(case, None), case_id=case.case_id)
    result["problem"] = case.truth.spec["problem"]
    return result


CONTRACT = f"""Task: each case gives you measurements of an UNKNOWN equation from the library below with UNKNOWN scalar
coefficients. Identify the equation class and its coefficients from the data, choose a solution method (numerical
or PINN), and return an accurate solution over the whole domain. One program (family "{PROGRAM_KEY}") defines
solve(problem) and is used for every case.

problem attributes:
- case_id, split, layout (one of "1d_periodic_time", "2d_periodic_time", "2d_time", "2d_steady"), domain,
  components (the names of the solution fields, e.g. ["u"], ["u", "v", "p"], ["Ex", "Ey", "Hz"], ["Ex", "Ey"]),
  forcing_components, boundary_description, evaluation_grid, relative_l2_tolerance, boundary_tolerance,
  initial_tolerance, max_seconds, memory_mb, device ("cuda" or "cpu").
- data_x, data_y, data_t, data_u: n_data noisy samples; data_u has one column per component (a plain array when
  there is one component). Gaussian noise with standard deviation noise_std (one value per component). data_y is
  None for 1-D cases, data_t for steady cases.
- forcing(*coords): the source term of the TRUE equation (momentum forcing, current density J, or f), with
  arrays in coordinate order f(x, t), f(x, y, t) or f(x, y); returns one column per forcing component on the last
  axis (a plain array when there is one forcing component).
- initial(*space): the fields at t = 0 (time-dependent layouts); components on the last axis, dropped when there
  is one component.
- boundary(x, y): the Dirichlet velocity (u, v) on the boundary (Navier-Stokes with Dirichlet data only).
- elapsed_seconds(): wall seconds since the solve process started (max_seconds includes start-up).
- library: dict {{class name: {{"equation": ..., "parameters": {{name: meaning}}}}}} of candidate equations.

solve(problem) returns a dict with "model": {{"family": <class name>, "parameters": {{name: value}},
"method": "numerical" or "pinn"}} and the solution:
- 1d_periodic_time: "x" uniform over the domain including both endpoints, "t" strictly increasing from 0 to T
  (at least 3 points), "u" of shape (len(t), len(x)); u[0] = initial(x) and u[:, 0] = u[:, -1].
- 2d_periodic_time / 2d_time: "x", "y" uniform over the domain including both endpoints, "t" from 0 to T, "u" of
  shape (len(t), len(y), len(x), number of components) (the last axis may be omitted for one component);
  u[0] = initial(X, Y); periodic layouts repeat their values on opposite edges; 2d_time satisfies the stated
  boundary condition.
- 2d_steady: "x", "y" as above and "u" of shape (len(y), len(x), number of components) satisfying the stated
  boundary condition.

Scoring: the solution is linearly interpolated onto the evaluation grid and compared with the exact or a
high-accuracy reference solution: space-time and final-time relative L2 over all components together (steady:
one relative L2). Pressure is compared after removing its spatial mean. A case passes when both errors are at
most relative_l2_tolerance, the initial and boundary conditions hold to their tolerances, and the solve fits
max_seconds (wall and CPU, including start-up) and memory_mb. The reported model is scored separately and does not
affect acceptance. Feedback for training (evolution) cases includes the true class and parameters; validation
feedback does not.

Allowed imports: {", ".join(sorted(ALLOWED_MODULES))}. Forbidden: file, network, or process access; eval, exec,
compile, open, getattr, setattr, type, dir; names or attributes beginning with "_"; frame or code introspection.
Programs return at most {MAX_OUTPUT_VALUES:,} values and are at most {MAX_CODE_CHARS:,} characters. Write genuine
fitting and numerical solution code; never hard-code answers or branch on case_id.

Solver methods: numerical (finite differences/volumes/elements, spectral, ...) with NumPy/SciPy, or pinn (PyTorch
on the GPU, problem.device); they may be combined. One shared GPU and one CPU thread."""

DIAGNOSTIC_FIELDS = ("status", "relative_l2", "final_relative_l2", "initial_error", "boundary_error", "runtime_seconds",
                     "wall_seconds", "peak_memory_mb", "accuracy_pass", "constraints_pass", "resources_pass", "passed",
                     "message", "identified_family", "identified_parameters", "method")


class PDEBenchExtProtocol(CodeProtocol):
    name = "pdebench-ext"
    gpu = True

    def __init__(self, noise_fraction=NOISE_FRACTION, modular=False):
        self.cases, self.noise_fraction = {}, noise_fraction
        self.modular = modular
        if modular:
            from .modular import INITIAL_WORKFLOW
            self.initial_workflow = INITIAL_WORKFLOW

    def classes(self):
        return list(LIBRARY)

    def code(self, version):
        if self.modular:
            from .modular import assemble
            return assemble(version.program)
        return version.program[PROGRAM_KEY]

    split_names = ("evolution", "development")  # ("bench-train", "bench-val") for RSI-PDE Bench runs

    def load_tasks(self, root, package, families):
        tasks = {split: make_cases(source, self.noise_fraction)
                 for split, source in zip(("evolution", "development"), self.split_names)}
        self.cases = {case.task_id: case for items in tasks.values() for case in items}
        return tasks

    def starting_program(self, start, families):
        if start != "scratch":
            raise ValueError("The pdebench-ext protocol starts from scratch only")
        if self.modular:
            from .modular import starting_program
            return starting_program()
        return {PROGRAM_KEY: STUB_PROGRAM}

    def self_test_tasks(self, evolution):
        chosen, seen = [], set()
        for case in evolution:
            if case.truth.layout not in seen:
                seen.add(case.truth.layout)
                chosen.append(case)
        return chosen[:SELF_TEST_KINDS]

    def problem_records(self, tasks):
        return [case.public_view() for case in tasks]

    def contract(self, families):
        from .modular import CONTRACT_ADDENDUM
        return CONTRACT+(CONTRACT_ADDENDUM if self.modular else "")+"\nlibrary = "+json.dumps(LIBRARY)

    def evaluate(self, tasks, version, budget, solution_directory=None):
        results = []
        if solution_directory is not None:
            solution_directory.mkdir(parents=True, exist_ok=True)
        for index, case in enumerate(tasks):
            self.cases.setdefault(case.task_id, case)
            try:
                path = solution_directory/f"{case.case_id}.npz" if solution_directory is not None else None
                results.append(run_case(case, self.code(version), budget, path, self.gpu))
            except BudgetExceeded:
                for rest in tasks[index:]:
                    row = failed_result(rest.truth, "budget_exhausted", "Case not executed; allowance exhausted", 0)
                    results.append({**row, **score_model(rest, None), "case_id": rest.case_id,
                                    "problem": rest.truth.spec["problem"]})
                break
        for row in results:
            row["evaluator_version"] = EVALUATOR_VERSION
        return results

    def diagnostics(self, results):
        rows = []
        for result in results:
            row = {"case_id": result["case_id"], "split": result["split"],
                   **{key: result.get(key) for key in DIAGNOSTIC_FIELDS}}
            if result["split"] == "evolution":
                case = self.cases[result["task_id"]]
                row.update(true_family=result.get("true_family"), true_parameters=true_parameters(case.truth.spec),
                           family_correct=result.get("family_correct"))
            rows.append(row)
        return rows

    def reasons(self, reasons, tasks):
        names = {case.task_id: case.case_id for case in self.cases.values()}
        cleaned = []
        for reason in reasons:
            head, _, subject = reason.partition(": ")
            cleaned.append(f"{head}: {names[subject]}" if subject in names else
                           f"{head}: a hidden equation class" if subject else reason)
        return cleaned

    def metadata(self, tasks):
        from .codegen import gpu_sandbox_self_test
        return {"protocol_version": VERSION, "evaluator_version": EVALUATOR_VERSION, "library": LIBRARY,
                "harness": "v2-modular" if self.modular else "v1", "splits": list(self.split_names), "n_points": N_POINTS,
                "noise_fraction": self.noise_fraction, "tolerance": TOLERANCE,
                "cases": {case.case_id: case.task_id for items in tasks.values() for case in items},
                "gpu_sandbox": gpu_sandbox_self_test()}
