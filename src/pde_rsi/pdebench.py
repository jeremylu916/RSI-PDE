"""pde_benchmarks.tex suite under the student protocol: identify the equation from data, then solve it.

Every case is one case of pde_benchmarks.tex (Problems 1-11) with its own numbers. The agent sees sparse
noisy samples of the exact solution, black-box forcing / initial / boundary data, the domain, and the
library of equation classes; it chooses the class, its coefficients, and a method (numerical or PINN),
and writes the solver from scratch. Manufactured cases use the tex profiles by split: training (sine) ->
evolution, validation (polynomial) -> development, reserved (mixed sine) -> sealed test. Literature cases
are assigned to splits with different numbers. The sealed test is evaluated only after the agents are
frozen (scripts/pdebench_sealed_test.py) and never enters the improvement loop.
"""

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import RegularGridInterpolator

from . import equations_bench as eq
from .code_agents import CodeProtocol
from .codegen import ALLOWED_MODULES, MAX_CODE_CHARS, MAX_OUTPUT_VALUES, STUB_PROGRAM, run_program_task
from .evaluator import failed_result
from .runner import BudgetExceeded
from .storage import project_path

PROTOCOL_VERSION = "pdebench-v1"
EVALUATOR_VERSION = "pdebench-evaluator-v1"
SPLITS = ("evolution", "development", "test")
# test2: a fresh sealed split created on 2026-10-18 for the harness-vs-coding-agent study, after "test" had
# been inspected. Every tex case appears once with numbers not used in any other split. Evaluator-only.
# test3: a fresh sealed split created on 2026-10-20 to evaluate harness v2, after test and test2 had been used to
# diagnose the harness. Same layout as test2 (every tex case once), new numbers. Evaluator-only.
SEALED_SPLITS = ("test", "test2", "test3")
N_POINTS = 256
NOISE_FRACTION = 0.01
MAX_SECONDS = 90.0
MEMORY_MB = 16384
SELF_TEST_CASES = 4
RELATIVE_L2_TOLERANCE = 2e-3
CONSTRAINT_TOLERANCE = 1e-10
GRID = {"space": 512, "time": 500, "wave_space": 128, "wave_time": 100}
PROGRAM_KEY = "identify"

LIBRARY = {
    "poisson": {"equation": "-D*u_xx = f(x)  (steady)", "parameters": {"D": "diffusivity > 0"}},
    "heat": {"equation": "u_t - D*u_xx = f(x,t)", "parameters": {"D": "diffusivity > 0"}},
    "convection_diffusion": {
        "equation": "u_t + v*u_x - D*u_xx = f(x,t); steady form v*u_x - D*u_xx = f(x)",
        "parameters": {"D": "diffusivity > 0", "v": "velocity (real)"}},
    "burgers": {"equation": "u_t + u*u_x - D*u_xx = f(x,t)", "parameters": {"D": "viscosity > 0"}},
    "wave2d": {"equation": "u_tt - D*(u_xx + u_yy) = f(x,y,t)", "parameters": {"D": "squared wave speed > 0"}},
    "fode": {"equation": "Caputo(alpha, u)(t) = f(t)", "parameters": {"alpha": "order in (0,1)"}},
    "caputo": {"equation": "Caputo(alpha, u) - D*u_xx = f(x,t)",
               "parameters": {"D": "diffusivity > 0", "alpha": "order in (0,1)"}},
    "multi_term": {"equation": "w1*Caputo(alpha1, u) + w2*Caputo(alpha2, u) - D*u_xx = f(x,t)",
                   "parameters": {"D": "diffusivity > 0", "alpha1": "order in (0,1)", "alpha2": "order in (0,1)",
                                  "w1": "weight > 0", "w2": "weight > 0"}},
    "variable_order_t": {"equation": "Caputo(alpha(t), u) - D*u_xx = f(x,t), alpha(t) = c*sin(t)**2",
                         "parameters": {"D": "diffusivity > 0", "c": "coefficient > 0 with alpha < 1 on the domain"}},
    "variable_order_xt": {"equation": "Caputo(alpha(x,t), u) - D*u_xx = f(x,t), alpha(x,t) = c*t**2*sin(x)",
                          "parameters": {"D": "diffusivity > 0", "c": "coefficient > 0 with alpha < 1 on the domain"}},
}
CLASS = {"1a": "poisson", "1b": "poisson", "1c": "poisson", "2a": "heat", "2b": "heat", "2c": "heat",
         "3a": "convection_diffusion", "3b": "convection_diffusion", "3c": "convection_diffusion",
         "3d": "convection_diffusion", "4a": "burgers", "4b": "burgers", "4c": "burgers", "5": "wave2d",
         "6": "fode", "7": "caputo", "8": "multi_term", "9": "caputo", "10": "variable_order_t",
         "11": "variable_order_xt"}
STEADY = ("1a", "1b", "1c", "3b", "3c")
MANUFACTURED = ("1a", "2a", "3a", "4a", "5", "7", "8")
PROFILE = {"evolution": "sine", "development": "polynomial", "test": "mixed_sine", "test2": "mixed_sine",
           "test3": "mixed_sine"}
TITLES = {"1a": "Poisson, manufactured", "1b": "Poisson, multiscale (Wang-Wang-Perdikaris)",
          "1c": "Poisson, interior layer (Mitchell)", "2a": "Heat, manufactured", "2b": "Heat, free decay",
          "2c": "Heat, time-periodic source", "3a": "Convection-diffusion, manufactured",
          "3b": "Convection-diffusion, steady boundary layer (Patankar)",
          "3c": "Convection-diffusion, steady layer with constant source", "3d": "Convection-diffusion, advected Gaussian",
          "4a": "Burgers, manufactured", "4b": "Burgers, sine data (Basdevant; Raissi et al.)",
          "4c": "Burgers, viscous shock (Whitham)", "5": "2-D wave, manufactured", "6": "Fractional ODE",
          "7": "Time-fractional diffusion, manufactured", "8": "Multi-term time-fractional diffusion",
          "9": "Time-fractional diffusion, zero initial data", "10": "Variable order alpha(t)",
          "11": "Variable order alpha(x,t)"}

# Literature cases with their numbers for each split (manufactured cases draw theirs below).
FIXED = {
    "evolution": [
        ("1c", dict(a=0.0, b=1.0, D=1.0, sharpness=20.0, x0=0.5)),
        ("2b", dict(a=0.0, b=2.0, T=1.0, D=0.3)),
        ("3b", dict(a=0.0, b=1.0, v=1.0, D=0.05)),                    # Pe = 20
        ("3d", dict(a=0.0, b=4.0, T=1.0, v=1.0, D=0.02, x0=1.0, sigma0=0.2)),
        ("4c", dict(a=-1.0, b=1.0, T=1.0, ul=1.0, ur=0.0, D=0.02)),
        ("6", dict(T=1.0, alpha=0.4, p=2.0)),
        ("9", dict(a=0.0, b=math.pi, T=1.0, D=1.0, alpha=0.5, p=2.0)),
        ("10", dict(a=0.0, b=math.pi, T=1.0, D=1.0, c=0.8, p=2.0)),
        ("11", dict(a=0.0, b=math.pi, T=1.0, D=1.0, c=0.5, p=2.0)),
    ],
    "development": [
        ("1b", dict(a=0.0, b=1.0, D=1.0)),
        ("2c", dict(a=0.0, b=1.0, T=2.0, D=0.2, gamma=2*math.pi)),
        ("3c", dict(a=0.0, b=1.0, v=1.0, D=0.02)),                    # Pe = 50
        ("3d", dict(a=0.0, b=4.0, T=1.0, v=-0.8, D=0.015, x0=2.5, sigma0=0.15)),
        ("4b", dict(a=-1.0, b=1.0, T=1.0, D=0.01/math.pi)),
        ("6", dict(T=1.0, alpha=0.7, p=0.8)),
        ("9", dict(a=0.0, b=math.pi, T=1.0, D=1.0, alpha=0.8, p=1.5)),
        ("10", dict(a=0.0, b=math.pi, T=1.0, D=1.0, c=1.3, p=2.5)),
        ("11", dict(a=0.0, b=math.pi, T=1.0, D=1.0, c=0.9, p=3.0)),
    ],
    "test2": [
        ("1b", dict(a=0.0, b=1.0, D=0.5)),
        ("1c", dict(a=0.0, b=1.0, D=0.8, sharpness=200.0, x0=0.4)),
        ("2b", dict(a=0.0, b=1.2, T=1.0, D=0.25)),
        ("2c", dict(a=0.0, b=1.0, T=1.5, D=0.3, gamma=4.0)),
        ("3b", dict(a=0.0, b=1.0, v=-1.0, D=0.02)),                   # Pe = -50: layer at x = 0
        ("3c", dict(a=0.0, b=1.0, v=1.0, D=1.0/30.0)),                # Pe = 30
        ("3d", dict(a=0.0, b=3.0, T=1.0, v=0.6, D=0.01, x0=1.2, sigma0=0.18)),
        ("4b", dict(a=-1.0, b=1.0, T=1.0, D=0.02/math.pi)),
        ("4c", dict(a=-1.0, b=1.5, T=1.0, ul=0.8, ur=-0.2, D=0.01)),
        ("6", dict(T=1.0, alpha=0.55, p=1.7)),
        ("9", dict(a=0.0, b=math.pi, T=1.0, D=1.0, alpha=0.65, p=1.0)),
        ("10", dict(a=0.0, b=math.pi, T=1.0, D=1.0, c=1.2, p=2.0)),
        ("11", dict(a=0.0, b=math.pi, T=1.0, D=1.0, c=0.6, p=2.5)),
    ],
    "test3": [
        ("1b", dict(a=0.0, b=1.0, D=0.7)),
        ("1c", dict(a=0.0, b=1.0, D=1.2, sharpness=100.0, x0=0.6)),
        ("2b", dict(a=0.0, b=1.8, T=1.0, D=0.15)),
        ("2c", dict(a=0.0, b=1.0, T=1.2, D=0.12, gamma=6.0)),
        ("3b", dict(a=0.0, b=1.0, v=1.0, D=0.03)),                    # Pe = 33
        ("3c", dict(a=0.0, b=1.0, v=1.0, D=0.04)),                    # Pe = 25
        ("3d", dict(a=0.0, b=3.5, T=1.0, v=-0.7, D=0.012, x0=2.3, sigma0=0.16)),
        ("4b", dict(a=-1.0, b=1.0, T=1.0, D=0.015/math.pi)),
        ("4c", dict(a=-1.0, b=1.2, T=1.0, ul=1.2, ur=0.2, D=0.008)),
        ("6", dict(T=1.0, alpha=0.35, p=2.3)),
        ("9", dict(a=0.0, b=math.pi, T=1.0, D=1.0, alpha=0.45, p=1.3)),
        ("10", dict(a=0.0, b=math.pi, T=1.0, D=1.0, c=1.1, p=1.8)),
        ("11", dict(a=0.0, b=math.pi, T=1.0, D=1.0, c=0.8, p=2.0)),
    ],
    "test": [
        ("1c", dict(a=0.0, b=1.0, D=1.0, sharpness=1000.0, x0=0.5)),
        ("2b", dict(a=0.0, b=1.5, T=1.0, D=0.1)),
        ("2c", dict(a=0.0, b=1.0, T=2.0, D=0.05, gamma=10.0)),
        ("3b", dict(a=0.0, b=1.0, v=1.0, D=0.005)),                   # Pe = 200
        ("3c", dict(a=0.0, b=1.0, v=1.0, D=0.01)),                    # Pe = 100
        ("4c", dict(a=-1.0, b=2.0, T=1.0, ul=1.5, ur=0.5, D=0.005)),
        ("6", dict(T=1.0, alpha=0.9, p=1.2)),
        ("9", dict(a=0.0, b=math.pi, T=1.0, D=1.0, alpha=0.3, p=0.6)),
        ("10", dict(a=0.0, b=math.pi, T=1.0, D=1.0, c=1.0, p=1.5)),
        ("11", dict(a=0.0, b=math.pi, T=1.0, D=1.0, c=0.7, p=1.5)),
    ],
}


def manufactured(problem, split):
    """Domain and coefficients of a manufactured case, drawn for this case alone."""
    rng = np.random.default_rng(int(hashlib.sha256(f"{PROTOCOL_VERSION}:{problem}:{split}".encode())
                                    .hexdigest()[:12], 16))
    L, T = float(rng.uniform(1.0, 3.0)), float(rng.uniform(0.5, 1.0))
    A, D = float(rng.uniform(0.5, 1.5)), float(np.exp(rng.uniform(np.log(0.05), np.log(1.0))))
    spec = dict(a=0.0, b=L, T=T, A=A, D=D, beta=float(rng.uniform(1.0, 2.0)))
    if problem == "1a":
        del spec["T"], spec["beta"]
    if problem == "3a":
        spec["v"] = float(rng.choice([-1.0, 1.0])*rng.uniform(0.3, 1.5))
    if problem == "4a":
        spec.update(A=float(rng.uniform(0.3, 1.0)), D=float(rng.uniform(0.05, 0.5)))
    if problem == "5":
        spec.update(b=float(rng.uniform(1.0, 2.0)), D=float(rng.uniform(0.1, 1.0)), beta=float(rng.uniform(2.0, 3.0)))
    if problem == "7":
        spec["alpha"] = float(rng.uniform(0.3, 0.9))
    if problem == "8":
        spec.update(alpha1=float(rng.uniform(0.2, 0.5)), alpha2=float(rng.uniform(0.6, 0.9)),
                    w1=float(rng.uniform(0.3, 1.5)), w2=float(rng.uniform(0.3, 1.5)))
    return spec


@dataclass(frozen=True)
class BenchTask:
    """The hidden true case; the harness reads task_id, family, split, and the limits."""
    task_id: str
    family: str
    split: str
    spec: dict = field(hash=False, compare=False)
    max_seconds: float = MAX_SECONDS
    memory_mb: int = MEMORY_MB
    relative_l2_tolerance: float = RELATIVE_L2_TOLERANCE
    boundary_tolerance: float = CONSTRAINT_TOLERANCE
    initial_tolerance: float = CONSTRAINT_TOLERANCE
    as_evolution: bool = False  # hidden-time-dependence variant: steady cases are posed on (x, t) with u(x, 0)

    @property
    def problem(self):
        return self.spec["problem"]

    @property
    def dimension(self):
        return 0 if self.problem == "6" else 2 if self.problem == "5" else 1

    @property
    def steady(self):
        """Whether the case is posed (and scored) without time; the truth may be steady either way."""
        return self.problem in STEADY and not self.as_evolution

    def to_dict(self):
        return {"task_id": self.task_id, "family": self.family, "split": self.split, "spec": self.spec,
                "max_seconds": self.max_seconds, "memory_mb": self.memory_mb,
                "relative_l2_tolerance": self.relative_l2_tolerance}


def make_spec(problem, split, values):
    spec = {"problem": problem, "family": CLASS[problem], "profile": PROFILE[split] if problem in MANUFACTURED else None,
            "a": 0.0, "b": None, "T": None, "A": None, "D": None, "beta": None, "v": None, "alpha": None,
            "alpha1": None, "alpha2": None, "w1": None, "w2": None, "c": None, "p": None, "gamma": None,
            "x0": None, "sharpness": None, "sigma0": None, "ul": None, "ur": None}
    spec.update(values)
    if problem == "6":
        spec.update(a=None, b=None)
    return spec


def true_parameters(task):
    s, problem = task.spec, task.problem
    if problem == "3b":  # f = 0 and steady: only the ratio v/D is identifiable
        return {"v/D": s["v"]/s["D"]}
    names = {"poisson": ("D",), "heat": ("D",), "convection_diffusion": ("D", "v"), "burgers": ("D",),
             "wave2d": ("D",), "fode": ("alpha",), "caputo": ("D", "alpha"),
             "multi_term": ("D", "alpha1", "alpha2", "w1", "w2"), "variable_order_t": ("D", "c"),
             "variable_order_xt": ("D", "c")}[task.family]
    return {name: s[name] for name in names}


_DATA = {}


@dataclass(frozen=True)
class BenchCase:
    case_id: str
    truth: BenchTask
    noise_fraction: float = NOISE_FRACTION  # noise std as a fraction of the clean samples' RMS

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

    @property
    def seed(self):
        return int(hashlib.sha256(f"{PROTOCOL_VERSION}:{self.task_id}".encode()).hexdigest()[:8], 16)

    def coordinates(self):
        return {0: ("t",), 1: ("x",) if self.truth.steady else ("x", "t"), 2: ("x", "y", "t")}[self.truth.dimension]

    def data(self):
        """Deterministic noisy samples of the exact solution at scattered interior points (cached)."""
        key = (self.task_id, self.noise_fraction, self.truth.steady)
        if key not in _DATA:
            _DATA[key] = self._sample()
        return _DATA[key]

    def _sample(self):
        s = self.truth.spec
        rng = np.random.default_rng(self.seed)
        coordinates = self.coordinates()
        x = rng.uniform(s["a"], s["b"], N_POINTS) if "x" in coordinates else None
        y = rng.uniform(s["a"], s["b"], N_POINTS) if "y" in coordinates else None
        t = rng.uniform(0, s["T"], N_POINTS) if "t" in coordinates else None
        clean = eq.exact(s, np.zeros(N_POINTS) if x is None else x, np.zeros(N_POINTS) if t is None else t, y)
        noise_std = self.noise_fraction*float(np.sqrt(np.mean(clean**2)))
        return {"x": x, "y": y, "t": t, "u": clean+noise_std*rng.standard_normal(N_POINTS), "noise_std": noise_std}

    def public_view(self):
        """Everything the model and the program may know: no class, coefficients, or source problem."""
        s, task = self.truth.spec, self.truth
        domain = {}
        if task.dimension >= 1:
            domain["x"] = [s["a"], s["b"]]
        if task.dimension == 2:
            domain["y"] = [s["a"], s["b"]]
        if not task.steady:
            domain["t"] = [0.0, s["T"]]
        space, time = output_grid(task)
        layout = {} if task.as_evolution else {"coordinates": list(self.coordinates()),
                                               "time_dependent": not task.steady}
        return {"case_id": self.case_id, "split": self.split, **layout,
                "domain": domain, "spatial_dimension": task.dimension,
                "boundary_description": ("none (no spatial variable)" if task.dimension == 0 else
                             "u = 0 on all four edges" if task.dimension == 2 else
                             "Dirichlet: u(x_left, t), u(x_right, t) = boundary(t)"),
                "n_data": N_POINTS, "noise_std": self.data()["noise_std"],
                "relative_l2_tolerance": task.relative_l2_tolerance, "boundary_tolerance": task.boundary_tolerance,
                "initial_tolerance": task.initial_tolerance, "max_seconds": task.max_seconds,
                "memory_mb": task.memory_mb, "evaluation_space_points": space, "evaluation_time_points": time}


def make_cases(split, noise_fraction=NOISE_FRACTION, hide_time_dependence=False):
    """All cases of one split, ordered by a hash so IDs reveal neither the class nor the problem.
    The sample points and the standardized noise draw do not depend on noise_fraction, only its scale.
    hide_time_dependence poses steady cases as evolution problems on [0, T] (T drawn per case) whose exact
    solution does not change in time, so that time dependence must be inferred."""
    if split.startswith("bench-"):  # RSI-PDE Bench splits (src/pde_rsi/bench/splits.py)
        from .bench.splits import bench_cases
        return bench_cases("tex", split, noise_fraction, hide_time_dependence=hide_time_dependence)
    if split not in SPLITS+SEALED_SPLITS:
        raise ValueError(f"split must be one of {SPLITS+SEALED_SPLITS}")
    entries = [(problem, manufactured(problem, split)) for problem in MANUFACTURED] + FIXED[split]
    counts, tasks = {}, []
    for problem, values in entries:
        counts[problem] = counts.get(problem, 0)+1
        task_id = f"pb-{problem}-{split}-{counts[problem]}"
        spec = make_spec(problem, split, values)
        if hide_time_dependence and problem in STEADY:
            rng = np.random.default_rng(int(hashlib.sha256(f"{PROTOCOL_VERSION}:T:{task_id}".encode()).hexdigest()[:12], 16))
            spec["T"] = float(rng.uniform(0.5, 1.0))
        tasks.append(BenchTask(task_id, CLASS[problem], split, spec, as_evolution=hide_time_dependence))
    tasks.sort(key=lambda t: hashlib.sha256(f"{PROTOCOL_VERSION}:{t.task_id}".encode()).hexdigest())
    prefix = {"evolution": "t", "development": "v", "test": "r", "test2": "s", "test3": "u"}[split]
    return [BenchCase(f"case-{prefix}{index:02d}", task, noise_fraction) for index, task in enumerate(tasks)]


# ---------------------------------------------------------------- evaluator
def output_grid(task):
    """(space nodes per axis, time nodes) of the evaluation grid."""
    if task.dimension == 2:
        return GRID["wave_space"], GRID["wave_time"]
    return (1 if task.dimension == 0 else GRID["space"]), (1 if task.steady else GRID["time"])


def evaluation_nodes(task):
    s = task.spec
    space, time = output_grid(task)
    x = np.zeros(1) if task.dimension == 0 else np.linspace(s["a"], s["b"], space)
    t = np.zeros(1) if task.steady else np.linspace(0.0, s["T"], time)
    return x, t


_REFERENCE = {}


def reference(task):
    """Exact solution on the evaluation grid: u[t, x] or u[t, y, x] (cached; Case 4b uses quadrature)."""
    key = (task.task_id, task.steady)
    if key not in _REFERENCE:
        x, t = evaluation_nodes(task)
        if task.dimension == 2:
            tt, yy, xx = np.meshgrid(t, x, x, indexing="ij")
            _REFERENCE[key] = eq.exact(task.spec, xx, tt, yy)
        else:
            tt, xx = np.meshgrid(t, x, indexing="ij")
            _REFERENCE[key] = eq.exact(task.spec, xx, tt)
    return _REFERENCE[key]


def _uniform(axis, low, high, name):
    if axis.ndim != 1 or len(axis) < 3 or not np.all(np.diff(axis) > 0):
        raise ValueError(f"{name} must be strictly increasing with at least 3 points")
    if not np.allclose(np.diff(axis), axis[1]-axis[0], rtol=1e-9, atol=1e-13):
        raise ValueError(f"Expected a uniform {name} grid")
    if not (np.isclose(axis[0], low, atol=1e-12, rtol=0) and np.isclose(axis[-1], high, atol=1e-12, rtol=0)):
        raise ValueError(f"{name} grid does not cover the domain")


def evaluate(task, solution, runtime_seconds=0.0, wall_seconds=0.0, peak_memory_mb=0.0):
    """Relative space-time and final-time L2 errors against the exact solution, plus IC/BC checks."""
    s = task.spec
    x, t, u = map(np.asarray, (solution.x, solution.t, solution.u))
    y = None if solution.y is None else np.asarray(solution.y)
    if not all(np.isfinite(a).all() for a in (x, t, u)+(() if y is None else (y,))) or not np.isrealobj(u):
        raise ValueError("Non-finite or complex solution")
    if task.dimension == 0:
        if x.shape != (1,) or u.shape != (len(t), 1):
            raise ValueError("FODE solutions use x = [0] and u of shape (len(t), 1)")
    elif task.dimension == 2:
        if y is None or u.shape != (len(t), len(y), len(x)):
            raise ValueError("2-D solutions need x, y and u[t, y, x]")
        _uniform(x, s["a"], s["b"], "x")
        _uniform(y, s["a"], s["b"], "y")
    else:
        if u.shape != (len(t), len(x)):
            raise ValueError("u must have shape (len(t), len(x))")
        _uniform(x, s["a"], s["b"], "x")
    if task.steady:
        if t.shape != (1,) or abs(t[0]) > 1e-12:
            raise ValueError("Steady solutions use t = [0]")
    elif (len(t) < 3 or not np.all(np.diff(t) > 0) or abs(t[0]) > 1e-12
          or not np.isclose(t[-1], s["T"], atol=1e-12, rtol=0)):
        raise ValueError("Solution does not cover the time horizon [0, T]")
    eval_x, eval_t = evaluation_nodes(task)
    if task.dimension == 0:
        predicted = np.interp(eval_t, t, u[:, 0])[:, None]
    elif task.steady:
        predicted = np.interp(eval_x, x, u[0])[None, :]
    elif task.dimension == 1:
        tt, xx = np.meshgrid(eval_t, eval_x, indexing="ij")
        predicted = RegularGridInterpolator((t, x), u)(np.stack([tt, xx], axis=-1))
    else:
        tt, yy, xx = np.meshgrid(eval_t, eval_x, eval_x, indexing="ij")
        predicted = RegularGridInterpolator((t, y, x), u)(np.stack([tt, yy, xx], axis=-1))
    truth = reference(task)
    error = predicted-truth
    if task.dimension == 0:
        space_error, space_norm = error[:, 0]**2, truth[:, 0]**2
    elif task.dimension == 1:
        space_error, space_norm = trapezoid(error**2, eval_x, axis=1), trapezoid(truth**2, eval_x, axis=1)
    else:
        space_error = trapezoid(trapezoid(error**2, eval_x, axis=2), eval_x, axis=1)
        space_norm = trapezoid(trapezoid(truth**2, eval_x, axis=2), eval_x, axis=1)
    integrated_error = space_error[0] if len(eval_t) == 1 else trapezoid(space_error, eval_t)
    integrated_norm = space_norm[0] if len(eval_t) == 1 else trapezoid(space_norm, eval_t)
    relative_l2 = float(np.sqrt(integrated_error/max(integrated_norm, 1e-30)))
    final_l2 = float(np.sqrt(space_error[-1]/max(space_norm[-1], 1e-30)))
    if task.dimension == 0:
        boundary_error = 0.0
    elif task.dimension == 2:
        boundary_error = float(max(np.abs(u[:, [0, -1], :]).max(), np.abs(u[:, :, [0, -1]]).max()))
    else:
        values = np.array([eq.boundary(s, float(time)) for time in t])
        boundary_error = float(np.max(np.abs(u[:, [0, -1]]-values)))
    if task.steady:
        initial_error = 0.0
    elif task.dimension == 2:
        yy, xx = np.meshgrid(y, x, indexing="ij")
        initial_error = float(np.max(np.abs(u[0]-eq.initial(s, xx, yy))))
    else:
        initial_error = float(np.max(np.abs(u[0]-eq.initial(s, x))))
    accuracy = relative_l2 <= task.relative_l2_tolerance and final_l2 <= task.relative_l2_tolerance
    constraints = boundary_error <= task.boundary_tolerance and initial_error <= task.initial_tolerance
    resources = wall_seconds <= task.max_seconds and peak_memory_mb <= task.memory_mb
    return {"task_id": task.task_id, "family": task.family, "split": task.split, "status": "ok",
            "relative_l2": relative_l2, "final_relative_l2": final_l2,
            "absolute_l2": float(np.sqrt(integrated_error)), "final_absolute_l2": float(np.sqrt(space_error[-1])),
            "absolute_linf": float(np.max(np.abs(error))), "boundary_error": boundary_error,
            "initial_error": initial_error, "physical_residual_relative": None,
            "runtime_seconds": runtime_seconds, "wall_seconds": wall_seconds, "peak_memory_mb": peak_memory_mb,
            "accuracy_pass": accuracy, "constraints_pass": constraints, "resources_pass": resources,
            "passed": accuracy and constraints and resources, "method": solution.method,
            "nx": len(x)-1, "nt": len(t)-1, "evaluator_version": EVALUATOR_VERSION}


def score_model(case, model):
    """Class accuracy and relative parameter errors of a returned model (reported, never gated)."""
    family = model.get("family") if isinstance(model, dict) else None
    parameters = model.get("parameters") if isinstance(model, dict) else None
    parameters = {k: v for k, v in (parameters if isinstance(parameters, dict) else {}).items()
                  if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}
    correct = family == case.family
    errors = {}
    if correct:
        estimates = dict(parameters)
        if "v" in estimates and estimates.get("D"):
            estimates["v/D"] = estimates["v"]/estimates["D"]
        for name, value in true_parameters(case.truth).items():
            estimate = estimates.get(name)
            errors[name] = None if estimate is None else abs(estimate-value)/max(abs(value), 1e-12)
    method = model.get("method") if isinstance(model, dict) else None
    return {"identified_family": family if isinstance(family, str) else None, "identified_parameters": parameters,
            "true_family": case.family, "family_correct": correct, "parameter_relative_errors": errors,
            "method": method if isinstance(method, str) else None}


def data_digest(cases):
    return {case.case_id: hashlib.sha256(json.dumps({k: (v.tolist() if hasattr(v, "tolist") else v)
                                                      for k, v in case.data().items()}).encode()).hexdigest()
            for case in cases}


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
        request_overrides={"mode": "pdebench", "gpu": gpu, "problem": view, "hidden": case.truth.spec,
                           # the sandbox needs the call layout even when the view hides it from the program
                           "layout": {"coordinates": list(case.coordinates()), "time_dependent": not case.truth.steady},
                           "data": {k: (None if data[k] is None else data[k].tolist()) for k in ("x", "y", "t", "u")}},
        postprocess=postprocess, evaluator=evaluate)
    if "case_id" not in result:  # failed solves carry no model
        result.update(score_model(case, None), case_id=case.case_id)
    result["problem"] = case.truth.problem
    return result


# ---------------------------------------------------------------- protocol for the improvement loop
CONTRACT = f"""Task: like a student, each case gives you measurements of an UNKNOWN equation from the library below with
UNKNOWN coefficients. Identify the equation class and its coefficients from the data, choose a solution method
(numerical or PINN), and return an accurate solution of that equation over the whole domain. One program
(family "{PROGRAM_KEY}") defines solve(problem) and is used for every case.

problem attributes:
- case_id, split, coordinates (e.g. ["x", "t"], ["x"] for a steady case, ["t"] for an ODE, ["x", "y", "t"]),
  domain {{"x": [x_left, x_right], "y": [...], "t": [0, T]}} (only the coordinates present),
  spatial_dimension (0, 1 or 2), time_dependent (bool), boundary_description.
- data_x, data_y, data_t, data_u: n_data noisy samples u + noise at scattered interior points (data_y is None
  unless spatial_dimension == 2; data_t is None for steady cases; data_x is None for ODE cases). The noise is
  Gaussian with standard deviation noise_std (also given).
- forcing(*coords): the source term f of the TRUE equation, called with arrays in the order of coordinates
  (f(x, t), f(x), f(t), or f(x, y, t)); t may be 0 or larger.
- initial(*space): u at t = 0 (initial(x), initial(x, y), or initial() for an ODE); time-dependent cases only.
- initial_velocity(x, y): u_t at t = 0 for second-order-in-time cases with spatial_dimension 2.
- boundary(t): array [u(x_left, t), u(x_right, t)] of Dirichlet values for spatial_dimension 1 (call
  boundary(0.0) for steady cases). The values may be nonzero. Two-dimensional cases have u = 0 on the boundary.
- library: dict {{class name: {{"equation": ..., "parameters": {{name: meaning}}}}}} of candidate equations.
- relative_l2_tolerance, boundary_tolerance, initial_tolerance, max_seconds, memory_mb,
  evaluation_space_points, evaluation_time_points, device ("cuda" or "cpu").
Caputo(alpha, u)(t) = 1/Gamma(1-alpha) * integral_0^t (t-s)^(-alpha) u_s(s) ds (history from t = 0). A variable
order alpha(t) or alpha(x, t) is evaluated at the current point and held fixed inside the integral.

solve(problem) returns a dict with "model": {{"family": <library class name>, "parameters": {{name: value}},
"method": "numerical" or "pinn"}} and the solution arrays:
- spatial_dimension 1, time-dependent: "x" uniform from x_left to x_right, "t" strictly increasing from 0 to T
  (at least 3 points), "u" of shape (len(t), len(x)) with u[0] = initial(x) and u[:, 0], u[:, -1] equal to
  boundary(t) within boundary_tolerance;
- steady: "x" uniform over the domain and "u" of shape (len(x),) with the boundary values at both ends;
- ODE: "t" strictly increasing from 0 to T and "u" of shape (len(t),) with u[0] = initial();
- spatial_dimension 2: "x", "y" uniform over the domain, "t" as above, "u" of shape (len(t), len(y), len(x))
  with u[0] = initial(x, y) and u = 0 on the four edges.

Scoring: u is compared with the TRUE solution after linear interpolation onto evaluation_space_points uniform
nodes per spatial axis and evaluation_time_points uniform t nodes (trapezoidal space-time and final-time
relative L2). A case passes when both are at most relative_l2_tolerance, initial and boundary values hold, and
the solve fits max_seconds (wall and CPU, including start-up) and memory_mb. The gate uses only these errors.
The reported model is scored separately (class accuracy and relative parameter errors) and does not affect
acceptance. Feedback for training (evolution) cases includes the true class and parameters; validation
feedback does not.

Allowed imports: {", ".join(sorted(ALLOWED_MODULES))}. Forbidden: file, network, or process access; eval, exec,
compile, open, getattr, setattr, type, dir; names or attributes beginning with "_"; frame or code
introspection. Programs return at most {MAX_OUTPUT_VALUES:,} values and are at most {MAX_CODE_CHARS:,}
characters. Write genuine fitting and numerical solution code; never hard-code answers or branch on case_id.

Solver methods. Choose for each case how to solve the equation you identified:
- numerical: finite differences, finite volumes, spectral methods, or L1/L2-1sigma-type schemes for Caputo
  terms, written with NumPy/SciPy;
- pinn: a physics-informed neural network trained with PyTorch on the GPU (problem.device) to minimize the
  equation residual at collocation points plus data misfit and initial/boundary penalties. Evaluate the trained
  network on a grid and return arrays.
You may also combine them. Allowed PyTorch modules: torch, torch.nn, torch.nn.functional, torch.optim,
torch.autograd (no hub, jit, compile, saving or loading). One shared GPU and one CPU thread; every solve,
including PyTorch import (about 3 s) and training, must fit max_seconds of wall and CPU time."""
HIDDEN_TIME_CONTRACT = CONTRACT.replace(
    """- case_id, split, coordinates (e.g. ["x", "t"], ["x"] for a steady case, ["t"] for an ODE, ["x", "y", "t"]),
  domain {"x": [x_left, x_right], "y": [...], "t": [0, T]} (only the coordinates present),
  spatial_dimension (0, 1 or 2), time_dependent (bool), boundary_description.""",
    """- case_id, split, domain {"x": [x_left, x_right], "y": [...], "t": [0, T]} (x and y only where present; every
  case has a time interval), spatial_dimension (0, 1 or 2), boundary_description.
- You are NOT told whether a case's solution changes in time. Some cases are steady: their true equation is a
  steady class (poisson, or the steady form of convection_diffusion), u does not depend on t, and f does not
  depend on t. Decide this from the data and the forcing, report the steady class when it applies, and still
  return u on the whole (x, t) domain.""").replace(
    """unless spatial_dimension == 2; data_t is None for steady cases; data_x is None for ODE cases).""",
    """unless spatial_dimension == 2; data_x is None for ODE cases; data_t is given for every case).""").replace(
    """- forcing(*coords): the source term f of the TRUE equation, called with arrays in the order of coordinates
  (f(x, t), f(x), f(t), or f(x, y, t)); t may be 0 or larger.
- initial(*space): u at t = 0 (initial(x), initial(x, y), or initial() for an ODE); time-dependent cases only.""",
    """- forcing(...): the source term f of the TRUE equation: f(x, t), f(t) for an ODE, or f(x, y, t); t >= 0.
- initial(...): u at t = 0: initial(x), initial(x, y), or initial() for an ODE.""").replace(
    """  boundary(t) within boundary_tolerance;
- steady: "x" uniform over the domain and "u" of shape (len(x),) with the boundary values at both ends;""",
    """  boundary(t) within boundary_tolerance (also for steady cases);""").replace(
    """- boundary(t): array [u(x_left, t), u(x_right, t)] of Dirichlet values for spatial_dimension 1 (call
  boundary(0.0) for steady cases). The values may be nonzero.""",
    """- boundary(t): array [u(x_left, t), u(x_right, t)] of Dirichlet values for spatial_dimension 1. The values
  may be nonzero.""")
for _marker in ("time_dependent", "coordinates (e.g.", "steady: \"x\""):
    if _marker in HIDDEN_TIME_CONTRACT:
        raise RuntimeError(f"Hidden-time contract still mentions {_marker!r}")
DIAGNOSTIC_FIELDS = ("status", "relative_l2", "final_relative_l2", "absolute_l2", "final_absolute_l2",
                     "initial_error", "boundary_error", "runtime_seconds", "wall_seconds", "peak_memory_mb",
                     "accuracy_pass", "constraints_pass", "resources_pass", "passed", "nx", "nt", "message",
                     "identified_family", "identified_parameters", "method")


class PDEBenchProtocol(CodeProtocol):
    """pde_benchmarks.tex cases: identify the class from data and solve it by a self-chosen method, from scratch."""
    name = "pdebench"
    gpu = True

    def __init__(self, noise_fraction=NOISE_FRACTION, hide_time_dependence=False, modular=False):
        self.cases = {}
        self.noise_fraction = noise_fraction
        self.hide_time_dependence = hide_time_dependence
        self.modular = modular  # harness v2: modular parts, patch edits, per-part gate, run-time self-checks
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
        tasks = {split: make_cases(source, self.noise_fraction, self.hide_time_dependence)
                 for split, source in zip(("evolution", "development"), self.split_names)}
        self.cases = {case.task_id: case for items in tasks.values() for case in items}
        return tasks

    def starting_program(self, start, families):
        if start != "scratch":
            raise ValueError("The pdebench protocol starts from scratch only")
        if self.modular:
            from .modular import starting_program
            return starting_program()
        return {PROGRAM_KEY: STUB_PROGRAM}

    def self_test_tasks(self, evolution):
        """The first training case of each output kind (1-D evolution, steady, ODE, 2-D)."""
        chosen, kinds = [], set()
        for case in evolution:
            kind = (case.truth.dimension, case.truth.problem in STEADY)  # true kind, even when posed in time
            if kind not in kinds:
                kinds.add(kind)
                chosen.append(case)
        return chosen[:SELF_TEST_CASES]

    def problem_records(self, tasks):
        return [case.public_view() for case in tasks]

    def contract(self, families):
        from .modular import CONTRACT_ADDENDUM
        return ((HIDDEN_TIME_CONTRACT if self.hide_time_dependence else CONTRACT)
                + (CONTRACT_ADDENDUM if self.modular else "")+"\nlibrary = "+json.dumps(LIBRARY))

    def evaluate(self, tasks, version, budget, solution_directory=None):
        results = []
        if solution_directory is not None:
            project_path(solution_directory).mkdir(parents=True, exist_ok=True)
        for index, case in enumerate(tasks):
            self.cases.setdefault(case.task_id, case)
            try:
                path = solution_directory/f"{case.case_id}.npz" if solution_directory is not None else None
                results.append(run_case(case, self.code(version), budget, path, self.gpu))
            except BudgetExceeded:
                for rest in tasks[index:]:
                    row = failed_result(rest.truth, "budget_exhausted", "Case not executed; allowance exhausted", 0)
                    results.append({**row, **score_model(rest, None), "case_id": rest.case_id,
                                    "problem": rest.truth.problem, "evaluator_version": EVALUATOR_VERSION})
                break
        for row in results:  # failed solves carry the shared failed_result evaluator tag
            row["evaluator_version"] = EVALUATOR_VERSION
        return results

    def diagnostics(self, results):
        rows = []
        for result in results:
            row = {"case_id": result["case_id"], "split": result["split"],
                   **{key: result.get(key) for key in DIAGNOSTIC_FIELDS}}
            if result["split"] == "evolution":  # labeled training data
                row.update(true_family=result["true_family"],
                           true_parameters=true_parameters(self.cases[result["task_id"]].truth),
                           family_correct=result["family_correct"])
            rows.append(row)
        return rows

    def reasons(self, reasons, tasks):
        names = {case.task_id: case.case_id for case in self.cases.values()}
        cleaned = []
        for reason in reasons:
            head, _, subject = reason.partition(": ")
            if subject in names:
                cleaned.append(f"{head}: {names[subject]}")
            elif subject:
                cleaned.append(f"{head}: a hidden equation class")
            else:
                cleaned.append(reason)
        return cleaned

    def metadata(self, tasks):
        from .codegen import gpu_sandbox_self_test
        cases = [case for items in tasks.values() for case in items]
        return {"protocol_version": PROTOCOL_VERSION, "evaluator_version": EVALUATOR_VERSION, "library": LIBRARY,
                "harness": "v2-modular" if self.modular else "v1", "splits": list(self.split_names),
                "n_points": N_POINTS, "noise_fraction": self.noise_fraction,
                "hide_time_dependence": self.hide_time_dependence, "max_seconds": MAX_SECONDS,
                "memory_mb": MEMORY_MB, "self_test_cases": SELF_TEST_CASES, "grid": GRID,
                "relative_l2_tolerance": RELATIVE_L2_TOLERANCE, "methods": ["numerical", "pinn"],
                "data_sha256": data_digest(cases), "cases": {case.case_id: case.task_id for case in cases},
                "gpu_sandbox": gpu_sandbox_self_test(), "source": "pde_benchmarks.tex"}


SPEC_FILE = Path(__file__).resolve().parents[2]/"pde_benchmarks.tex"
