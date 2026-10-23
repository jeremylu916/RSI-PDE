"""PDE Data Inventory suite (PDE_Data_Inventory_Overleaf.pdf, X. Kong, 19 Oct 2026) under the student protocol.

Twelve families, one case per family in each of training (evolution), validation (development), and a fresh
sealed split (inventory_test). The agent identifies the equation class and its scalar coefficients from 256 noisy
samples and black-box forcing / initial / boundary / coefficient data, then returns the solution.

Inputs follow the inventory and, where it leaves gaps, the DiffusionPDE data and generators (Huang et al. 2024):
Burgers with random periodic initial data, Darcy, Poisson, Helmholtz, and unit-torus Navier-Stokes take their
input fields from the DiffusionPDE test set (instances 0/1/2 for training/validation/sealed). The other families
are generated from the inventory's formulas with per-case random inputs. References are recomputed at high
resolution (inventory_reference.py) and stored under data/inventory/references (evaluator-only).
Documented choices where neither source fixes a value: Kolmogorov horizon T = 1 (inventory: 10 s on 2048^2),
Gaussian-pulse initial data for linear convection-diffusion, unit-L2 forcing modes for 2-D Burgers, unit-variance
zero-mean Matern-3/2 field for the 4-D coefficient, m = 4 active diffusivity modes for ADR.
"""

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import RegularGridInterpolator

from . import equations_inventory as eq
from .code_agents import CodeProtocol
from .codegen import ALLOWED_MODULES, MAX_CODE_CHARS, MAX_OUTPUT_VALUES, STUB_PROGRAM, run_program_task
from .evaluator import failed_result
from .runner import BudgetExceeded
from .storage import PROJECT_ROOT, project_path

VERSION = "inventory-v1"
EVALUATOR_VERSION = "inventory-evaluator-v1"
SPLITS = ("evolution", "development", "inventory_test")
# inventory_test2: a fresh sealed split created on 2026-10-20 to evaluate harness v2, after inventory_test had been
# used to diagnose the harness. DiffusionPDE instance 3, new per-case draws. Evaluator-only.
SEALED_SPLITS = ("inventory_test", "inventory_test2")
PREFIX = {"evolution": "a", "development": "b", "inventory_test": "c", "inventory_test2": "d"}
INSTANCE = {"evolution": 0, "development": 1, "inventory_test": 2, "inventory_test2": 3}
DATA_DIR = PROJECT_ROOT/"data/inventory"
INPUTS = DATA_DIR/"inputs.npz"
REFERENCES = DATA_DIR/"references"
N_POINTS = 256
NOISE_FRACTION = 0.01
TOLERANCE = 2e-3
CONSTRAINT_TOLERANCE = 1e-10
MEMORY_MB = 16384
PROGRAM_KEY = "identify"
SELF_TEST_KINDS = 5

# family: (library class, kind, output layout, max seconds)
FAMILIES = {
    "burgers_gauss": ("burgers", "burgers_gauss", "1d_periodic", 90.0),
    "burgers_random": ("burgers", "burgers_random", "1d_periodic", 90.0),
    "convdiff": ("convection_diffusion", "convdiff", "1d_dirichlet", 90.0),
    "darcy": ("darcy_2d", "darcy", "2d_steady", 300.0),
    "poisson": ("poisson_2d", "poisson", "2d_steady", 300.0),
    "helmholtz": ("helmholtz_2d", "helmholtz", "2d_steady", 300.0),
    "ns": ("vorticity_ns", "ns", "2d_periodic_time", 300.0),
    "kolmogorov": ("vorticity_ns", "kolmogorov", "2d_periodic_time", 300.0),
    "allen_cahn": ("allen_cahn_2d", "allen_cahn", "2d_periodic_time", 300.0),
    "burgers2d": ("burgers_2d", "burgers2d", "2d_periodic_time", 300.0),
    "adr": ("adr_2d", "adr", "2d_periodic_time", 300.0),
    "elliptic4d": ("elliptic_4d", "elliptic4d", "4d_points", 300.0),
}
TITLES = {"burgers_gauss": "1-D Burgers, fixed Gaussian initial state (inventory 1.1)",
          "burgers_random": "1-D Burgers, random periodic initial state (inventory 1.2; DiffusionPDE data)",
          "convdiff": "1-D linear convection-diffusion (inventory 10)",
          "darcy": "Darcy flow, binary permeability (inventory 3; DiffusionPDE data)",
          "poisson": "Poisson equation, random forcing (inventory 4.1; DiffusionPDE data)",
          "helmholtz": "Inhomogeneous Helmholtz, k = 1 (inventory 4.2; DiffusionPDE data)",
          "ns": "2-D Navier-Stokes vorticity, unit torus (inventory 5; DiffusionPDE data)",
          "kolmogorov": "2-D Kolmogorov flow with drag (inventory 6)",
          "allen_cahn": "Heterogeneous Allen-Cahn (inventory 7)",
          "burgers2d": "2-D stochastic scalar Burgers (inventory 2)",
          "adr": "Stochastic nonlinear advection-diffusion-reaction (inventory 8)",
          "elliptic4d": "4-D elliptic, invariant coefficient (inventory 9)"}

LIBRARY_ADDITIONS = {
    "darcy_2d": {"equation": "-div(a(x,y) grad u) = f(x,y) (steady); a = problem.coefficient(x, y)", "parameters": {}},
    "poisson_2d": {"equation": "u_xx + u_yy = f(x,y) (steady)", "parameters": {}},
    "helmholtz_2d": {"equation": "u_xx + u_yy + k**2 * u = f(x,y) (steady)", "parameters": {"k": "wavenumber > 0"}},
    "vorticity_ns": {"equation": "w_t + v.grad(w) = nu*(w_xx + w_yy) - mu*w + q(x,y), v = (psi_y, -psi_x), "
                                 "-(psi_xx + psi_yy) = w (periodic)",
                     "parameters": {"nu": "viscosity > 0", "mu": "linear drag >= 0"}},
    "allen_cahn_2d": {"equation": "u_t = div(kappa(x,y) grad u) + lam*(u - u**3) + f; kappa = problem.coefficient(x, y)",
                      "parameters": {"lam": "reaction rate > 0"}},
    "burgers_2d": {"equation": "u_t + cx*u*u_x + cy*u*u_y = div(kappa(x,y) grad u) + f(x,y); kappa = problem.coefficient(x, y)",
                   "parameters": {"cx": "x transport coefficient", "cy": "y transport coefficient"}},
    "adr_2d": {"equation": "u_t + cx*u*u_x + cy*u*u_y = div(kappa(x,y) grad u) + lam*(u - u**3) + f(x,y); "
                           "kappa = problem.coefficient(x, y)",
               "parameters": {"cx": "x transport coefficient", "cy": "y transport coefficient", "lam": "reaction rate"}},
    "elliptic_4d": {"equation": "-div(a(x) grad u) = f(x) in B^3 x (-1, 1) (steady), u = 0 on the boundary; "
                                "a = problem.coefficient(x1, x2, x3, z)", "parameters": {}},
}


def library():
    from .pdebench import LIBRARY
    return {**LIBRARY, **LIBRARY_ADDITIONS}


# ---------------------------------------------------------------- per-case inputs
def _rng(*parts):
    return np.random.default_rng(int(hashlib.sha256(":".join([VERSION, *map(str, parts)]).encode()).hexdigest()[:12], 16))


def grf_torus(n, length, rng):
    """Inventory 6: w0 ~ N(0, 7^{3/2} (-Lap + 49 I)^{-5/2}) on the periodic [0, length)^2 grid (real field)."""
    k = 2*math.pi/length*np.fft.fftfreq(n, d=1.0/n)
    kx, ky = np.meshgrid(k, k)
    variance = 7**1.5*(kx**2+ky**2+49.0)**-2.5
    variance[0, 0] = 0.0  # zero-mean vorticity (periodic velocity has zero mean curl)
    noise = rng.standard_normal((n, n))
    # real white noise -> spectral colouring with the covariance's eigenvalues (L2 normalisation on the torus)
    field_hat = np.fft.fft2(noise)*np.sqrt(variance)/length*n
    return np.real(np.fft.ifft2(field_hat))


def matern32_field(rng, n=65, ell=0.20):
    """Separable Matern-3/2 field on the (q, z) grid of [0, 1] x [-1, 1] with unit variance, mean 0."""
    def chol(points):
        r = np.abs(points[:, None]-points[None, :])
        c = (1+math.sqrt(3)*r/ell)*np.exp(-math.sqrt(3)*r/ell)
        return np.linalg.cholesky(c+1e-10*np.eye(len(points)))
    lq, lz = chol(np.linspace(0, 1, n)), chol(np.linspace(-1, 1, n))
    return lq@rng.standard_normal((n, n))@lz.T


EXPANDED_INPUTS = PROJECT_ROOT/"data/expanded/inventory_inputs.npz"  # further DiffusionPDE instances (1001-case release)
VARIANT_KEYS = ("D", "x0", "sigma0", "c", "k", "mu", "ic_center", "ic_width")


BENCH_INPUTS = (PROJECT_ROOT/"data/bench/inventory_inputs.npz",  # RSI-PDE Bench train and validation instances
                PROJECT_ROOT/"evaluator_private/bench/inventory_inputs.npz")  # its hidden-test instances


def _dataset_fields(family, split, instance=None):
    instance = INSTANCE[split] if instance is None else instance
    sources = (INPUTS,) if instance in INSTANCE.values() else (EXPANDED_INPUTS, *BENCH_INPUTS)
    key = f"{family}_{instance}"
    for source in sources:
        if not source.exists():
            continue
        with np.load(source) as archive:
            fields = {name.split("__", 1)[1]: archive[name] for name in archive.files if name.startswith(key+"__")}
        if fields:
            return fields
    raise KeyError(f"No inputs for {family} instance {instance} in {[str(s) for s in sources]} "
                   "(run scripts/extract_inventory_inputs.py)")


def make_spec(family, split, variant=None):
    """The case spec of a family in a split. A variant (expanded release) gives its own random seed, DiffusionPDE
    instance and coefficient overrides (VARIANT_KEYS); without one the split's fixed values apply."""
    klass, kind, layout, _ = FAMILIES[family]
    variant = variant or {}
    rng = _rng(family, variant.get("seed", split))
    instance = variant.get("instance")
    if variant:  # values of the fixed tables below are taken from the training split, then overridden
        split_values = "evolution"
    else:
        split_values = split
    spec = {"family_key": family, "kind": kind, "layout": layout, "a": 0.0, "b": 1.0, "T": None}
    if kind == "burgers_gauss":
        spec.update(a=-8.0, b=8.0, T=10.0, D={"evolution": 0.1, "development": 0.07, "inventory_test": 0.15,
                                              "inventory_test2": 0.12}[split_values])
    elif kind == "burgers_random":
        spec.update(a=0.0, b=1.0, T=1.0, D=0.01, fields=_dataset_fields(family, split, instance))
    elif kind == "convdiff":
        x0, s0 = {"evolution": (5.0, 1.0), "development": (4.0, 0.8), "inventory_test": (6.0, 1.2),
                  "inventory_test2": (7.0, 0.9)}[split_values]
        spec.update(a=0.0, b=20.0, T=5.0, c=1.0, D=1.0, x0=x0, sigma0=s0)
    elif kind == "darcy":
        spec.update(fields=_dataset_fields(family, split, instance))
    elif kind == "poisson":
        spec.update(fields=_dataset_fields(family, split, instance))
    elif kind == "helmholtz":
        spec.update(k=1.0, fields=_dataset_fields(family, split, instance))
    elif kind == "ns":
        spec.update(T=1.0, D=1e-3, mu=0.0, fields=_dataset_fields(family, split, instance))
    elif kind == "kolmogorov":
        spec.update(b=2*math.pi, T=1.0, D=1e-3, mu=0.1, fields={"w0": grf_torus(128, 2*math.pi, rng)})
    elif kind == "allen_cahn":
        xi = rng.uniform(-1, 1, 57)
        spec.update(b=2*math.pi, T=1.0, xi=xi.tolist(), lam=0.80+0.20*xi[56])
    elif kind == "burgers2d":
        xi = rng.uniform(-1, 1, 20)
        spec.update(b=2*math.pi, T=0.25, xi=xi.tolist(), cx=0.90+0.15*xi[14], cy=0.45+0.10*xi[15])
    elif kind == "adr":
        m = 4
        diff = rng.standard_normal(4*2*m)
        diff = diff[np.abs(diff) <= 1][:2*m]  # standard normals conditioned on [-1, 1]
        xi = np.zeros(60)
        xi[:2*m] = diff
        xi[56], xi[57] = rng.uniform(-1, 1, 2)
        xi[58], xi[59] = 2*rng.beta(2, 2, 2)-1
        spec.update(b=2*math.pi, T=2.0, m=m, xi=xi.tolist(), cx=0.55+0.15*xi[56], cy=0.30+0.10*xi[57],
                    lam=0.45+0.15*xi[58], gamma=0.04*(1+0.25*xi[59]))
    elif kind == "elliptic4d":
        g = matern32_field(rng)
        spec.update(a=-1.0, b=1.0, fields={"ahat": 1.25+0.75*np.tanh(0.8*g)})
    spec["class"] = klass
    for key in VARIANT_KEYS:
        if key in variant:
            spec[key] = variant[key]
    if instance is not None:
        spec["dataset_instance"] = instance  # which DiffusionPDE test instance the gridded inputs come from
    return spec


def true_parameters(spec):
    kind = spec["kind"]
    if kind in ("burgers_gauss", "burgers_random"):
        return {"D": spec["D"]}
    if kind == "convdiff":
        return {"D": spec["D"], "v": spec["c"]}
    if kind == "helmholtz":
        return {"k": spec["k"]}
    if kind in ("ns", "kolmogorov"):
        return {"nu": spec["D"], "mu": spec["mu"]}
    if kind == "allen_cahn":
        return {"lam": spec["lam"]}
    if kind == "burgers2d":
        return {"cx": spec["cx"], "cy": spec["cy"]}
    if kind == "adr":
        return {"cx": spec["cx"], "cy": spec["cy"], "lam": spec["lam"]}
    return {}


@dataclass(frozen=True)
class InvTask:
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

    def to_dict(self):
        return {"task_id": self.task_id, "family": self.family, "split": self.split}


def evaluation_axes(task):
    s, layout = task.spec, task.layout
    if layout in ("1d_periodic", "1d_dirichlet"):
        endpoint = layout == "1d_dirichlet"
        x = np.linspace(s["a"], s["b"], 512, endpoint=endpoint) if endpoint else s["a"]+(s["b"]-s["a"])*np.arange(512)/512
        return {"x": x, "t": np.linspace(0, s["T"], 500)}
    if layout == "2d_steady":
        nodes = np.linspace(0, 1, 256)
        return {"x": nodes, "y": nodes}
    if layout == "2d_periodic_time":
        nodes = s["a"]+(s["b"]-s["a"])*np.arange(128)/128
        return {"x": nodes, "y": nodes, "t": np.linspace(0, s["T"], 11)}
    return {}


HIDDEN_REFERENCES = PROJECT_ROOT/"evaluator_private/bench/references"  # RSI-PDE Bench hidden test


def reference_path(task):
    if task.task_id.startswith("bench-") and task.split == "test":
        return HIDDEN_REFERENCES/f"{task.task_id}.npz"
    return REFERENCES/f"{task.task_id}.npz"


_REFERENCE = {}


def reference(task):
    """Stored reference: evaluation-grid solution (and 4-D points) plus noise-free data samples."""
    if task.task_id not in _REFERENCE:
        with np.load(reference_path(task)) as archive:
            _REFERENCE[task.task_id] = {name: archive[name] for name in archive.files}
    return _REFERENCE[task.task_id]


@dataclass(frozen=True)
class InvCase:
    case_id: str
    truth: InvTask
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
        ref = reference(self.truth)
        clean = ref["sample_u"]
        rng = np.random.default_rng(int(hashlib.sha256(f"{VERSION}:noise:{self.task_id}".encode()).hexdigest()[:8], 16))
        std = self.noise_fraction*float(np.sqrt(np.mean(clean**2)))
        data = {"u": clean+std*rng.standard_normal(clean.shape), "noise_std": std}
        for axis in ("x", "y", "t", "points"):
            data[axis] = ref.get(f"sample_{axis}")
        return data

    def public_view(self):
        s, task = self.truth.spec, self.truth
        layout = task.layout
        domain = {"x": [s["a"], s["b"]]} if layout != "4d_points" else {"x1,x2,x3": "unit ball |x'| < 1", "z": [-1.0, 1.0]}
        if layout == "2d_steady":
            domain = {"x": [0.0, 1.0], "y": [0.0, 1.0]}
        if layout == "2d_periodic_time":
            domain["y"] = [s["a"], s["b"]]
        if s.get("T"):
            domain["t"] = [0.0, s["T"]]
        boundary = {"1d_periodic": "periodic in x", "1d_dirichlet": "Dirichlet: [u(x_left, t), u(x_right, t)] = boundary(t)",
                    "2d_steady": "Dirichlet u = 0 on the four edges", "2d_periodic_time": "periodic in x and y",
                    "4d_points": "Dirichlet u = 0 on the boundary of B^3 x (-1, 1)"}[layout]
        axes = evaluation_axes(task)
        view = {"case_id": self.case_id, "split": self.split, "layout": layout, "domain": domain,
                "boundary_description": boundary, "n_data": len(reference(task)["sample_u"]),
                "noise_std": self.data()["noise_std"], "relative_l2_tolerance": task.relative_l2_tolerance,
                "boundary_tolerance": task.boundary_tolerance, "initial_tolerance": task.initial_tolerance,
                "max_seconds": task.max_seconds, "memory_mb": task.memory_mb,
                "has_coefficient_field": s["kind"] in ("darcy", "allen_cahn", "burgers2d", "adr", "elliptic4d"),
                "gridded_inputs": sorted((s.get("fields") or {}).keys()),
                "evaluation_grid": {k: len(v) for k, v in axes.items()}}
        if layout == "4d_points":
            view["n_output_points"] = len(reference(task)["points"])
        return view


def make_cases(split, noise_fraction=NOISE_FRACTION):
    if split.startswith("bench-"):  # RSI-PDE Bench splits (src/pde_rsi/bench/splits.py)
        from .bench.splits import bench_cases
        return bench_cases("inventory", split, noise_fraction)
    if split not in SPLITS+SEALED_SPLITS[1:]:
        raise ValueError(f"split must be one of {SPLITS+SEALED_SPLITS[1:]}")
    tasks = []
    for family, (klass, _, _, seconds) in FAMILIES.items():
        task_id = f"inv-{family}-{split}"
        tasks.append(InvTask(task_id, klass, split, make_spec(family, split), max_seconds=seconds,
                             relative_l2_tolerance=TOLERANCES.get(family, TOLERANCE)))
    tasks.sort(key=lambda t: hashlib.sha256(f"{VERSION}:{t.task_id}".encode()).hexdigest())
    return [InvCase(f"case-{PREFIX[split]}{index:02d}", task, noise_fraction) for index, task in enumerate(tasks)]


# Per-family pass tolerances, set from the measured reference self-convergence (build_inventory_references.py
# report): the tolerance is at least 5x the reference error. Families not listed use TOLERANCE.
TOLERANCES = {
    # 4-D: the 63^4 staircased finite-volume reference differs from 21^4 by 2.1-2.8 % and from an interpolated 45^4
    # solve by 0.9 %, so its own error is at most about 1 %; the tolerance is 5 %. All other families keep 2e-3
    # (measured reference errors 1e-16 to 3.5e-4; data/inventory/reference_report.json).
    "elliptic4d": 5e-2,
}


# ---------------------------------------------------------------- evaluator
def _uniform(axis, low, high, name, endpoint=True):
    if axis.ndim != 1 or len(axis) < 3 or not np.all(np.diff(axis) > 0):
        raise ValueError(f"{name} must be strictly increasing with at least 3 points")
    if not np.allclose(np.diff(axis), axis[1]-axis[0], rtol=1e-9, atol=1e-13):
        raise ValueError(f"Expected a uniform {name} grid")
    if not (np.isclose(axis[0], low, atol=1e-12, rtol=0) and np.isclose(axis[-1], high, atol=1e-12, rtol=0)):
        raise ValueError(f"{name} grid must run from {low} to {high} (both endpoints included)")


def _relative(error, truth, weights=None):
    if weights is None:
        return float(np.sqrt(np.sum(error**2)/max(np.sum(truth**2), 1e-300)))
    return float(np.sqrt(weights(error**2)/max(weights(truth**2), 1e-300)))


def evaluate(task, solution, runtime_seconds=0.0, wall_seconds=0.0, peak_memory_mb=0.0):
    s, layout = task.spec, task.layout
    ref = reference(task)
    u = np.asarray(solution.u, dtype=float)
    if not np.isfinite(u).all():
        raise ValueError("Non-finite solution")
    boundary_error = initial_error = 0.0
    if layout == "4d_points":
        values = u.reshape(-1)
        if values.shape != ref["u"].shape:
            raise ValueError(f"Return u at the {len(ref['u'])} problem.points (shape ({len(ref['u'])},))")
        error, truth = values-ref["u"], ref["u"]
        relative = final = _relative(error, truth)
    elif layout == "2d_steady":
        x, y = np.asarray(solution.x), np.asarray(solution.y)
        if u.ndim == 3 and u.shape[0] == 1:
            u = u[0]
        if u.shape != (len(y), len(x)):
            raise ValueError("u must have shape (len(y), len(x))")
        _uniform(x, 0, 1, "x")
        _uniform(y, 0, 1, "y")
        ax = evaluation_axes(task)
        yy, xx = np.meshgrid(ax["y"], ax["x"], indexing="ij")
        predicted = RegularGridInterpolator((y, x), u)(np.stack([yy, xx], -1))
        error, truth = predicted-ref["u"], ref["u"]
        integrate = lambda f: trapezoid(trapezoid(f, ax["x"], axis=1), ax["y"])  # noqa: E731
        relative = final = _relative(error, truth, integrate)
        boundary_error = float(max(np.abs(u[[0, -1], :]).max(), np.abs(u[:, [0, -1]]).max()))
    elif layout == "2d_periodic_time":
        x, y, t = np.asarray(solution.x), np.asarray(solution.y), np.asarray(solution.t)
        if u.shape != (len(t), len(y), len(x)):
            raise ValueError("u must have shape (len(t), len(y), len(x))")
        _uniform(x, s["a"], s["b"], "x")
        _uniform(y, s["a"], s["b"], "y")
        if len(t) < 3 or not np.all(np.diff(t) > 0) or abs(t[0]) > 1e-12 or not np.isclose(t[-1], s["T"], atol=1e-12):
            raise ValueError("t must increase from 0 to T with at least 3 points")
        ax = evaluation_axes(task)
        tt, yy, xx = np.meshgrid(ax["t"], ax["y"], ax["x"], indexing="ij")
        predicted = RegularGridInterpolator((t, y, x), u)(np.stack([tt, yy, xx], -1))
        error, truth = predicted-ref["u"], ref["u"]
        space = lambda f: f.sum(axis=(-1, -2))  # noqa: E731 - uniform periodic grid: plain sums
        err_t, norm_t = space(error**2), space(truth**2)
        relative = float(np.sqrt(trapezoid(err_t, ax["t"])/trapezoid(norm_t, ax["t"])))
        final = float(np.sqrt(err_t[-1]/norm_t[-1]))
        boundary_error = float(max(np.abs(u[:, :, 0]-u[:, :, -1]).max(), np.abs(u[:, 0, :]-u[:, -1, :]).max()))
        X, Y = np.meshgrid(x, y)
        initial_error = float(np.abs(u[0]-eq.initial(s, X, Y)).max())
    else:  # 1-D in time
        x, t = np.asarray(solution.x), np.asarray(solution.t)
        if u.shape != (len(t), len(x)):
            raise ValueError("u must have shape (len(t), len(x))")
        _uniform(x, s["a"], s["b"], "x")
        if len(t) < 3 or not np.all(np.diff(t) > 0) or abs(t[0]) > 1e-12 or not np.isclose(t[-1], s["T"], atol=1e-12):
            raise ValueError("t must increase from 0 to T with at least 3 points")
        ax = evaluation_axes(task)
        tt, xx = np.meshgrid(ax["t"], ax["x"], indexing="ij")
        predicted = RegularGridInterpolator((t, x), u)(np.stack([tt, xx], -1))
        error, truth = predicted-ref["u"], ref["u"]
        err_t = trapezoid(error**2, ax["x"], axis=1) if layout == "1d_dirichlet" else (error**2).sum(axis=1)
        norm_t = trapezoid(truth**2, ax["x"], axis=1) if layout == "1d_dirichlet" else (truth**2).sum(axis=1)
        relative = float(np.sqrt(trapezoid(err_t, ax["t"])/trapezoid(norm_t, ax["t"])))
        final = float(np.sqrt(err_t[-1]/norm_t[-1]))
        if layout == "1d_dirichlet":
            values = np.array([eq.boundary(s, float(time)) for time in t])
            boundary_error = float(np.abs(u[:, [0, -1]]-values).max())
        else:
            boundary_error = float(np.abs(u[:, 0]-u[:, -1]).max())
        initial_error = float(np.abs(u[0]-eq.initial(s, x)).max())
    accuracy = relative <= task.relative_l2_tolerance and final <= task.relative_l2_tolerance
    constraints = boundary_error <= task.boundary_tolerance and initial_error <= task.initial_tolerance
    resources = wall_seconds <= task.max_seconds and peak_memory_mb <= task.memory_mb
    return {"task_id": task.task_id, "family": task.family, "split": task.split, "status": "ok",
            "relative_l2": relative, "final_relative_l2": final,
            "absolute_l2": float(np.sqrt(np.mean(error**2))), "final_absolute_l2": float(np.sqrt(np.mean(error**2))),
            "absolute_linf": float(np.abs(error).max()), "boundary_error": boundary_error, "initial_error": initial_error,
            "physical_residual_relative": None, "runtime_seconds": runtime_seconds, "wall_seconds": wall_seconds,
            "peak_memory_mb": peak_memory_mb, "accuracy_pass": accuracy, "constraints_pass": constraints,
            "resources_pass": resources, "passed": accuracy and constraints and resources, "method": solution.method,
            "nx": len(np.asarray(solution.x)), "nt": len(np.asarray(solution.t)), "evaluator_version": EVALUATOR_VERSION}


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
    out = {}
    for key, value in spec.items():
        if key == "fields":
            out[key] = {name: np.asarray(array).tolist() for name, array in value.items()}
        elif isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def run_case(case, code, budget, solution_path=None, gpu=True):
    data = case.data()
    ref = reference(case.truth)

    def postprocess(result, extras):
        model = None
        if "model_json" in extras:
            try:
                model = json.loads(str(extras["model_json"]))
            except ValueError:
                model = None
        return {**result, **score_model(case, model), "case_id": case.case_id}

    view = {**case.public_view(), "library": library()}
    hidden = _jsonable(case.truth.spec)
    if case.truth.layout == "4d_points":
        hidden["points"] = ref["points"].tolist()
    result = run_program_task(
        case.truth, code, budget, solution_path,
        request_overrides={"mode": "inventory", "gpu": gpu, "problem": view, "hidden": hidden,
                           "data": {k: (None if data[k] is None else np.asarray(data[k]).tolist())
                                    for k in ("x", "y", "t", "points", "u")}},
        postprocess=postprocess, evaluator=evaluate)
    if "case_id" not in result:
        result.update(score_model(case, None), case_id=case.case_id)
    result["problem"] = case.truth.spec["family_key"]
    return result


CONTRACT = f"""Task: each case gives you measurements of an UNKNOWN equation from the library below with UNKNOWN scalar
coefficients. Identify the equation class and its coefficients from the data, choose a solution method (numerical
or PINN), and return an accurate solution over the whole domain. One program (family "{PROGRAM_KEY}") defines
solve(problem) and is used for every case.

problem attributes:
- case_id, split, layout (one of "1d_periodic", "1d_dirichlet", "2d_steady", "2d_periodic_time", "4d_points"),
  domain, boundary_description, evaluation_grid (sizes of the scoring grid), relative_l2_tolerance,
  boundary_tolerance, initial_tolerance, max_seconds, memory_mb, device ("cuda" or "cpu").
- data_x, data_y, data_t, data_points, data_u: n_data noisy samples of the solution (only the coordinates the
  layout has; data_points is an (n, 4) array for 4d_points). Gaussian noise with standard deviation noise_std.
- forcing(*coords): the source term f of the TRUE equation, with arrays in coordinate order: f(x, t), f(x, y),
  f(x, y, t), or f(x1, x2, x3, z).
- initial(*space): u (or vorticity w) at t = 0: initial(x) or initial(x, y); time-dependent layouts only.
- boundary(t): [u(x_left, t), u(x_right, t)] for "1d_dirichlet".
- coefficient(*space): the known spatial coefficient field a or kappa of the equation when has_coefficient_field.
- grids: dict of the gridded input arrays named in gridded_inputs (DiffusionPDE-style data), each with its
  coordinates: "u0" (periodic samples of the initial state), "w0" (periodic initial vorticity, rows = y),
  "a" (cell-centred permeability on [0,1]^2, rows = y), "f" (node values of the forcing on [0,1]^2 including the
  boundary, rows = y), "ahat" (4-D coefficient a-hat(q, z) on [0,1] x [-1,1]). The callables are the exact
  continuous inputs: trigonometric interpolation of periodic grids, piecewise constant cells, bicubic spline of
  node grids.
- points: (N, 4) array of the points where a 4d_points solution must be returned.
- library: dict {{class name: {{"equation": ..., "parameters": {{name: meaning}}}}}} of candidate equations.

solve(problem) returns a dict with "model": {{"family": <class name>, "parameters": {{name: value}},
"method": "numerical" or "pinn"}} and the solution:
- 1d_periodic / 1d_dirichlet: "x" uniform over the domain including both endpoints, "t" strictly increasing from
  0 to T (at least 3 points), "u" of shape (len(t), len(x)); u[0] = initial(x); periodic: u[:, 0] = u[:, -1];
  Dirichlet: u[:, 0], u[:, -1] = boundary(t).
- 2d_steady: "x", "y" uniform over [0, 1] including endpoints, "u" of shape (len(y), len(x)), u = 0 on the edges.
- 2d_periodic_time: "x", "y" uniform over the domain including both endpoints, "t" from 0 to T, "u" of shape
  (len(t), len(y), len(x)) with u[0] = initial(X, Y) and equal values on opposite edges.
- 4d_points: "u" of shape (N,) with the solution at problem.points.

Scoring: the solution is linearly interpolated onto the evaluation grid (4d_points: compared at the points) and
compared with a high-accuracy reference of the TRUE equation: space-time and final-time relative L2 (steady and
4-D: one relative L2). A case passes when both are at most relative_l2_tolerance, the initial and boundary
conditions hold, and the solve fits max_seconds (wall and CPU, including start-up) and memory_mb. Time-dependent
2-D cases are scored at evaluation_grid["t"] equally spaced times from 0 to T. The reported model is scored
separately and does not affect acceptance. Feedback for training (evolution) cases includes the true class and
parameters; validation feedback does not.

Allowed imports: {", ".join(sorted(ALLOWED_MODULES))}. Forbidden: file, network, or process access; eval, exec,
compile, open, getattr, setattr, type, dir; names or attributes beginning with "_"; frame or code introspection.
Programs return at most {MAX_OUTPUT_VALUES:,} values and are at most {MAX_CODE_CHARS:,} characters. Write genuine
fitting and numerical solution code; never hard-code answers or branch on case_id.

Solver methods: numerical (finite differences/volumes, spectral, ...) with NumPy/SciPy, or pinn (PyTorch on the
GPU, problem.device); they may be combined. One shared GPU and one CPU thread."""

DIAGNOSTIC_FIELDS = ("status", "relative_l2", "final_relative_l2", "initial_error", "boundary_error", "runtime_seconds",
                     "wall_seconds", "peak_memory_mb", "accuracy_pass", "constraints_pass", "resources_pass", "passed",
                     "message", "identified_family", "identified_parameters", "method")


class InventoryProtocol(CodeProtocol):
    name = "inventory"
    gpu = True

    def __init__(self, noise_fraction=NOISE_FRACTION, modular=False):
        self.cases, self.noise_fraction = {}, noise_fraction
        self.modular = modular  # harness v2: modular parts, patch edits, per-part gate, run-time self-checks
        if modular:
            from .modular import INITIAL_WORKFLOW
            self.initial_workflow = INITIAL_WORKFLOW

    def classes(self):
        return list(library())

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
            raise ValueError("The inventory protocol starts from scratch only")
        if self.modular:
            from .modular import starting_program
            return starting_program()
        return {PROGRAM_KEY: STUB_PROGRAM}

    def self_test_tasks(self, evolution):
        """The first training case of each output layout."""
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
        return CONTRACT+(CONTRACT_ADDENDUM if self.modular else "")+"\nlibrary = "+json.dumps(library())

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
                                    "problem": rest.truth.spec["family_key"]})
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
                row.update(true_family=result["true_family"], true_parameters=true_parameters(case.truth.spec),
                           family_correct=result["family_correct"])
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
        return {"protocol_version": VERSION, "evaluator_version": EVALUATOR_VERSION, "library": library(),
                "harness": "v2-modular" if self.modular else "v1", "splits": list(self.split_names),
                "n_points": N_POINTS, "noise_fraction": self.noise_fraction, "tolerances": {**TOLERANCES},
                "default_tolerance": TOLERANCE, "families": {k: list(v) for k, v in FAMILIES.items()},
                "cases": {case.case_id: case.task_id for items in tasks.values() for case in items},
                "gpu_sandbox": gpu_sandbox_self_test(), "source": "PDE_Data_Inventory_Overleaf.pdf + DiffusionPDE test data"}
