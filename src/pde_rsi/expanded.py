"""The 1001-case expanded release: every benchmark problem with new coefficients, initial data and boundary data.

Each of the 53 problem templates (pde_benchmarks.tex Problems 1-11 and 12-16, the 12 inventory families, and the
11 manufactured code-protocol families) gets several variants. A variant keeps the template's equation class and
geometry type and redraws its coefficients, its initial data (profile, modes, amplitudes, DiffusionPDE instance or
random field) and, where the problem has nonzero Dirichlet data, its boundary values. Variants are drawn from a
seeded generator (VERSION), split per template into train / validation / test, and checked never to repeat a case
of the project's sealed test splits, which stay private.
"""

import hashlib
import json
import math

import numpy as np

from . import inventory as inv
from . import pdebench as pb
from . import pdebench_ext as px
from .equations_bench import LIFTED
from .tasks import FAMILIES as ONE_D, GROUP_FAMILIES, Task, make_group_pilot, make_pilot, make_wave_pilot

VERSION = "expanded-1001-v1"
TEX_CASES = ["1a", "1b", "1c", "2a", "2b", "2c", "3a", "3b", "3c", "3d", "4a", "4b", "4c", "5", "6", "7", "8", "9",
             "10", "11"]
EXT_CASES = list(px.CASES)
COUNTS = {"pdebench": 20, "pdebench_ext": 20, "inventory": 20, "manufactured_1d": 15, "manufactured_wave2d": 15,
          "manufactured_group": 14}
SPLIT_SIZES = {20: (14, 3, 3), 15: (11, 2, 2), 14: (10, 2, 2)}
PROFILES = ("sine", "polynomial", "mixed_sine")
TASK_SPLIT = {"train": "evolution", "validation": "development", "test": "sealed"}  # tasks.Task split names


def rng(*parts, version=VERSION):
    return np.random.default_rng(int(hashlib.sha256(":".join([version, *map(str, parts)]).encode()).hexdigest()[:12], 16))


def split_of(index, count):
    train, validation, _ = SPLIT_SIZES[count]
    return "train" if index < train else "validation" if index < train+validation else "test"


def log_uniform(r, low, high):
    return float(np.exp(r.uniform(math.log(low), math.log(high))))


# ---------------------------------------------------------------- pde_benchmarks.tex Problems 1-11
def tex_values(problem, r, index):
    """Coefficients, initial data and boundary data of one variant (the keys of pdebench.make_spec)."""
    u = lambda a, b: float(r.uniform(a, b))  # noqa: E731
    if problem in pb.MANUFACTURED:
        values = dict(a=0.0, b=u(1.0, 3.0), T=u(0.5, 1.0), A=u(0.5, 1.5), D=log_uniform(r, 0.05, 1.0), beta=u(1.0, 2.0))
        if problem == "1a":
            del values["T"], values["beta"]
        if problem == "3a":
            values["v"] = float(r.choice([-1.0, 1.0]))*u(0.3, 1.5)
        if problem == "4a":
            values.update(A=u(0.3, 1.0), D=u(0.05, 0.5))
        if problem == "5":
            values.update(b=u(1.0, 2.0), D=u(0.1, 1.0), beta=u(2.0, 3.0))
        if problem == "7":
            values["alpha"] = u(0.3, 0.9)
        if problem == "8":
            values.update(alpha1=u(0.2, 0.5), alpha2=u(0.6, 0.9), w1=u(0.3, 1.5), w2=u(0.3, 1.5))
        if problem in LIFTED and index % 4:  # nonhomogeneous Dirichlet data in 3 of 4 variants
            values.update(bc_left=u(-1.0, 1.0), bc_right=u(-1.0, 1.0))
        return values
    if problem == "1b":
        return dict(a=0.0, b=1.0, D=log_uniform(r, 0.3, 3.0))
    if problem == "1c":
        return dict(a=0.0, b=1.0, D=log_uniform(r, 0.3, 3.0), sharpness=log_uniform(r, 10.0, 500.0), x0=u(0.3, 0.7))
    if problem == "2b":
        return dict(a=0.0, b=u(1.0, 3.0), T=u(0.5, 2.0), D=log_uniform(r, 0.05, 0.5))
    if problem == "2c":
        return dict(a=0.0, b=u(1.0, 2.0), T=u(1.0, 3.0), D=log_uniform(r, 0.05, 0.5), gamma=u(1.0, 12.0))
    if problem == "3b":
        pe = float(r.choice([-1.0, 1.0]))*log_uniform(r, 5.0, 150.0)
        D = u(0.005, 0.05)
        return dict(a=0.0, b=1.0, v=pe*D, D=D, U=u(0.5, 2.0) if index % 3 else 1.0)
    if problem == "3c":
        pe = float(r.choice([-1.0, 1.0]))*log_uniform(r, 5.0, 120.0)
        return dict(a=0.0, b=1.0, v=math.copysign(1.0, pe), D=1.0/abs(pe))
    if problem == "3d":
        L, T = u(3.0, 5.0), 1.0
        return dict(a=0.0, b=L, T=T, v=u(-0.2, 0.2)*L/T, D=u(0.005, 0.03), x0=u(0.35, 0.65)*L, sigma0=u(0.1, 0.25))
    if problem == "4b":
        return dict(a=-1.0, b=1.0, T=1.0, D=u(0.005, 0.03)/math.pi)
    if problem == "4c":
        ul = u(0.8, 1.6)
        ur = u(-0.5, ul-0.3)
        b = max(u(1.0, 2.0), (ul+ur)/2+0.5)
        return dict(a=-1.0, b=b, T=1.0, ul=ul, ur=ur, D=log_uniform(r, 0.005, 0.03))
    if problem in ("6", "9"):
        alpha = u(0.2, 0.9)
        values = dict(T=1.0, alpha=alpha, p=u(max(alpha, 0.5), 3.0))
        if problem == "9":
            values.update(a=0.0, b=math.pi, D=1.0)
        return values
    if problem == "10":
        return dict(a=0.0, b=math.pi, T=1.0, D=1.0, c=u(0.3, 1.35), p=u(0.8, 3.0))
    if problem == "11":
        return dict(a=0.0, b=math.pi, T=1.0, D=1.0, c=u(0.2, 0.95), p=u(0.8, 3.0))
    raise ValueError(problem)


def tex_spec(problem, index, version=VERSION, adjust=None):
    """adjust(problem, values, rng) may change the drawn values (RSI-PDE Bench hardening); it gets its own stream."""
    r = rng("tex", problem, index, version=version)
    values = tex_values(problem, r, index)
    if adjust is not None:
        values = adjust(problem, values, rng("adjust", problem, index, version=version))
    spec = pb.make_spec(problem, "evolution", values)
    if problem in pb.MANUFACTURED:
        spec["profile"] = PROFILES[int(r.integers(0, 3))]
    return spec


def tex_cases():
    cases = []
    count = COUNTS["pdebench"]
    for problem in TEX_CASES:
        for index in range(count):
            spec = tex_spec(problem, index)
            task = pb.BenchTask(f"x-pb-{problem}-{index:02d}", pb.CLASS[problem], split_of(index, count), spec)
            cases.append(pb.BenchCase(f"x-pb-{problem}-{index:02d}", task))
    return cases


# ---------------------------------------------------------------- pde_benchmarks.tex Problems 12-16
def maxwell_gaps(low, high):
    values = sorted({math.pi*math.hypot(m, n) for m in range(0, 6) for n in range(0, 6) if m or n})
    return [(a, b) for a, b in zip(values, values[1:]) if a >= low and b <= high and b-a > 0.3]


def ext_values(case, r):
    u = lambda a, b: float(r.uniform(a, b))  # noqa: E731
    if case == "12a":
        return dict(b=u(2*math.pi, 6*math.pi), A=u(0.5, 1.5), beta=u(1.0, 2.0), T=u(0.5, 1.0))
    if case == "12b":
        return dict(b=32*math.pi, T=u(20.0, 100.0), ic_amp=u(0.5, 1.5), ic_shift=u(0.0, 32*math.pi))
    if case == "13a":
        k = int(r.integers(1, 5))
        T = u(0.5, 2.0)
        nu = min(log_uniform(r, 0.005, 0.1), 2.0/(k*k*T))  # keep e^(-2 nu k^2 T) >= e^-4
        return dict(b=2*math.pi, d=2*math.pi, k=k, nu=nu, T=T)
    if case == "13b":
        return dict(nu=1.0/u(10.0, 100.0))
    if case == "14a":
        length = u(2.0, 4.0)
        return dict(b=length, d=length, A=u(0.3, 0.8), beta=u(1.0, 2.0), M=u(0.5, 1.5), eps=u(0.05, 0.15),
                    T=u(0.1, 0.3))
    if case == "14b":
        modes = []
        while len(modes) < 5:
            kx, ky = int(r.integers(0, 6)), int(r.integers(0, 6))
            if (kx or ky) and (kx, ky) not in [(m[1], m[2]) for m in modes]:
                modes.append([u(0.04, 0.1), kx, ky, u(0.0, 2*math.pi)])
        return dict(b=2*math.pi, d=2*math.pi, M=1.0, eps=u(0.1, 0.16), T=u(0.3, 0.5), modes=modes)
    if case == "15a":
        return dict(b=u(1.0, 2.0), d=u(1.0, 2.0), m=int(r.integers(1, 4)), n=int(r.integers(1, 4)), c=u(0.5, 2.0),
                    T=u(0.5, 2.0))
    if case == "15b":
        return dict(b=u(1.0, 2.0), d=u(1.0, 2.0), A=u(0.5, 1.5), beta=u(1.0, 2.0), c=u(0.5, 2.0), T=u(0.5, 1.0))
    integers = {key: int(r.integers(1, 4 if case == "16a" else 5)) for key in ("p", "n", "m", "q")}
    if case == "16a":
        return dict(A=u(0.5, 1.5), B=u(0.5, 1.5), k=u(0.5, 2.9), **integers)  # below the first eigenvalue pi
    gaps = maxwell_gaps(5.0, 14.0)
    low, high = gaps[int(r.integers(0, len(gaps)))]
    return dict(A=u(0.5, 1.5), B=u(0.5, 1.5), k=low+(high-low)*u(0.3, 0.7), **integers)


def ext_spec(case, index, version=VERSION):
    spec = px.make_spec(case, "evolution")
    spec.update(ext_values(case, rng("ext", case, index, version=version)))
    return spec


def ext_cases():
    cases = []
    count = COUNTS["pdebench_ext"]
    for case in EXT_CASES:
        klass, _, seconds, _ = px.CASES[case]
        for index in range(count):
            spec = ext_spec(case, index)
            split = split_of(index, count)
            task = px.ExtTask(f"x-ext-{case}-{index:02d}", klass, split, spec, max_seconds=seconds)
            cases.append(px.ExtCase(f"x-ext-{case}-{index:02d}", task))
    return cases


# ---------------------------------------------------------------- PDE Data Inventory
DATASET_FAMILIES = ("burgers_random", "darcy", "poisson", "helmholtz", "ns")  # inputs from DiffusionPDE instances


def inventory_variant(family, r, index, version=VERSION, instance=None):
    """instance: the DiffusionPDE test instance of a dataset family (default 4 + index, this release's)."""
    u = lambda a, b: float(r.uniform(a, b))  # noqa: E731
    variant = {"seed": f"{version}:{family}:{index}"}
    instance = 4+index if instance is None else instance
    if family == "burgers_gauss":
        variant.update(D=u(0.05, 0.2), ic_center=u(-3.0, 5.0), ic_width=u(0.7, 1.5))
    elif family in ("burgers_random", "darcy", "poisson"):
        variant["instance"] = instance
    elif family == "helmholtz":
        variant.update(instance=instance, k=u(0.5, 2.0))
    elif family == "ns":
        variant.update(instance=instance, D=u(5e-4, 2e-3))
    elif family == "convdiff":
        variant.update(x0=u(4.0, 9.0), sigma0=u(0.7, 1.3), c=u(0.5, 1.5), D=u(0.5, 1.5))
    elif family == "kolmogorov":
        variant.update(D=u(7e-4, 1.5e-3), mu=u(0.05, 0.15))
    return variant  # allen_cahn, burgers2d, adr, elliptic4d: new random fields and coefficients from the seed


def inventory_cases():
    cases = []
    count = COUNTS["inventory"]
    for family, (klass, _, _, seconds) in inv.FAMILIES.items():
        for index in range(count):
            spec = inv.make_spec(family, "evolution", inventory_variant(family, rng("inventory", family, index), index))
            split = split_of(index, count)
            task = inv.InvTask(f"x-inv-{family}-{index:02d}", klass, split, spec, max_seconds=seconds,
                               relative_l2_tolerance=inv.TOLERANCES.get(family, inv.TOLERANCE))
            cases.append(inv.InvCase(f"x-inv-{family}-{index:02d}", task))
    return cases


# ---------------------------------------------------------------- manufactured code-protocol suites
def manufactured_tasks():
    tasks = []
    for family in ONE_D:
        count = COUNTS["manufactured_1d"]
        for index in range(count):
            r = rng("manufactured", family, index)
            orders, weights = (), ()
            if family == "caputo":
                orders, weights = (float(r.uniform(0.3, 0.9)),), (1.0,)
            if family == "multi_term":
                orders, weights = (float(r.uniform(0.25, 0.5)), float(r.uniform(0.6, 0.9))), (float(r.uniform(0.3, 0.7)), 1.0)
            low = max(orders)+0.1 if orders else 1.0
            split = split_of(index, count)
            tasks.append(Task(
                task_id=f"x-{family}-{index:02d}", family=family, template_group=f"{family}/expanded-{index:02d}",
                split=TASK_SPLIT[split], length=float(r.uniform(0.8, 1.5)), final_time=float(r.uniform(0.2, 0.5)),
                diffusivity=float(r.uniform(0.03, 0.25)), velocity=float(r.uniform(0.2, 0.6)),
                amplitude=float(r.uniform(0.2, 0.7)), mode=int(r.integers(1, 3)), profile=PROFILES[int(r.integers(0, 3))],
                beta=float(r.uniform(low, 2.5)), orders=orders, weights=weights,
                source_id="PDE-RSI-expanded-1001").validate())
    count = COUNTS["manufactured_wave2d"]
    for index in range(count):
        r = rng("manufactured", "wave2d", index)
        speed = float(r.uniform(0.5, 1.25))
        tasks.append(Task(
            task_id=f"x-wave2d-{index:02d}", family="wave2d", template_group=f"wave2d/expanded-{index:02d}",
            split=TASK_SPLIT[split_of(index, count)], length=float(r.uniform(1.0, 1.4)), final_time=float(r.uniform(0.5, 1.0)),
            diffusivity=speed*speed, velocity=0.0, amplitude=float(r.uniform(0.2, 0.7)), mode=int(r.integers(1, 3)),
            profile=PROFILES[int(r.integers(0, 3))], beta=float(r.uniform(1.5, 3.0)), orders=(), weights=(),
            source_id="PDE-RSI-expanded-1001").validate())
    for family in GROUP_FAMILIES:
        count = COUNTS["manufactured_group"]
        for index in range(count):
            r = rng("manufactured", family, index)
            tasks.append(Task(
                task_id=f"x-{family}-{index:02d}", family=family, template_group=f"{family}/expanded-{index:02d}",
                split=TASK_SPLIT[split_of(index, count)], length=1.0 if family == "fode" else math.pi, final_time=1.0,
                diffusivity=float(r.uniform(0.5, 1.5)), velocity=0.0, amplitude=float(r.uniform(0.5, 1.5)),
                mode=1 if family == "fode" else int(r.integers(1, 3)), profile="sine", beta=float(r.uniform(2.0, 4.0)),
                orders=(float(r.uniform(0.2, 0.9)),), weights=(1.0,),
                output_time_points=65 if family == "fode" else 17, source_id="PDE-RSI-expanded-1001").validate())
    return tasks


def release_split(task):
    return {v: k for k, v in TASK_SPLIT.items()}[task.split]


# ---------------------------------------------------------------- checks
def _key(spec, drop=("dataset_instance",)):
    """Spec identity: every value, with gridded input fields replaced by a hash of their contents."""
    fields = spec.get("fields") or {}
    digest = hashlib.sha256(b"".join(name.encode()+np.ascontiguousarray(fields[name], dtype=float).tobytes()
                                     for name in sorted(fields))).hexdigest()
    return json.dumps({**{k: v for k, v in spec.items() if k not in drop and k != "fields"}, "fields": digest},
                      sort_keys=True, default=float)


def sealed_overlap():
    """Expanded cases whose spec equals any existing case (public or sealed) of the project; must be empty."""
    existing = {_key(c.truth.spec) for split in pb.SPLITS+pb.SEALED_SPLITS[1:] for c in pb.make_cases(split)}
    existing |= {_key(c.truth.spec) for split in px.SPLITS for c in px.make_cases(split)}
    existing |= {_key(inv.make_spec(f, s)) for f in inv.FAMILIES for s in inv.SPLITS+inv.SEALED_SPLITS[1:]}
    manufactured = {json.dumps({k: v for k, v in t.to_dict().items() if k not in ("task_id", "template_group", "split",
                                                                                  "source_id", "reuse_permission")},
                               sort_keys=True) for t in make_pilot()+make_wave_pilot()+make_group_pilot()}
    clashes = [c.case_id for c in tex_cases()+ext_cases() if _key(c.truth.spec) in existing]
    clashes += [c.case_id for c in inventory_cases() if _key(c.truth.spec) in existing]
    clashes += [t.task_id for t in manufactured_tasks()
                if json.dumps({k: v for k, v in t.to_dict().items() if k not in ("task_id", "template_group", "split",
                                                                                 "source_id", "reuse_permission")},
                              sort_keys=True) in manufactured]
    return clashes


def all_cases():
    return {"pdebench": tex_cases(), "pdebench_ext": ext_cases(), "inventory": inventory_cases(),
            "manufactured": manufactured_tasks()}
