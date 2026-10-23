"""Task specifications, grouped splits, and self-generated pilot provenance."""

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math

FAMILIES = ("poisson", "heat", "convection_diffusion", "burgers", "caputo", "multi_term")
# Families added after the published 1D benchmark; stored in separately versioned files.
WAVE_FAMILIES = ("wave2d",)
# Group FPDE examples: exact solution A * t**beta * sin(k x) (or A * t**beta for the FODE).
GROUP_FAMILIES = ("fode", "fpde_caputo", "fpde_variable_order", "fpde_variable_order_xt")
ALL_FAMILIES = FAMILIES + WAVE_FAMILIES + GROUP_FAMILIES
SPLITS = ("evolution", "development", "sealed")
# Evaluator node counts (per axis). "legacy" reproduces evaluator v2-v4, which used the task records'
# output_space_points/output_time_points (65 x 17; FODE 65 times).
FINE_GRID = {"space": 512, "time": 500, "wave_space": 128, "wave_time": 100}
GRIDS = ("fine", "legacy")


def evaluation_points(task, grid="fine"):
    """(space nodes per axis, time nodes); space is 1 for the FODE and time is 1 for Poisson."""
    if grid not in GRIDS:
        raise ValueError(f"grid must be one of {GRIDS}")
    if grid == "legacy":
        space, time = task.output_space_points, task.output_time_points
    elif task.family == "wave2d":
        space, time = FINE_GRID["wave_space"], FINE_GRID["wave_time"]
    else:
        space, time = FINE_GRID["space"], FINE_GRID["time"]
    if task.family == "fode":
        space = 1
    if task.family == "poisson":
        time = 1
    return space, time


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    template_group: str
    split: str
    length: float
    final_time: float
    diffusivity: float
    velocity: float
    amplitude: float
    mode: int
    profile: str
    beta: float
    orders: tuple[float, ...]
    weights: tuple[float, ...]
    relative_l2_tolerance: float = 0.002
    boundary_tolerance: float = 1e-10
    initial_tolerance: float = 1e-10
    max_seconds: float = 15.0
    memory_mb: int = 1024
    output_space_points: int = 65
    output_time_points: int = 17
    source_id: str = "PDE-RSI-manufactured-v1"
    reuse_permission: str = "self-generated; upstream benchmarks not imported"

    def validate(self):
        if self.family not in ALL_FAMILIES or self.split not in SPLITS:
            raise ValueError("Unknown family or split")
        if not (self.length > 0 and self.final_time > 0 and self.diffusivity > 0):
            raise ValueError("Domain, time horizon, and diffusivity must be positive")
        if self.beta <= 0 or self.mode < 1 or self.mode != int(self.mode):
            raise ValueError("Invalid manufactured profile")
        if self.profile not in ("sine", "polynomial", "mixed_sine"):
            raise ValueError("Unknown spatial template")
        if self.amplitude <= 0 or not math.isfinite(self.amplitude):
            raise ValueError("Amplitude must be finite and positive")
        if self.family == "wave2d" and not self.beta > 1:
            raise ValueError("Wave tasks need beta > 1 for a finite initial velocity")
        if self.family in GROUP_FAMILIES:
            self.validate_group_orders()
        elif self.family in ("caputo", "multi_term"):
            if not self.orders or len(self.orders) != len(self.weights):
                raise ValueError("Fractional orders and weights are required")
            if any(not 0 < a < 1 for a in self.orders) or any(w <= 0 for w in self.weights):
                raise ValueError("The pilot supports positive sums of Caputo orders in (0,1)")
        elif self.orders or self.weights:
            raise ValueError("Local tasks must not contain fractional orders")
        if self.relative_l2_tolerance <= 0 or self.max_seconds <= 0 or self.memory_mb < 256:
            raise ValueError("Invalid evaluation limits")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, record):
        fields = cls.__dataclass_fields__
        values = {k: v for k, v in record.items() if k in fields}
        values["orders"] = tuple(values["orders"])
        values["weights"] = tuple(values["weights"])
        return cls(**values).validate()

    def validate_group_orders(self):
        if self.beta <= 1:
            raise ValueError("Group FPDE tasks need a temporal power beta > 1")
        if len(self.orders) != 1 or len(self.weights) != 1 or not 0 < self.orders[0] < 1:
            raise ValueError("Group tasks need one coefficient in (0,1): the order, or c in a variable order")
        if any(w <= 0 for w in self.weights):
            raise ValueError("Fractional weights must be positive")

    def public_record(self):
        if self.family in WAVE_FAMILIES:
            return self.wave_public_record()
        if self.family in GROUP_FAMILIES:
            return self.group_public_record()
        k = self.mode * math.pi / self.length
        temporal = f"(1+t**{self.beta})"
        if self.profile == "polynomial":
            spatial = f"4*(x/{self.length})*(1-x/{self.length})"
            gradient = f"4/{self.length}*(1-2*x/{self.length})"
            negative_laplacian = f"8/{self.length**2}"
        elif self.profile == "mixed_sine":
            spatial = f"(sin({k:.17g}*x)+0.25*sin({3*k:.17g}*x))"
            gradient = f"({k:.17g}*cos({k:.17g}*x)+{0.75*k:.17g}*cos({3*k:.17g}*x))"
            negative_laplacian = f"({k*k:.17g}*sin({k:.17g}*x)+{2.25*k*k:.17g}*sin({3*k:.17g}*x))"
        else:
            spatial = f"sin({k:.17g}*x)"
            gradient = f"{k:.17g}*cos({k:.17g}*x)"
            negative_laplacian = f"{k*k:.17g}*{spatial}"
        diffusion = f"{self.diffusivity*self.amplitude:.17g}*{temporal}*({negative_laplacian})"
        if self.family == "poisson":
            equation = "-diffusivity*u_xx = f(x)"
            forcing = f"{self.diffusivity*self.amplitude:.17g}*({negative_laplacian})"
        else:
            equation = "u_t - diffusivity*u_xx = f(x,t)"
            derivative = f"{self.amplitude*self.beta:.17g}*t**({self.beta-1:.17g})*({spatial})"
            if self.orders:
                equation = "sum_j weights[j]*Caputo(orders[j],u) - diffusivity*u_xx = f(x,t)"
                derivative = "+".join(
                    f"{w*self.amplitude*math.gamma(self.beta+1)/math.gamma(self.beta+1-a):.17g}"
                    f"*t**({self.beta-a:.17g})*({spatial})" for a, w in zip(self.orders, self.weights)
                )
            forcing = f"({derivative})+({diffusion})"
            if self.family == "convection_diffusion":
                equation = "u_t + velocity*u_x - diffusivity*u_xx = f(x,t)"
                forcing += f"+{self.velocity*self.amplitude:.17g}*{temporal}*({gradient})"
            if self.family == "burgers":
                equation = "u_t + u*u_x - diffusivity*u_xx = f(x,t)"
                forcing += f"+{self.amplitude**2:.17g}*{temporal}**2*({spatial})*({gradient})"
        return {
            **self.to_dict(),
            "equation": equation,
            "forcing_expression": forcing,
            "domain": {"space": [0, self.length], "time": [0, self.final_time]},
            "boundary_conditions": {"type": "Dirichlet", "left": 0.0, "right": 0.0},
            "initial_expression": None if self.family == "poisson" else f"{self.amplitude}*({spatial})",
            "operator": {"definition": "Caputo integral of u_t from history_start to t" if self.orders else "local",
                         "history_start": 0.0, "exterior_conditions": "not applicable to local spatial Laplacian"},
            "permitted_observations": "equation, coefficients, forcing, IC, BC, evolution/development diagnostics",
            "requested_output": "finite real u on evaluator-specified uniform x/t nodes; NPZ or internal solution object",
            "compute_limits": {"seconds": self.max_seconds, "memory_mb": self.memory_mb, "cpu_threads": 1, "gpus": 0},
        }


    def wave_public_record(self):
        """u = A*s(x)*s(y)*(cos(omega*t) + t**beta) on [0,L]^2; diffusivity is the squared wave speed."""
        k = self.mode * math.pi / self.length
        omega = wave_frequency(self)
        def profile(z):
            if self.profile == "polynomial":
                return f"4*({z}/{self.length})*(1-{z}/{self.length})", f"(-8/{self.length**2})"
            if self.profile == "mixed_sine":
                return (f"(sin({k:.17g}*{z})+0.25*sin({3*k:.17g}*{z}))",
                        f"(-{k*k:.17g}*sin({k:.17g}*{z})-{2.25*k*k:.17g}*sin({3*k:.17g}*{z}))")
            return f"sin({k:.17g}*{z})", f"(-{k*k:.17g}*sin({k:.17g}*{z}))"
        (sx, dxx), (sy, dyy) = profile("x"), profile("y")
        temporal = f"(cos({omega:.17g}*t)+t**{self.beta})"
        acceleration = f"(-{omega*omega:.17g}*cos({omega:.17g}*t)+{self.beta*(self.beta-1):.17g}*t**({self.beta-2:.17g}))"
        forcing = (f"{self.amplitude:.17g}*({acceleration}*{sx}*{sy}"
                   f"-{self.diffusivity:.17g}*{temporal}*({dxx}*{sy}+{sx}*{dyy}))")
        return {
            **self.to_dict(),
            "equation": "u_tt - diffusivity*(u_xx + u_yy) = f(x,y,t)",
            "forcing_expression": forcing,
            "domain": {"space": [[0, self.length], [0, self.length]], "time": [0, self.final_time]},
            "boundary_conditions": {"type": "Dirichlet", "value": 0.0, "edges": "all four"},
            "initial_expression": f"{self.amplitude}*{sx}*{sy}",
            "initial_velocity_expression": "0",
            "operator": {"definition": "local second-order wave operator", "wave_speed": math.sqrt(self.diffusivity)},
            "permitted_observations": "equation, coefficients, forcing, IC, BC, evolution/development diagnostics",
            "requested_output": "finite real u[t, y, x] on uniform x and y grids; NPZ or internal solution object",
            "compute_limits": {"seconds": self.max_seconds, "memory_mb": self.memory_mb, "cpu_threads": 1, "gpus": 0},
        }


    def order_expression(self):
        if self.family == "fpde_variable_order":
            return f"{self.orders[0]}*sin(t)**2"
        if self.family == "fpde_variable_order_xt":
            return f"{self.orders[0]}*sin(x)*t**2"
        return None

    def group_public_record(self):
        """Group FPDE examples. Exact solution A*t**p*sin(k x); p (beta) and A stay evaluator-only."""
        k = self.mode*math.pi/self.length
        p, amplitude = self.beta, self.amplitude

        def caputo(order):
            return f"{math.gamma(p+1)/math.gamma(p+1-order):.17g}*t**({p-order:.17g})"
        if self.family == "fode":
            equation = "Caputo(orders[0], u)(t) = f(t), u(0) = 0, t in [0, final_time]"
            forcing = f"{amplitude:.17g}*{caputo(self.orders[0])}"
        else:
            space = f"sin({k:.17g}*x)"
            diffusion = f"{self.diffusivity*k*k:.17g}*t**{p}"
            if self.family == "fpde_caputo":
                equation = "Caputo(orders[0], u) = diffusivity*u_xx + f(x,t)"
                temporal = caputo(self.orders[0])
            else:
                alpha = self.order_expression()
                variable = "alpha(t)" if self.family == "fpde_variable_order" else "alpha(x,t)"
                equation = f"Caputo({variable}, u) = diffusivity*u_xx + f(x,t), {variable} = {alpha}"
                temporal = f"{math.gamma(p+1):.17g}/gamma({p+1:.17g}-{alpha})*t**({p:.17g}-{alpha})"
            forcing = f"{amplitude:.17g}*(({temporal})+{diffusion})*{space}"
        return {
            **self.to_dict(),
            "equation": equation,
            "forcing_expression": forcing,
            "domain": ({"time": [0, self.final_time]} if self.family == "fode"
                       else {"space": [0, self.length], "time": [0, self.final_time]}),
            "boundary_conditions": (None if self.family == "fode"
                                    else {"type": "Dirichlet", "left": 0.0, "right": 0.0}),
            "initial_expression": "0",
            "operator": {"definition": ("Caputo derivative from 0; a variable order is evaluated at the "
                                        "current point (x, t)"), "history_start": 0.0,
                         "order_function": self.order_expression()},
            "permitted_observations": "equation, coefficients, forcing, IC, BC, evolution/development diagnostics",
            "requested_output": ("finite real u(t) on a time grid" if self.family == "fode"
                                 else "finite real u[t, x] on uniform x and increasing t grids"),
            "compute_limits": {"seconds": self.max_seconds, "memory_mb": self.memory_mb, "cpu_threads": 1, "gpus": 0},
        }


def wave_frequency(task):
    """Natural frequency of the fundamental sine mode on the square."""
    return math.sqrt(task.diffusivity) * math.pi * task.mode * math.sqrt(2) / task.length


def make_wave_pilot():
    """12 two-dimensional wave cases: 4 per split, one template group per split."""
    tasks = []
    for group, split in enumerate(SPLITS):
        for variant in range(4):
            speed = 0.5 + 0.25*variant
            tasks.append(Task(
                task_id=f"wave2d-g{group}-v{variant}", family="wave2d",
                template_group=f"wave2d/{('sine', 'polynomial', 'mixed_sine')[group]}", split=split,
                length=1.0 + 0.2*(variant % 2), final_time=0.5 + 0.25*(variant//2),
                diffusivity=speed*speed, velocity=0.0, amplitude=0.3 + 0.1*variant, mode=1,
                profile=("sine", "polynomial", "mixed_sine")[group], beta=(2.5, 1.5, 1.25)[group],
                orders=(), weights=(), source_id="PDE-RSI-manufactured-wave2d-v1",
            ).validate())
    validate_groups(tasks)
    return tasks


# (family, split, order coefficient, temporal power p); the paper instances are coefficient 0.5, p=3.
GROUP_VARIANTS = {
    "fode": [("evolution", (0.3,), 2.5), ("evolution", (0.7,), 3.5), ("development", (0.5,), 3.0),
             ("development", (0.9,), 4.0), ("sealed", (0.6,), 3.25)],
    "fpde_caputo": [("evolution", (0.3,), 2.5), ("evolution", (0.7,), 3.5), ("development", (0.5,), 3.0),
                    ("development", (0.9,), 4.0), ("sealed", (0.6,), 3.25)],
    "fpde_variable_order": [("evolution", (0.3,), 2.5), ("evolution", (0.7,), 3.5),
                            ("development", (0.5,), 3.0), ("development", (0.9,), 4.0),
                            ("sealed", (0.6,), 3.25)],
    "fpde_variable_order_xt": [("evolution", (0.3,), 2.5), ("evolution", (0.7,), 3.5),
                               ("development", (0.5,), 3.0), ("development", (0.9,), 4.0),
                               ("sealed", (0.6,), 3.25)],
}
GROUP_DIRECTORY = "benchmarks/group_fpde"


def make_group_pilot():
    """20 group FPDE cases (v2): per family 2 evolution, 2 development (including the paper case), 1 sealed."""
    tasks = []
    for family, variants in GROUP_VARIANTS.items():
        counters = {}
        for split, orders, power in variants:
            group = SPLITS.index(split)
            variant = counters[split] = counters.get(split, -1) + 1
            tasks.append(Task(
                task_id=f"{family}-g{group}-v{variant}", family=family, template_group=f"{family}/{split}",
                split=split, length=1.0 if family == "fode" else math.pi, final_time=1.0, diffusivity=1.0,
                velocity=0.0, amplitude=1.0, mode=1, profile="sine", beta=power, orders=orders,
                weights=(1.0,)*len(orders), output_time_points=65 if family == "fode" else 17,
                source_id="PDE-RSI-group-fpde-v2",
                reuse_permission="user-supplied group FPDE examples (t^3 sin x family); extra variants self-generated",
            ).validate())
    validate_groups(tasks)
    return tasks


def write_group_benchmark(root):
    return _write_extra_benchmark(root, make_group_pilot(), GROUP_DIRECTORY, "group_fpde_sealed.jsonl",
                                  "group-fpde-v2", GROUP_FAMILIES,
                                  "user-supplied group FPDE examples with self-generated order/power variants")


def load_group_tasks(root, split):
    return _load_extra_tasks(root, split, GROUP_DIRECTORY, GROUP_FAMILIES)


def _write_extra_benchmark(root, tasks, directory, sealed_name, version, families, status):
    from .storage import project_path
    root = project_path(root)
    manifest = {"schema_version": 1, "benchmark_version": version, "cases": len(tasks),
                "families": list(families), "files": {}, "source_status": status}
    for split in SPLITS:
        path = project_path(root / ("evaluator_private/"+sealed_name if split == "sealed"
                                    else f"{directory}/{split}.jsonl"))
        path.parent.mkdir(parents=True, exist_ok=True)
        records = "".join(json.dumps(t.public_record(), sort_keys=True) + "\n" for t in tasks if t.split == split)
        if path.exists() and path.read_text() != records:
            raise FileExistsError(f"Refusing to overwrite changed benchmark: {path}")
        path.write_text(records)
        if split == "sealed":
            path.chmod(0o600)
        manifest["files"][split] = {"path": str(path.relative_to(root)),
                                     "cases": sum(t.split == split for t in tasks),
                                     "sha256": hashlib.sha256(records.encode()).hexdigest()}
    project_path(root / directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _load_extra_tasks(root, split, directory, families):
    from .storage import project_path
    if split not in ("evolution", "development"):
        raise ValueError("Only public splits can be loaded for agent experiments")
    root = project_path(root)
    expected = json.loads((root / directory / "manifest.json").read_text())["files"][split]
    path = project_path(root / expected["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
        raise ValueError(f"Task file differs from its manifest: {path}")
    tasks = load_tasks(path)
    if len(tasks) != expected["cases"] or any(t.split != split or t.family not in families for t in tasks):
        raise ValueError(f"Task count, split, or family differs from the manifest: {path}")
    return tasks


WAVE_DIRECTORY = "benchmarks/wave2d"


def write_wave_benchmark(root):
    """Write the wave cases beside, not inside, the published 1D benchmark files."""
    from .storage import project_path
    root = project_path(root)
    tasks = make_wave_pilot()
    manifest = {"schema_version": 1, "benchmark_version": "manufactured-wave2d-v1", "cases": len(tasks),
                "families": list(WAVE_FAMILIES), "files": {},
                "source_status": "self-generated manufactured 2D wave cases; not part of the uploaded HF dataset"}
    for split in SPLITS:
        path = project_path(root / ("evaluator_private/wave2d_sealed.jsonl" if split == "sealed"
                                    else f"{WAVE_DIRECTORY}/{split}.jsonl"))
        path.parent.mkdir(parents=True, exist_ok=True)
        records = "".join(json.dumps(t.public_record(), sort_keys=True) + "\n" for t in tasks if t.split == split)
        if path.exists() and path.read_text() != records:
            raise FileExistsError(f"Refusing to overwrite changed benchmark: {path}")
        path.write_text(records)
        if split == "sealed":
            path.chmod(0o600)
        manifest["files"][split] = {"path": str(path.relative_to(root)), "cases": 4,
                                     "sha256": hashlib.sha256(records.encode()).hexdigest()}
    project_path(root / WAVE_DIRECTORY / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_wave_tasks(root, split):
    """Public wave cases, checked against their manifest. Sealed cases are not loadable here."""
    from .storage import project_path
    if split not in ("evolution", "development"):
        raise ValueError("Only public wave splits can be loaded for agent experiments")
    root = project_path(root)
    expected = json.loads((root / WAVE_DIRECTORY / "manifest.json").read_text())["files"][split]
    path = project_path(root / expected["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
        raise ValueError(f"Wave task file differs from its manifest: {split}")
    tasks = load_tasks(path)
    if len(tasks) != expected["cases"] or any(t.split != split or t.family not in WAVE_FAMILIES for t in tasks):
        raise ValueError(f"Wave task count, split, or family differs from the manifest: {split}")
    return tasks


def make_pilot():
    """72 cases, 18 non-overlapping groups, 24 cases in each split."""
    tasks = []
    for family in FAMILIES:
        for group, split in enumerate(SPLITS):
            for variant in range(4):
                fractional = family in ("caputo", "multi_term")
                orders = ((0.4 + 0.15*variant,) if family == "caputo" else
                          (0.3 + 0.05*variant, 0.7 + 0.05*variant) if family == "multi_term" else ())
                tasks.append(Task(
                    task_id=f"{family}-g{group}-v{variant}", family=family,
                    template_group=f"{family}/{('sine','polynomial','mixed_sine')[group]}", split=split,
                    length=1.0 + 0.2*(variant % 2), final_time=0.25 + 0.1*(variant//2),
                    diffusivity=0.05 + 0.05*variant, velocity=0.25 + 0.1*variant,
                    amplitude=0.3 + 0.1*variant, mode=1,
                    profile=("sine", "polynomial", "mixed_sine")[group],
                    beta=(2.0, 0.8, 0.55)[group] if fractional else (1.0, 2.0, 1.5)[group],
                    orders=orders, weights=(1.0,) if family == "caputo" else (0.4, 1.0) if family == "multi_term" else (),
                ).validate())
    validate_groups(tasks)
    return tasks


def validate_groups(tasks):
    seen_ids, groups = set(), {}
    for task in tasks:
        task.validate()
        if task.task_id in seen_ids:
            raise ValueError(f"Duplicate task ID: {task.task_id}")
        seen_ids.add(task.task_id)
        if task.template_group in groups and groups[task.template_group] != task.split:
            raise ValueError(f"Template leakage: {task.template_group}")
        groups[task.template_group] = task.split


def load_tasks(path):
    tasks = [Task.from_dict(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]
    validate_groups(tasks)
    return tasks


def write_pilot(root):
    from .storage import project_path
    root = project_path(root)
    tasks = make_pilot()
    manifest = {"schema_version": 1, "benchmark_version": "manufactured-v1", "cases": len(tasks),
                "source_status": "self-generated pilot; literature sources pending verification and permission review",
                "files": {}}
    for split in SPLITS:
        path = project_path(root / ("evaluator_private" if split == "sealed" else "benchmarks") / f"{split}.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        records = "".join(json.dumps(t.public_record(), sort_keys=True) + "\n" for t in tasks if t.split == split)
        if path.exists() and path.read_text() != records:
            raise FileExistsError(f"Refusing to overwrite changed benchmark: {path}")
        path.write_text(records)
        if split == "sealed":
            path.chmod(0o600)
        manifest["files"][split] = {"path": str(path.relative_to(root)), "cases": 24,
                                     "sha256": hashlib.sha256(records.encode()).hexdigest()}
    project_path(root / "benchmarks" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
