"""PDE identification protocol: choose the equation that fits sparse noisy data, then solve it.

Each case hides a true public task (family and coefficients). The agent receives a library of
equation classes, 256 noisy samples of the exact solution, the forcing and initial data as black
boxes, and the domain. Its program returns the solution on a grid of its choice plus the chosen
model {"family": class, "parameters": {...}}. Scoring is the unchanged evaluator on the true task;
class accuracy and parameter errors are reported separately and never enter the gate.
"""

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np


# Candidate equation classes shown to the agent. Every class uses u(0,t) = u(L,t) = 0.
LIBRARY = {
    "heat": {"equation": "u_t - D*u_xx = f(x,t)", "parameters": {"D": "diffusivity > 0"}},
    "convection_diffusion": {"equation": "u_t + v*u_x - D*u_xx = f(x,t)",
                             "parameters": {"D": "diffusivity > 0", "v": "velocity"}},
    "burgers": {"equation": "u_t + u*u_x - D*u_xx = f(x,t)", "parameters": {"D": "viscosity > 0"}},
    "caputo": {"equation": "Caputo(alpha, u) - D*u_xx = f(x,t)",
               "parameters": {"D": "diffusivity > 0", "alpha": "order in (0,1)"}},
    "multi_term": {"equation": "w1*Caputo(alpha1, u) + Caputo(alpha2, u) - D*u_xx = f(x,t)",
                   "parameters": {"D": "diffusivity > 0", "alpha1": "order in (0,1)", "alpha2": "order in (0,1)",
                                  "w1": "weight > 0 of the first term"}},
    "variable_order_t": {"equation": "Caputo(alpha(t), u) - D*u_xx = f(x,t), alpha(t) = c*sin(t)**2",
                         "parameters": {"D": "diffusivity > 0", "c": "coefficient in (0,1)"}},
    "variable_order_xt": {"equation": "Caputo(alpha(x,t), u) - D*u_xx = f(x,t), alpha(x,t) = c*sin(x)*t**2",
                          "parameters": {"D": "diffusivity > 0", "c": "coefficient in (0,1)"}},
}
TRUE_CLASS = {"heat": "heat", "convection_diffusion": "convection_diffusion", "burgers": "burgers",
              "caputo": "caputo", "fpde_caputo": "caputo", "multi_term": "multi_term",
              "fpde_variable_order": "variable_order_t", "fpde_variable_order_xt": "variable_order_xt"}
SOURCES = ("heat", "convection_diffusion", "burgers", "caputo", "multi_term",
           "fpde_caputo", "fpde_variable_order", "fpde_variable_order_xt")
CASES_PER_FAMILY = 2
N_POINTS = 256
NOISE_FRACTION = 0.01
MAX_SECONDS = 30.0  # identification includes fitting, so it gets more time than a single solve
# v2 samples every case's domain and coefficients independently. v1 reused the benchmark variants,
# whose coefficients were fixed functions of L and identical across splits, so training labels
# revealed validation coefficients.
PROTOCOL_VERSION = "identification-v2"


def true_parameters(task):
    family = task.family
    if family in ("heat", "burgers"):
        return {"D": task.diffusivity}
    if family == "convection_diffusion":
        return {"D": task.diffusivity, "v": task.velocity}
    if family in ("caputo", "fpde_caputo"):
        return {"D": task.diffusivity, "alpha": task.orders[0]}
    if family == "multi_term":
        return {"D": task.diffusivity, "alpha1": task.orders[0], "alpha2": task.orders[1],
                "w1": task.weights[0]/task.weights[1]}
    return {"D": task.diffusivity, "c": task.orders[0]}


@dataclass(frozen=True)
class IdentificationCase:
    case_id: str
    truth: object  # the hidden Task, with max_seconds raised to MAX_SECONDS

    # The harness groups results and budgets by these, exactly as for ordinary tasks.
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
        return int(hashlib.sha256(f"{PROTOCOL_VERSION}:{self.truth.task_id}".encode()).hexdigest()[:8], 16)

    def data(self):
        """Deterministic noisy samples of the exact solution at scattered interior points."""
        from .reference import reference, reference_group
        rng = np.random.default_rng(self.seed)
        x = rng.uniform(0, self.truth.length, N_POINTS)
        t = rng.uniform(0, self.truth.final_time, N_POINTS)
        exact = reference_group if self.truth.family.startswith("fpde_") else reference
        clean = np.array([exact(self.truth, np.array([xi]), np.array([ti]))[0, 0] for xi, ti in zip(x, t)])
        noise_std = NOISE_FRACTION*float(np.sqrt(np.mean(clean**2)))
        return {"x": x, "t": t, "u": clean+noise_std*rng.standard_normal(N_POINTS), "noise_std": noise_std}

    def public_view(self):
        """Everything the model and the program may know about this case; no family or coefficients."""
        from .tasks import evaluation_points
        space, time = evaluation_points(self.truth)
        return {"case_id": self.case_id, "split": self.split, "length": self.truth.length,
                "final_time": self.truth.final_time, "left_boundary": 0.0, "right_boundary": 0.0,
                "n_data": N_POINTS, "noise_std": self.data()["noise_std"],
                "relative_l2_tolerance": self.truth.relative_l2_tolerance,
                "boundary_tolerance": self.truth.boundary_tolerance,
                "initial_tolerance": self.truth.initial_tolerance, "max_seconds": self.truth.max_seconds,
                "memory_mb": self.truth.memory_mb, "output_space_points": space, "output_time_points": time}


def sample_truth(family, split, index, max_seconds=MAX_SECONDS, memory_mb=1024):
    """A hidden true task with domain and coefficients drawn for this case alone."""
    from .tasks import Task
    rng = np.random.default_rng(int(hashlib.sha256(f"{PROTOCOL_VERSION}:{family}:{split}:{index}".encode())
                                    .hexdigest()[:12], 16))
    group = family.startswith("fpde_")
    length = rng.uniform(0.8, 3.1)  # <= pi keeps the space-time variable order c*sin(x)*t^2 non-negative
    final_time = rng.uniform(0.25, 1.0)
    diffusivity = float(np.exp(rng.uniform(np.log(0.03), np.log(1.0))))
    velocity = float(rng.choice([-1, 1])*rng.uniform(0.2, 1.0)) if family == "convection_diffusion" else 0.0
    if family in ("caputo", "fpde_caputo"):
        orders, weights = (float(rng.uniform(0.3, 0.9)),), (1.0,)
    elif family == "multi_term":
        orders, weights = (float(rng.uniform(0.2, 0.5)), float(rng.uniform(0.6, 0.9))), (float(rng.uniform(0.2, 0.8)), 1.0)
    elif family in ("fpde_variable_order", "fpde_variable_order_xt"):
        orders, weights = (float(rng.uniform(0.3, 0.9)),), (1.0,)
    else:
        orders, weights = (), ()
    if group:  # u = A t^p sin(pi x / L), zero initial data
        profile, beta, amplitude = "sine", float(rng.uniform(2.5, 4.0)), float(rng.uniform(0.5, 1.5))
    else:      # u = A s(x) (1 + t^beta); the profile alternates with the split as in the benchmark
        profile = "sine" if split == "evolution" else "polynomial"
        fractional = family in ("caputo", "multi_term")
        beta = float(rng.uniform(1.5, 2.5) if fractional else rng.uniform(1.0, 2.0))
        amplitude = float(rng.uniform(0.3, 0.6))
    return Task(task_id=f"ident-{family}-{split}-{index}", family=family, template_group=f"ident/{family}/{split}",
                split=split, length=float(length), final_time=float(final_time), diffusivity=diffusivity,
                velocity=velocity, amplitude=amplitude, mode=1, profile=profile, beta=beta, orders=orders,
                weights=weights, max_seconds=max_seconds, memory_mb=memory_mb, output_time_points=17,
                source_id=f"PDE-RSI-{PROTOCOL_VERSION}",
                reuse_permission="self-generated identification truths").validate()


def make_cases(root, split, per_family=CASES_PER_FAMILY, max_seconds=MAX_SECONDS, memory_mb=1024):
    """Cases per public split (per_family per true family), ordered by a hash so IDs do not reveal families."""
    if split not in ("evolution", "development"):
        raise ValueError("Only public identification splits are generated here")
    truths = [sample_truth(family, split, index, max_seconds, memory_mb)
              for family in SOURCES for index in range(per_family)]
    truths.sort(key=lambda t: hashlib.sha256(f"{PROTOCOL_VERSION}:{t.task_id}".encode()).hexdigest())
    prefix = {"evolution": "t", "development": "v"}[split]
    return [IdentificationCase(f"case-{prefix}{index:02d}", task) for index, task in enumerate(truths)]


def score_model(case, model):
    """Class accuracy and relative parameter errors of a returned model against the hidden truth."""
    truth_class = TRUE_CLASS[case.family]
    family = model.get("family") if isinstance(model, dict) else None
    parameters = model.get("parameters") if isinstance(model, dict) else None
    parameters = parameters if isinstance(parameters, dict) else {}
    correct = family == truth_class
    errors = {}
    if correct:
        for name, value in true_parameters(case.truth).items():
            estimate = parameters.get(name)
            if isinstance(estimate, (int, float)) and math.isfinite(estimate):
                errors[name] = abs(estimate-value)/max(abs(value), 1e-12)
            else:
                errors[name] = None
    return {"identified_family": family if isinstance(family, str) else None,
            "identified_parameters": {k: v for k, v in parameters.items()
                                      if isinstance(v, (int, float)) and math.isfinite(v)},
            "true_family": truth_class, "family_correct": correct, "parameter_relative_errors": errors,
            "method": model.get("method") if isinstance(model, dict) and isinstance(model.get("method"), str) else None}


def data_digest(cases):
    return {case.case_id: hashlib.sha256(json.dumps({k: (v.tolist() if hasattr(v, "tolist") else v)
                                                      for k, v in case.data().items()}).encode()).hexdigest()
            for case in cases}


# ---------------------------------------------------------------- protocol for the improvement loop
from pathlib import Path  # noqa: E402

from .code_agents import CodeProtocol  # noqa: E402
from .codegen import ALLOWED_MODULES, MAX_CODE_CHARS, MAX_OUTPUT_VALUES, STUB_PROGRAM, run_program_task  # noqa: E402
from .evaluator import failed_result  # noqa: E402
from .runner import BudgetExceeded  # noqa: E402
from .storage import project_path  # noqa: E402

PROGRAM_KEY = "identify"
CLASSICAL_PROGRAM = Path(__file__).resolve().parent/"classical_programs/identify.py"
CONTRACT = f"""Task: each case comes from an UNKNOWN equation in the library below with UNKNOWN coefficients. Identify
the equation class and its coefficients from noisy data, then return an accurate solution of that equation
over the whole domain. One program (family "{PROGRAM_KEY}") defines solve(problem) and is used for every case.

problem attributes:
- case_id, split, length L, final_time T; u(0,t) = u(L,t) = 0 for every case.
- data_x, data_t, data_u: n_data noisy samples u(x_i, t_i) + noise, scattered over (0, L) x (0, T).
  The noise is Gaussian with standard deviation noise_std (also given).
- forcing(x, t): the source term f of the TRUE equation at an array x and scalar t > 0 (black box).
- initial(x): u(x, 0) (black box).
- library: dict {{class name: {{"equation": ..., "parameters": {{name: meaning}}}}}} of candidate equations.
- relative_l2_tolerance, boundary_tolerance, initial_tolerance, max_seconds, memory_mb,
  output_space_points, output_time_points.
Caputo(alpha, u)(t) = 1/Gamma(1-alpha) * integral_0^t (t-s)^(-alpha) u_s(s) ds (history from t = 0). A
variable order alpha(t) or alpha(x, t) is evaluated at the current point inside the integral.

solve(problem) returns {{"x": x, "t": t, "u": u, "model": {{"family": <library class name>,
"parameters": {{name: value, ...}}}}}}: x uniform from 0 to L, t strictly increasing from 0 to T (at least 3
points), u finite with shape (len(t), len(x)), u[0] = initial(x) and u[:, 0] = u[:, -1] = 0 within 1e-10.

Scoring: u is compared with the TRUE solution after linear interpolation onto output_space_points uniform x
nodes and output_time_points uniform t nodes (trapezoidal space-time and final-time relative L2). A case
passes when both are at most relative_l2_tolerance, IC/BC hold, and the solve fits max_seconds (wall and CPU,
including about 0.5 s start-up) and memory_mb. The gate uses only these errors. The reported model is scored
separately (class accuracy and relative parameter errors) and does not affect acceptance.
Feedback for training (evolution) cases includes the true class and parameters; validation feedback does not.

Allowed imports: {", ".join(sorted(ALLOWED_MODULES))}. Forbidden: file, network, or process access; eval, exec,
compile, open, getattr, setattr, type, dir; names or attributes beginning with "_"; frame or code
introspection. Programs run single-threaded without a GPU in an isolated sandbox, return at most
{MAX_OUTPUT_VALUES:,} values, and are at most {MAX_CODE_CHARS:,} characters. Write genuine fitting and numerical
solution code; never hard-code answers or branch on case_id."""
DIAGNOSTIC_FIELDS = ("status", "relative_l2", "final_relative_l2", "absolute_l2", "final_absolute_l2",
                     "initial_error", "boundary_error", "runtime_seconds", "wall_seconds", "peak_memory_mb",
                     "accuracy_pass", "constraints_pass", "resources_pass", "passed", "nx", "nt", "message",
                     "identified_family", "identified_parameters", "method")


def run_case(case, code, budget, solution_path=None, gpu=False):
    data = case.data()

    def postprocess(result, extras):
        model = None
        if "model_json" in extras:
            try:
                model = json.loads(str(extras["model_json"]))
            except ValueError:
                model = None
        return {**result, **score_model(case, model), "case_id": case.case_id}

    result = run_program_task(
        case.truth, code, budget, solution_path,
        request_overrides={"mode": "identification", "gpu": gpu, "problem": {**case.public_view(), "library": LIBRARY},
                           "hidden": case.truth.to_dict(),
                           "data": {key: data[key].tolist() for key in ("x", "t", "u")}},
        postprocess=postprocess)
    if "case_id" not in result:  # failed solves carry no model
        result.update(score_model(case, None), case_id=case.case_id)
    return result


class IdentificationProtocol(CodeProtocol):
    name = "identification"
    gpu = False

    def __init__(self):
        self.cases = {}

    def load_tasks(self, root, package, families):
        tasks = {split: make_cases(root, split) for split in ("evolution", "development")}
        self.cases = {case.task_id: case for items in tasks.values() for case in items}
        return tasks

    def starting_program(self, start, families):
        start = {"seed": "classical"}.get(start, start)
        if start == "classical":
            return {PROGRAM_KEY: CLASSICAL_PROGRAM.read_text()}
        if start == "scratch":
            return {PROGRAM_KEY: STUB_PROGRAM}
        raise ValueError(f"Unknown starting program: {start}")

    def problem_records(self, tasks):
        return [case.public_view() for case in tasks]

    def contract(self, families):
        return CONTRACT+"\nlibrary = "+json.dumps(LIBRARY)

    def evaluate(self, tasks, version, budget, solution_directory=None):
        results = []
        if solution_directory is not None:
            project_path(solution_directory).mkdir(parents=True, exist_ok=True)
        for index, case in enumerate(tasks):
            self.cases.setdefault(case.task_id, case)
            try:
                path = solution_directory/f"{case.case_id}.npz" if solution_directory is not None else None
                results.append(run_case(case, version.program[PROGRAM_KEY], budget, path, self.gpu))
            except BudgetExceeded:
                for rest in tasks[index:]:
                    row = failed_result(rest.truth, "budget_exhausted", "Case not executed; allowance exhausted", 0)
                    results.append({**row, **score_model(rest, None), "case_id": rest.case_id})
                break
        return results

    def diagnostics(self, results):
        rows = []
        for result in results:
            row = {"case_id": result["case_id"], "split": result["split"],
                   **{key: result.get(key) for key in DIAGNOSTIC_FIELDS}}
            if result["split"] == "evolution":  # labeled training data
                case = self.cases[result["task_id"]]
                row.update(true_family=result["true_family"], true_parameters=true_parameters(case.truth),
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
        cases = [case for items in tasks.values() for case in items]
        return {"protocol_version": PROTOCOL_VERSION, "library": LIBRARY, "n_points": N_POINTS,
                "noise_fraction": NOISE_FRACTION, "max_seconds": MAX_SECONDS, "data_sha256": data_digest(cases),
                "cases": {case.case_id: case.task_id for case in cases}}


# ---------------------------------------------------------------- student protocol: numerical or PINN, from scratch
STUDENT_VERSION = "student-v1"
STUDENT_MAX_SECONDS = 90.0
STUDENT_MEMORY_MB = 16384
STUDENT_SELF_TEST_CASES = 3
METHODS_CONTRACT = """Solver methods. Like a student, choose for each case how to solve the equation you identified:
- numerical: finite differences, finite volumes, spectral methods, or L1/L2-1sigma-type schemes for Caputo
  terms, written with NumPy/SciPy;
- pinn: a physics-informed neural network u_theta(x, t) trained with PyTorch on the GPU (problem.device is
  "cuda" when a GPU is available) to minimize the PDE residual at collocation points plus the data misfit and
  initial/boundary penalties. Caputo terms need a quadrature of the history integral (e.g. an L1 rule in t).
  Evaluate the trained network on a grid and return arrays.
You may also combine them (e.g. fit coefficients numerically, then train a PINN). Report the method used as
model["method"] = "numerical" or "pinn". forcing(x, t) and initial(x) take NumPy arrays; convert to tensors
yourself. Allowed PyTorch modules: torch, torch.nn, torch.nn.functional, torch.optim, torch.autograd (no hub,
jit, compile, saving or loading). One shared GPU and one CPU thread; every solve, including PyTorch import
(about 3 s) and training, must fit max_seconds of wall and CPU time. memory_mb bounds peak host memory."""


class StudentProtocol(IdentificationProtocol):
    """Identify the equation from data and solve it by a self-chosen method (numerical or PINN), from scratch."""
    name = "student"
    gpu = True

    def load_tasks(self, root, package, families):
        tasks = {split: make_cases(root, split, per_family=1, max_seconds=STUDENT_MAX_SECONDS,
                                   memory_mb=STUDENT_MEMORY_MB) for split in ("evolution", "development")}
        self.cases = {case.task_id: case for items in tasks.values() for case in items}
        return tasks

    def starting_program(self, start, families):
        if start != "scratch":
            raise ValueError("The student protocol starts from scratch only")
        return {PROGRAM_KEY: STUB_PROGRAM}

    def self_test_tasks(self, evolution):
        return evolution[:STUDENT_SELF_TEST_CASES]

    def contract(self, families):
        return CONTRACT+"\n"+METHODS_CONTRACT+"\nlibrary = "+json.dumps(LIBRARY)

    def metadata(self, tasks):
        from .codegen import gpu_sandbox_self_test
        return {**super().metadata(tasks), "gpu_sandbox": gpu_sandbox_self_test(),
                "student_version": STUDENT_VERSION, "max_seconds": STUDENT_MAX_SECONDS,
                "memory_mb": STUDENT_MEMORY_MB, "self_test_cases": STUDENT_SELF_TEST_CASES,
                "methods": ["numerical", "pinn"]}
