"""Static policy, sandboxed execution, and metering for agent-written solver programs.

A solver program is one Python module per PDE family that defines ``solve(problem)``.
Programs run in a bubblewrap sandbox without network, project files, credentials,
evaluator code, or references. Forcing and initial data are black-box callables.
"""

import ast
import hashlib
import io
import json
import platform
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np

from .evaluator import evaluate, failed_result
from .runner import cpu_usage
from .solvers import Solution
from .storage import ENV_PREFIX, project_path
from .tasks import FAMILIES, GROUP_FAMILIES, WAVE_FAMILIES, evaluation_points

MAX_CODE_CHARS = 30000
MAX_OUTPUT_VALUES = 4_000_000
SANDBOX_VERSION = "bwrap-v9-ext"  # v5: private /dev/shm and a 256 MB file-size cap; v6: pdebench equations;
# v7: inventory inputs; v8: problem.elapsed_seconds() (wall seconds since the solve process started);
# v9: pdebench-ext equations (Problems 12-16)
ERROR_METRICS = ("relative_l2", "final_relative_l2", "absolute_l2", "final_absolute_l2", "absolute_linf",
                 "initial_error", "boundary_error")
CLASSICAL_DIRECTORY = Path(__file__).resolve().parent/"classical_programs"
CHILD_SCRIPT = Path(__file__).resolve().parent/"sandbox_worker.py"
EQUATIONS_SCRIPT = Path(__file__).resolve().parent/"equations.py"
EQUATIONS2D_SCRIPT = Path(__file__).resolve().parent/"equations2d.py"
EQUATIONS_GROUP_SCRIPT = Path(__file__).resolve().parent/"equations_group.py"
EQUATIONS_BENCH_SCRIPT = Path(__file__).resolve().parent/"equations_bench.py"
EQUATIONS_INVENTORY_SCRIPT = Path(__file__).resolve().parent/"equations_inventory.py"
EQUATIONS_EXT_SCRIPT = Path(__file__).resolve().parent/"equations_ext.py"
WORK_DIRECTORY = Path(__file__).resolve().parents[2]/".runtime/tmp"

ALLOWED_MODULES = frozenset({
    "math", "cmath", "functools", "itertools", "collections", "dataclasses", "typing",
    "numpy", "numpy.linalg", "numpy.fft", "numpy.polynomial",
    "scipy", "scipy.fft", "scipy.integrate", "scipy.interpolate", "scipy.linalg",
    "scipy.optimize", "scipy.sparse", "scipy.sparse.linalg", "scipy.special",
})
# Importable only in GPU (PINN-capable) solves; the sandbox imports them before applying limits.
TORCH_MODULES = frozenset({"torch", "torch.nn", "torch.nn.functional", "torch.optim", "torch.autograd"})
ALLOWED_MODULES = ALLOWED_MODULES | TORCH_MODULES
FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "open", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "input", "breakpoint", "help", "exit", "quit", "memoryview", "dir", "type",
})
# File I/O, foreign-function, and frame/code introspection attributes.
FORBIDDEN_ATTRIBUTES = frozenset({
    "load", "save", "savez", "savez_compressed", "loadtxt", "savetxt", "genfromtxt", "fromfile",
    "tofile", "memmap", "fromregex", "dump", "dumps", "ctypeslib", "ctypes", "f2py", "distutils",
    "testing", "DataSource", "show_config", "show_runtime",
    # PyTorch entry points that load or compile code, use files, or spawn processes.
    # ("library" is not listed: problem.library is the identification equation library; loading compiled
    # operator libraries goes through load_library, which is forbidden.)
    "hub", "jit", "compile", "cpp_extension", "load_library", "classes", "multiprocessing",
    "distributed", "onnx", "export", "package", "from_file",
    "f_back", "f_locals", "f_globals", "f_builtins", "f_code", "gi_frame", "gi_code", "cr_frame",
    "cr_code", "ag_frame", "ag_code", "tb_frame", "tb_next", "co_code", "co_consts",
})
# Problem fields visible to programs and to the model. The manufactured-profile
# fields (profile, amplitude, mode, beta) would reveal the exact solution.
PUBLIC_PROBLEM_FIELDS = (
    "family", "length", "final_time", "diffusivity", "velocity", "orders", "weights",
    "relative_l2_tolerance", "boundary_tolerance", "initial_tolerance", "max_seconds",
    "memory_mb", "output_space_points", "output_time_points",
)
HIDDEN_PROBLEM_FIELDS = ("profile", "amplitude", "mode", "beta")
EQUATIONS = {
    "poisson": "-diffusivity*u_xx = f(x)",
    "heat": "u_t - diffusivity*u_xx = f(x,t)",
    "convection_diffusion": "u_t + velocity*u_x - diffusivity*u_xx = f(x,t)",
    "burgers": "u_t + u*u_x - diffusivity*u_xx = f(x,t)",
    "caputo": "sum_j weights[j]*Caputo(orders[j],u) - diffusivity*u_xx = f(x,t)",
    "multi_term": "sum_j weights[j]*Caputo(orders[j],u) - diffusivity*u_xx = f(x,t)",
    "wave2d": "u_tt - diffusivity*(u_xx + u_yy) = f(x,y,t) on [0,length]^2",
    "fode": "Caputo(orders[0], u)(t) = f(t), u(0) = 0",
    "fpde_caputo": "Caputo(orders[0], u) = diffusivity*u_xx + f(x,t)",
    "fpde_variable_order": "Caputo(alpha(t), u) = diffusivity*u_xx + f(x,t), alpha(t) = orders[0]*sin(t)**2",
    "fpde_variable_order_xt": "Caputo(alpha(x,t), u) = diffusivity*u_xx + f(x,t), alpha(x,t) = orders[0]*sin(x)*t**2",
}


class ProgramRejected(ValueError):
    pass


STUB_PROGRAM = (
    '''"""No solver has been written for this family yet."""


def solve(problem):
    raise NotImplementedError("No solver program has been written for this family yet")
'''
)
STARTS = ("classical", "scratch")
# Runs before the rename recorded the classical start as "seed".
LEGACY_STARTS = {"seed": "classical"}


def classical_program(families=FAMILIES):
    return {family: (CLASSICAL_DIRECTORY/f"{family}.py").read_text() for family in families}


def starting_program(start, families=FAMILIES):
    """Classical: working classical solvers with tunable constants. Scratch: stubs the agent must replace."""
    start = LEGACY_STARTS.get(start, start)
    if start == "classical":
        return classical_program(families)
    if start == "scratch":
        return {family: STUB_PROGRAM for family in families}
    raise ValueError(f"Unknown starting program: {start}")


def code_sha256(code):
    return hashlib.sha256(code.encode()).hexdigest()


def program_digest(program):
    return {family: {"sha256": code_sha256(code), "chars": len(code)} for family, code in program.items()}


def public_problem(task):
    if task.family in GROUP_FAMILIES:
        return group_public_problem(task)
    record = {name: getattr(task, name) for name in PUBLIC_PROBLEM_FIELDS}
    record["output_space_points"], record["output_time_points"] = evaluation_points(task)
    record["orders"], record["weights"] = list(task.orders), list(task.weights)
    record["equation"] = EQUATIONS[task.family]
    record["left_boundary"] = record["right_boundary"] = 0.0
    if task.family in WAVE_FAMILIES:
        record["spatial_dimension"] = 2
        record["wave_speed"] = task.diffusivity**0.5
        record["bottom_boundary"] = record["top_boundary"] = 0.0
    return record


def group_public_problem(task):
    """Public numbers for the group FPDE examples; irrelevant 1D fields are omitted."""
    names = ("family", "final_time", "relative_l2_tolerance", "initial_tolerance", "max_seconds", "memory_mb",
             "output_time_points")
    if task.family != "fode":
        names += ("length", "diffusivity", "boundary_tolerance", "output_space_points")
    record = {name: getattr(task, name) for name in names}
    space, time = evaluation_points(task)
    record["output_time_points"] = time
    if task.family != "fode":
        record["output_space_points"] = space
    record.update(orders=list(task.orders), weights=list(task.weights), equation=EQUATIONS[task.family],
                  spatial_dimension=0 if task.family == "fode" else 1)
    if task.family != "fode":
        record["left_boundary"] = record["right_boundary"] = 0.0
    if task.order_expression() is not None:
        record["order_function"] = task.order_expression()
    return record


def agent_problem_record(task):
    """Public task view for the model: identifiers and coefficients, no closed-form data."""
    return {"task_id": task.task_id, "split": task.split, **public_problem(task)}


def check_program(code, family=None, require_solve=True):
    """Reject programs outside the solver contract before any execution. require_solve=False checks a modular
    part (harness v2) that must not be the module defining solve()."""
    label = f"{family} program" if family else "program"
    if not isinstance(code, str) or not code.strip():
        raise ProgramRejected(f"{label} must be nonempty Python source")
    if len(code) > MAX_CODE_CHARS:
        raise ProgramRejected(f"{label} exceeds {MAX_CODE_CHARS} characters")
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise ProgramRejected(f"{label} syntax error at line {error.lineno}: {error.msg}") from None
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_MODULES:
                    raise ProgramRejected(f"{label} line {line}: import {alias.name!r} is not permitted")
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ALLOWED_MODULES:
                raise ProgramRejected(f"{label} line {line}: import from {node.module!r} is not permitted")
            for alias in node.names:
                if alias.name == "*" or alias.name in FORBIDDEN_ATTRIBUTES or alias.name.startswith("_"):
                    raise ProgramRejected(f"{label} line {line}: importing {alias.name!r} is not permitted")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                raise ProgramRejected(f"{label} line {line}: name {node.id!r} is not permitted")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRIBUTES:
                raise ProgramRejected(f"{label} line {line}: attribute {node.attr!r} is not permitted")
        elif isinstance(node, (ast.Global, ast.Nonlocal)) and any(n.startswith("__") for n in node.names):
            raise ProgramRejected(f"{label} line {line}: dunder names are not permitted")
    solvers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "solve"]
    if not require_solve:
        return True
    if len(solvers) != 1 or not solvers[0].args.args:
        raise ProgramRejected(f"{label} must define exactly one top-level solve(problem)")
    return True


GPU_ENV_CONFIG = Path(__file__).resolve().parents[2]/"configs/gpu_env.json"
GPU_ENV_MOUNT = "/opt/gpu-env"


def gpu_env_prefix():
    """Interpreter environment with CUDA PyTorch used for GPU solves (configs/gpu_env.json, or PDE_RSI_GPU_ENV)."""
    prefix = Path(os.environ.get("PDE_RSI_GPU_ENV") or json.loads(GPU_ENV_CONFIG.read_text())["prefix"])
    if not (prefix/"bin/python").exists():
        raise RuntimeError(f"GPU interpreter not found: {prefix}/bin/python")
    return prefix


def sandbox_command(gpu=False):
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError("bubblewrap (bwrap) is required to execute agent-written programs")
    prefix = str(ENV_PREFIX.resolve())
    command = [bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
               "--clearenv", "--ro-bind", "/usr", "/usr",
               "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/lib", "/lib", "--symlink", "usr/bin", "/bin",
               "--ro-bind", prefix, prefix,
               "--ro-bind", str(CHILD_SCRIPT), "/sandbox/child.py",
               "--ro-bind", str(EQUATIONS_SCRIPT), "/sandbox/equations.py",
               "--ro-bind", str(EQUATIONS2D_SCRIPT), "/sandbox/equations2d.py",
               "--ro-bind", str(EQUATIONS_GROUP_SCRIPT), "/sandbox/equations_group.py",
               "--ro-bind", str(EQUATIONS_BENCH_SCRIPT), "/sandbox/equations_bench.py",
               "--ro-bind", str(EQUATIONS_INVENTORY_SCRIPT), "/sandbox/equations_inventory.py",
               "--ro-bind", str(EQUATIONS_EXT_SCRIPT), "/sandbox/equations_ext.py",
               "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/dev/shm", "--tmpfs", "/tmp", "--chdir", "/tmp",
               "--setenv", "HOME", "/tmp", "--setenv", "PYTHONDONTWRITEBYTECODE", "1"]
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                     "BLIS_NUM_THREADS"):
        command += ["--setenv", variable, "1"]
    interpreter = f"{prefix}/bin/python"
    if gpu:
        # A separate CUDA PyTorch environment, mounted read-only away from its home-directory path.
        command += ["--ro-bind", str(gpu_env_prefix().resolve()), GPU_ENV_MOUNT]
        interpreter = f"{GPU_ENV_MOUNT}/bin/python"
        # NVIDIA device nodes and sysfs for device discovery; the driver library is under /usr.
        for device in sorted(Path("/dev").glob("nvidia*")):
            if device.is_char_device():
                command += ["--dev-bind", str(device), str(device)]
        command += ["--ro-bind", "/sys", "/sys"]
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            command += ["--setenv", "CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"]]
    return command + [interpreter, "-I", "/sandbox/child.py"]


def gpu_sandbox_self_test():
    """Fail closed unless GPU solves see CUDA PyTorch but still no project files, home, or network."""
    probe = ("import os, json, socket, torch, numpy, scipy\n"
             "paths = {p: os.path.exists(p) for p in (%r, %r, %r)}\n"
             "try:\n    socket.create_connection(('1.1.1.1', 443), timeout=2); network = True\n"
             "except OSError:\n    network = False\n"
             "print(json.dumps({'visible_paths': paths, 'network': network, 'torch': torch.__version__,"
             " 'cuda': torch.cuda.is_available(), 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available()"
             " else None, 'numpy': numpy.__version__, 'scipy': scipy.__version__}))\n") % (
                 str(project_path(Path(__file__).resolve().parents[2]/".runtime/secrets")),
                 str(Path(__file__).resolve().parent), os.path.expanduser("~"))
    command = sandbox_command(gpu=True)
    command = command[:-2] + ["-I", "-c", probe]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if proc.returncode:
        raise RuntimeError("GPU sandbox self-test failed to start: " + proc.stderr[-500:])
    report = json.loads(proc.stdout)
    report["isolated"] = not report["network"] and not any(report["visible_paths"].values())
    report["gpu_env"] = str(gpu_env_prefix())
    if not report["isolated"] or not report["cuda"]:
        raise RuntimeError(f"GPU sandbox is not isolated or has no CUDA: {report}")
    return report


def sandbox_self_test():
    """Fail closed unless the sandbox hides project files and has no network."""
    probe = ("import os, socket, json\n"
             "paths = {p: os.path.exists(p) for p in (%r, %r, %r)}\n"
             "try:\n    socket.create_connection(('1.1.1.1', 443), timeout=2); network = True\n"
             "except OSError:\n    network = False\n"
             "print(json.dumps({'visible_paths': paths, 'network': network}))\n") % (
                 str(project_path(Path(__file__).resolve().parents[2]/".runtime/secrets")),
                 str(Path(__file__).resolve().parent), os.path.expanduser("~"))
    command = sandbox_command()
    command = command[:-2] + ["-I", "-c", probe]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if proc.returncode:
        raise RuntimeError("Sandbox self-test failed to start: " + proc.stderr[-500:])
    report = json.loads(proc.stdout)
    report["isolated"] = not report["network"] and not any(report["visible_paths"].values())
    report["sandbox_version"] = SANDBOX_VERSION
    if not report["isolated"]:
        raise RuntimeError(f"Sandbox is not isolated: {report}")
    return report


def _failure_message(proc):
    stderr = proc.stderr.decode("utf-8", "replace")
    if proc.returncode < 0:
        try:
            name = signal.Signals(-proc.returncode).name
        except ValueError:
            name = str(-proc.returncode)
        hint = " (CPU-time limit)" if name == "SIGXCPU" else " (likely memory limit)" if name == "SIGKILL" else ""
        return f"Program terminated by {name}{hint}. {stderr[-800:]}"
    return stderr[-1500:] or f"Program exited with status {proc.returncode}"


def run_program_task(task, code, budget, solution_path=None, request_overrides=None, postprocess=None,
                     evaluator=None):
    """One metered sandboxed solve. Failed and timed-out solves are charged too.

    request_overrides replaces request fields (the identification protocol sends its own problem view,
    the full hidden task, and data). postprocess(result, extras) receives extra NPZ arrays on success.
    evaluator replaces evaluator.evaluate (the pdebench protocol scores against its own exact solutions).
    """
    budget.reserve(1)
    budget.solver_calls += 1
    timeout = min(task.max_seconds, budget.remaining_seconds)
    request = {"code": code, "problem": public_problem(task) if request_overrides is None or
               "problem" not in request_overrides else None,
               "hidden": None if "hidden" in (request_overrides or {}) else
               {name: getattr(task, name) for name in HIDDEN_PROBLEM_FIELDS},
               "cpu_limit_seconds": min(task.max_seconds, budget.remaining_cpu_seconds),
               "max_output_values": MAX_OUTPUT_VALUES}
    request.update(request_overrides or {})
    gpu = bool(request.get("gpu"))
    request = json.dumps(request).encode()
    extras = {}
    sha = code_sha256(code)
    sandbox_cpu = None
    started = time.perf_counter()
    child_cpu_start = cpu_usage()[1]
    try:
        proc = subprocess.run(sandbox_command(gpu), input=request, capture_output=True,
                              timeout=max(timeout, 0.001), cwd=project_path(WORK_DIRECTORY))
        elapsed = time.perf_counter()-started
        if proc.returncode:
            result = failed_result(task, "execution_failure", _failure_message(proc), elapsed)
        else:
            with np.load(io.BytesIO(proc.stdout), allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in ("x", "t", "u", "runtime_seconds", "peak_memory_mb")}
                arrays["y"] = archive["y"] if "y" in archive.files else None
                sandbox_cpu = float(archive["cpu_seconds"])
                extras = {name: archive[name] for name in archive.files
                          if name not in ("x", "y", "t", "u", "runtime_seconds", "peak_memory_mb", "cpu_seconds")}
            solution = Solution(arrays["x"], arrays["t"], arrays["u"], f"agent program {sha[:12]}", arrays["y"])
            result = (evaluator or evaluate)(task, solution, float(arrays["runtime_seconds"]), elapsed,
                                             float(arrays["peak_memory_mb"]))
            if not all(math.isfinite(result[key]) for key in ERROR_METRICS):
                # Finite but enormous values can overflow the squared-error integrals.
                raise ValueError("Solution magnitude overflows the L2 error integrals; the scheme is unstable")
            residual = result["physical_residual_relative"]
            if residual is not None and not math.isfinite(residual):
                result["physical_residual_relative"] = None
            # The program's own prints (stderr); only the coding-agent evaluation service releases them.
            result["program_output"] = proc.stderr.decode("utf-8", "replace")[-2000:]
            if postprocess is not None:
                result = postprocess(result, extras)
            if solution_path is not None:
                solution.save_npz(solution_path)
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter()-started
        result = failed_result(task, "timeout", "Per-task or experiment wall-time ceiling reached", elapsed)
    except (ValueError, KeyError, OSError, FloatingPointError) as error:
        elapsed = time.perf_counter()-started
        result = failed_result(task, "invalid_output", str(error), elapsed)
    budget.child_wall_seconds += elapsed
    # The sandboxed process is a grandchild in its own PID namespace, so RUSAGE_CHILDREN misses it.
    # It reports its own CPU on success; failed or killed solves are charged their wall time.
    sandbox_cpu = elapsed if sandbox_cpu is None else sandbox_cpu
    budget.sandbox_cpu_seconds += sandbox_cpu
    result["cpu_seconds"] = cpu_usage()[1]-child_cpu_start+sandbox_cpu
    result["sandbox_cpu_seconds"] = sandbox_cpu
    result["program_sha256"] = sha
    if budget.remaining_seconds < 0 or budget.remaining_cpu_seconds < 0:
        result["resources_pass"] = result["passed"] = False
        result["budget_exceeded"] = True
    if budget.ledger_path is not None:
        with budget.ledger_path.open("a") as stream:
            json.dump({"call": budget.solver_calls, "phase": budget.phase, "task_id": task.task_id,
                       "family": task.family, "split": task.split, "program_sha256": sha,
                       "status": result["status"], "passed": result["passed"],
                       "relative_l2": result["relative_l2"], "wall_seconds": result["wall_seconds"],
                       "cpu_seconds": result["cpu_seconds"], "peak_memory_mb": result["peak_memory_mb"],
                       "solve_seconds": result.get("runtime_seconds"), "host": platform.node(),
                       "cores": sorted(os.sched_getaffinity(0)), "load_1min": os.getloadavg()[0]},
                      stream, allow_nan=False)
            stream.write("\n")
    return result
