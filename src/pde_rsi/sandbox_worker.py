"""Run one agent-written solver inside the bubblewrap sandbox.

This file is bind-mounted read-only as /sandbox/child.py next to the public
equations modules (1D and 2D). It must not import pde_rsi: the project source tree, the
evaluator, references, credentials, and the network are absent from the sandbox.

The solver sees forcing and initial data only as callables. Manufactured-profile
parameters stay in closures so the program cannot read the exact solution.
"""

import builtins
import importlib.util
import io
import json
import linecache
import math
import resource
import sys
import time
import traceback
import types

import numpy as np
import scipy
import scipy.fft
import scipy.integrate
import scipy.interpolate
import scipy.linalg
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
import scipy.special
import cmath
import collections
import dataclasses
import functools
import itertools
import typing

# Mirrors codegen.ALLOWED_MODULES; all are imported before limits are applied.
ALLOWED_MODULES = {
    "math", "cmath", "functools", "itertools", "collections", "dataclasses", "typing",
    "numpy", "numpy.linalg", "numpy.fft", "numpy.polynomial",
    "scipy", "scipy.fft", "scipy.integrate", "scipy.interpolate", "scipy.linalg",
    "scipy.optimize", "scipy.sparse", "scipy.sparse.linalg", "scipy.special",
    "torch", "torch.nn", "torch.nn.functional", "torch.optim", "torch.autograd",
}
SAFE_BUILTINS = (
    "abs", "all", "any", "bool", "complex", "dict", "divmod", "enumerate", "filter", "float",
    "frozenset", "hasattr", "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max",
    "min", "next", "object", "pow", "print", "property", "range", "repr", "reversed", "round",
    "set", "slice", "sorted", "staticmethod", "classmethod", "str", "sum", "super", "tuple", "zip",
    "ArithmeticError", "AssertionError", "Exception", "FloatingPointError", "IndexError",
    "KeyError", "LookupError", "NotImplementedError", "OverflowError", "RuntimeError",
    "StopIteration", "TypeError", "ValueError", "ZeroDivisionError", "NotImplemented",
    "__build_class__",
)
FILENAME = "<agent-solver>"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GROUP_FAMILIES = ("fode", "fpde_caputo", "fpde_variable_order", "fpde_variable_order_xt")


def build_group_problem(public, task):
    equations = load_module("public_equations_group", "/sandbox/equations_group.py")

    if public["family"] == "fpde_variable_order_xt":
        def alpha(x, t):
            return np.asarray(equations.order(task, float(t), np.asarray(x, dtype=float)), dtype=float)
    else:
        def alpha(t):
            return float(equations.order(task, float(t)))

    if public["family"] == "fode":
        def forcing(t):
            return float(equations.forcing_group(task, 0.0, float(t)))

        def initial():
            return 0.0

        return types.SimpleNamespace(**public, forcing=forcing, initial=initial, alpha=alpha)

    def forcing(x, t):
        return np.asarray(equations.forcing_group(task, np.asarray(x, dtype=float), float(t)), dtype=float)

    def initial(x):
        return np.asarray(equations.initial_group(task, np.asarray(x, dtype=float)), dtype=float)

    return types.SimpleNamespace(**public, forcing=forcing, initial=initial, alpha=alpha)


def build_identification_problem(public, hidden, data):
    """Identification view: data, domain, and black-box forcing/initial data of the hidden true task."""
    task = types.SimpleNamespace(**hidden)
    if hidden["family"].startswith("fpde_"):
        equations = load_module("public_equations_group", "/sandbox/equations_group.py")
        source, start = equations.forcing_group, equations.initial_group
    else:
        equations = load_module("public_equations", "/sandbox/equations.py")
        source, start = equations.forcing, equations.initial

    def forcing(x, t):
        t = float(t)
        if not t > 0:
            raise ValueError("forcing(x, t) is defined for t > 0")
        return np.asarray(source(task, np.asarray(x, dtype=float), t), dtype=float)

    def initial(x):
        return np.asarray(start(task, np.asarray(x, dtype=float)), dtype=float)

    return types.SimpleNamespace(**public, forcing=forcing, initial=initial,
                                 data_x=np.asarray(data["x"], dtype=float), data_t=np.asarray(data["t"], dtype=float),
                                 data_u=np.asarray(data["u"], dtype=float))


def build_bench_problem(public, spec, data, layout=None):
    """pde_benchmarks.tex view: data, domain, and black-box forcing / initial / boundary data of the hidden case."""
    equations = load_module("public_equations_bench", "/sandbox/equations_bench.py")
    layout = layout or {"coordinates": public["coordinates"], "time_dependent": public["time_dependent"]}
    coordinates = tuple(layout["coordinates"])
    arrays = {name: (None if data[name] is None else np.asarray(data[name], dtype=float)) for name in ("x", "y", "t", "u")}

    def forcing(*args):
        if len(args) != len(coordinates):
            raise TypeError(f"forcing takes {len(coordinates)} arguments in the order {coordinates}")
        values = dict(zip(coordinates, (np.asarray(a, dtype=float) for a in args)))
        if "t" in values and np.any(values["t"] < 0):
            raise ValueError("forcing is defined for t >= 0")
        x, t = values.get("x", np.float64(0.0)), values.get("t", np.float64(0.0))
        shape = np.broadcast(*values.values()).shape
        return np.broadcast_to(np.asarray(equations.forcing(spec, x, t, values.get("y")), dtype=float), shape).copy()

    extras = {}
    if layout["time_dependent"]:
        if public["spatial_dimension"] == 0:
            def initial():
                return float(equations.initial(spec, 0.0))
        elif public["spatial_dimension"] == 2:
            def initial(x, y):
                x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
                return np.asarray(equations.initial(spec, x, y), dtype=float)

            def initial_velocity(x, y):
                return np.asarray(equations.initial_velocity(spec, x, y), dtype=float)
            extras["initial_velocity"] = initial_velocity
        else:
            def initial(x):
                return np.asarray(equations.initial(spec, np.asarray(x, dtype=float)), dtype=float)
        extras["initial"] = initial
    if public["spatial_dimension"] == 1:
        def boundary(t=0.0):
            return np.asarray(equations.boundary(spec, float(t)), dtype=float)
        extras["boundary"] = boundary
    return types.SimpleNamespace(**public, forcing=forcing, data_x=arrays["x"], data_y=arrays["y"],
                                 data_t=arrays["t"], data_u=arrays["u"], **extras)


def build_inventory_problem(public, spec, data):
    """Inventory view: data, domain, black-box forcing/initial/boundary/coefficient, gridded inputs, 4-D points."""
    equations = load_module("public_equations_inventory", "/sandbox/equations_inventory.py")
    spec = dict(spec)
    spec["fields"] = {name: np.asarray(values, dtype=float) for name, values in (spec.get("fields") or {}).items()}
    layout, kind = spec["layout"], spec["kind"]
    arrays = {name: (None if data.get(name) is None else np.asarray(data[name], dtype=float))
              for name in ("x", "y", "t", "points", "u")}
    coordinate_count = {"1d_periodic": 2, "1d_dirichlet": 2, "2d_steady": 2, "2d_periodic_time": 3, "4d_points": 4}[layout]

    def forcing(*coords):
        if len(coords) != coordinate_count:
            raise TypeError(f"forcing takes {coordinate_count} coordinate arrays for layout {layout}")
        coords = [np.asarray(c, dtype=float) for c in coords]
        shape = np.broadcast(*coords).shape
        return np.broadcast_to(np.asarray(equations.forcing(spec, *coords), dtype=float), shape).copy()

    extras = {}
    if layout in ("1d_periodic", "1d_dirichlet", "2d_periodic_time"):
        def initial(*space):
            space = [np.asarray(c, dtype=float) for c in space]
            return np.asarray(equations.initial(spec, *space), dtype=float)
        extras["initial"] = initial
    if layout == "1d_dirichlet":
        def boundary(t=0.0):
            return np.asarray(equations.boundary(spec, float(t)), dtype=float)
        extras["boundary"] = boundary
    if public.get("has_coefficient_field"):
        def coefficient(*space):
            space = [np.asarray(c, dtype=float) for c in space]
            return np.asarray(equations.coefficient(spec, *space), dtype=float)
        extras["coefficient"] = coefficient
    grids = {}
    a, b = spec.get("a", 0.0), spec.get("b", 1.0)
    for name, values in spec["fields"].items():
        n = values.shape[0]
        if name == "u0":
            grids[name] = {"values": values, "x": a+(b-a)*np.arange(n)/n}
        elif name == "w0":
            grids[name] = {"values": values, "x": a+(b-a)*np.arange(n)/n, "y": a+(b-a)*np.arange(n)/n}
        elif name == "a":
            grids[name] = {"values": values, "x": (np.arange(n)+0.5)/n, "y": (np.arange(n)+0.5)/n}
        elif name == "f":
            grids[name] = {"values": values, "x": np.linspace(0, 1, n), "y": np.linspace(0, 1, n)}
        elif name == "ahat":
            grids[name] = {"values": values, "q": np.linspace(0, 1, n), "z": np.linspace(-1, 1, values.shape[1])}
    if layout == "4d_points":
        extras["points"] = np.asarray(spec.pop("points"), dtype=float)
    return types.SimpleNamespace(**public, forcing=forcing, grids=grids, data_x=arrays["x"], data_y=arrays["y"],
                                 data_t=arrays["t"], data_points=arrays["points"], data_u=arrays["u"], **extras)


def build_ext_problem(public, spec, data):
    """pdebench-ext view (Problems 12-16): data and black-box forcing / initial / boundary data; vector fields keep
    their components on the last axis, single-component fields drop it."""
    equations = load_module("public_equations_ext", "/sandbox/equations_ext.py")
    layout = spec["layout"]
    coordinates = {"1d_periodic_time": ("x", "t"), "2d_periodic_time": ("x", "y", "t"), "2d_time": ("x", "y", "t"),
                   "2d_steady": ("x", "y")}[layout]

    def squeeze(values):
        return values[..., 0] if values.shape[-1] == 1 else values

    def forcing(*args):
        if len(args) != len(coordinates):
            raise TypeError(f"forcing takes {len(coordinates)} arguments in the order {coordinates}")
        values = dict(zip(coordinates, (np.asarray(a, dtype=float) for a in args)))
        if "t" in values and np.any(values["t"] < 0):
            raise ValueError("forcing is defined for t >= 0")
        shape = np.broadcast(*values.values()).shape
        out = np.asarray(equations.forcing(spec, values["x"], values.get("t", np.float64(0.0)), values.get("y")),
                         dtype=float)
        return squeeze(np.broadcast_to(out, shape+out.shape[-1:]).copy())

    extras = {}
    if layout != "2d_steady":
        def initial(*space):
            space = [np.asarray(c, dtype=float) for c in space]
            out = np.asarray(equations.initial(spec, space[0], space[1] if len(space) > 1 else None), dtype=float)
            return squeeze(np.broadcast_to(out, np.broadcast(*space).shape+out.shape[-1:]).copy())
        extras["initial"] = initial
    if spec["class"] == "navier_stokes_2d" and layout == "2d_steady":
        def boundary(x, y):
            return np.asarray(equations.boundary(spec, np.asarray(x, dtype=float), np.asarray(y, dtype=float)), dtype=float)
        extras["boundary"] = boundary
    arrays = {name: (None if data.get(name) is None else np.asarray(data[name], dtype=float)) for name in ("x", "y", "t", "u")}
    if arrays["u"] is not None and arrays["u"].ndim == 2 and arrays["u"].shape[1] == 1:
        arrays["u"] = arrays["u"][:, 0]
    return types.SimpleNamespace(**public, forcing=forcing, data_x=arrays["x"], data_y=arrays["y"],
                                 data_t=arrays["t"], data_u=arrays["u"], **extras)


def build_problem(public, hidden):
    task = types.SimpleNamespace(**public, **hidden)
    if public["family"] in GROUP_FAMILIES:
        return build_group_problem(public, task)
    if public.get("spatial_dimension") == 2:
        equations = load_module("public_equations2d", "/sandbox/equations2d.py")

        def forcing(x, y, t):
            x, y, t = np.asarray(x, dtype=float), np.asarray(y, dtype=float), float(t)
            return np.asarray(equations.forcing2d(task, x, y, t), dtype=float)

        def initial(x, y):
            return np.asarray(equations.initial2d(task, np.asarray(x, dtype=float), np.asarray(y, dtype=float)), dtype=float)

        def initial_velocity(x, y):
            return np.asarray(equations.initial_velocity2d(task, np.asarray(x, dtype=float),
                                                            np.asarray(y, dtype=float)), dtype=float)

        return types.SimpleNamespace(**public, forcing=forcing, initial=initial, initial_velocity=initial_velocity)
    equations = load_module("public_equations", "/sandbox/equations.py")
    stationary = public["family"] == "poisson"

    def forcing(x, t=0.0):
        x = np.asarray(x, dtype=float)
        t = float(t)
        if not stationary and not t > 0:
            raise ValueError("forcing(x, t) is defined for t > 0")
        return np.asarray(equations.forcing(task, x, t), dtype=float)

    def initial(x):
        return np.asarray(equations.initial(task, np.asarray(x, dtype=float)), dtype=float)

    return types.SimpleNamespace(**public, forcing=forcing, initial=initial)


# Packages whose own C code imports submodules lazily through the caller's __import__ (e.g. torch.cuda on
# first CUDA use, numpy._core._methods on some reductions); with an agent frame on top, that is this hook.
# Agent-written import statements are checked statically against ALLOWED_MODULES before execution.
INTERNAL_PACKAGES = ("numpy", "scipy", "torch")


def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name not in ALLOWED_MODULES and name.partition(".")[0] in INTERNAL_PACKAGES \
            and name.partition(".")[0] in sys.modules:
        module = importlib.import_module(name)
        return module if fromlist else sys.modules[name.partition(".")[0]]
    if level != 0 or name not in ALLOWED_MODULES:
        raise ImportError(f"Import of {name!r} is not permitted in solver programs")
    if name not in sys.modules:
        raise ImportError(f"{name!r} is not available in this protocol (PyTorch is loaded only for GPU solves)")
    module = sys.modules[name]
    if fromlist:
        return module
    return sys.modules[name.partition(".")[0]]


def as_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numbers")
    return np.ascontiguousarray(array, dtype=np.float64)


def main():
    started_process = time.perf_counter()  # max_seconds includes start-up; programs read it via elapsed_seconds()
    request = json.loads(sys.stdin.read())
    code, family = request["code"], request["problem"].get("family", "identify")
    gpu = bool(request.get("gpu"))
    if gpu:  # imported before the limits apply; one CPU thread as for every other solve
        import torch
        import torch.autograd  # noqa: F401
        import torch.nn  # noqa: F401
        import torch.nn.functional  # noqa: F401
        import torch.optim  # noqa: F401
        torch.set_num_threads(1)
        request["problem"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if request.get("mode") == "identification":
        problem = build_identification_problem(request["problem"], request.pop("hidden"), request.pop("data"))
    elif request.get("mode") == "inventory":
        problem = build_inventory_problem(request["problem"], request.pop("hidden"), request.pop("data"))
    elif request.get("mode") == "pdebench_ext":
        problem = build_ext_problem(request["problem"], request.pop("hidden"), request.pop("data"))
    elif request.get("mode") == "pdebench":
        problem = build_bench_problem(request["problem"], request.pop("hidden"), request.pop("data"),
                                      request.pop("layout", None))
    else:
        problem = build_problem(request["problem"], request.pop("hidden"))
    problem.elapsed_seconds = lambda: time.perf_counter() - started_process
    limit_values = int(request["max_output_values"])
    cpu = max(1, math.ceil(float(request["cpu_limit_seconds"])))
    del request
    output = sys.stdout.buffer
    sys.stdout = sys.stderr  # Program prints are diagnostics, never the result channel.
    filename = f"{FILENAME[:-1]}:{family}>"
    linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
    allowed = {name: getattr(builtins, name) for name in SAFE_BUILTINS}
    allowed["__import__"] = restricted_import
    namespace = {"__builtins__": allowed, "__name__": "agent_solver"}
    memory = int(problem.memory_mb) * 1024**2
    if not gpu:  # CUDA reserves far more virtual address space than it uses; GPU solves are checked by peak RSS
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    # OpenMP/BLAS need a shared-memory segment in /dev/shm (a private tmpfs). Programs still cannot open
    # files: the static policy and restricted builtins forbid it.
    resource.setrlimit(resource.RLIMIT_FSIZE, (256*1024**2, 256*1024**2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    model_json = None
    try:
        exec(compile(code, filename, "exec"), namespace)
        solve = namespace.get("solve")
        if not callable(solve):
            raise ValueError("The program must define solve(problem)")
        started = time.perf_counter()
        result = solve(problem)
        runtime = time.perf_counter() - started
        if isinstance(result, dict) and "model" in result:
            # Identification programs also report the chosen equation; numpy scalars become floats.
            model_json = json.dumps(result["model"], default=float, allow_nan=False)
            if len(model_json) > 4000:
                raise ValueError("The returned model description exceeds 4,000 characters")
        two_d = getattr(problem, "spatial_dimension", 1) == 2
        layout = getattr(problem, "layout", None)
        if layout is not None:  # inventory suite: a dict with the layout's arrays
            if not isinstance(result, dict) or "u" not in result:
                raise ValueError("solve() must return a dict with 'u' (and the layout's coordinate arrays)")
            two_d = layout.startswith("2d")
            result = {"x": result.get("x", np.zeros(1)), "y": result.get("y", np.zeros(1)),
                      "t": result.get("t", np.zeros(1)), "u": result["u"]}
        if getattr(problem, "time_dependent", True) is False and isinstance(result, dict) and "t" not in result:
            u = np.asarray(result["u"])  # steady pdebench output: u(x) becomes u[t=0, x]
            result = {"x": result["x"], "t": np.zeros(1), "u": u[None, :] if u.ndim == 1 else u}
        if getattr(problem, "spatial_dimension", 1) == 0:
            t, u = (result["t"], result["u"]) if isinstance(result, dict) else result
            u = np.asarray(u)
            if u.ndim == 2 and 1 in u.shape:
                u = u.reshape(-1)
            if u.ndim != 1:
                raise ValueError("FODE solve() must return t and a one-dimensional u(t)")
            result = {"x": np.zeros(1), "t": t, "u": u[:, None]}
        if isinstance(result, dict):
            x, y, t, u = (result[key] for key in ("x", "y", "t", "u")) if two_d else \
                (result["x"], None, result["t"], result["u"])
        elif two_d:
            x, y, t, u = result
        else:
            (x, t, u), y = result, None
        x, t, u = as_array(x, "x"), as_array(t, "t"), as_array(u, "u")
        y = as_array(y, "y") if two_d else None
        if x.size + t.size + u.size + (y.size if two_d else 0) > limit_values:
            raise ValueError(f"Returned arrays exceed {limit_values} values; return fewer output nodes")
    except BaseException as error:  # noqa: B902 - report every program failure, including SystemExit
        frames = traceback.extract_tb(error.__traceback__)
        agent_frames = traceback.StackSummary.from_list([f for f in frames if f.filename == filename])
        text = "".join(agent_frames.format()) + "".join(traceback.format_exception_only(error))
        sys.stderr.write(text[-2000:])
        sys.stderr.flush()
        raise SystemExit(3)
    buffer = io.BytesIO()
    arrays = {"x": x, "t": t, "u": u} if y is None else {"x": x, "y": y, "t": t, "u": u}
    if model_json is not None:
        arrays["model_json"] = np.asarray(model_json)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    np.savez(buffer, **arrays, runtime_seconds=np.float64(runtime),
             cpu_seconds=np.float64(usage.ru_utime + usage.ru_stime),
             peak_memory_mb=np.float64(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    output.write(buffer.getvalue())
    output.flush()


if __name__ == "__main__":
    main()
