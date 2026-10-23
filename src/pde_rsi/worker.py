"""One trusted numerical solve in a resource-limited child process."""

import json
import math
import resource
import sys
import time

from .policy import SolverSettings
from .solvers import solve
from .tasks import Task


def main():
    request = json.load(sys.stdin)
    task = Task.from_dict(request["task"])
    # Modules are imported before the cap. BLAS thread count is fixed by the parent.
    resource.setrlimit(resource.RLIMIT_AS, (task.memory_mb*1024**2, task.memory_mb*1024**2))
    cpu_limit = max(1, math.ceil(min(task.max_seconds, request.get("cpu_limit_seconds", task.max_seconds))))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    started = time.perf_counter()
    result = solve(task, SolverSettings(**request["settings"]))
    runtime = time.perf_counter()-started
    if request.get("solution_path"):
        result.save_npz(request["solution_path"])
    json.dump({"x": result.x.tolist(), "t": result.t.tolist(), "u": result.u.tolist(),
               "y": None if result.y is None else result.y.tolist(),
               "method": result.method, "runtime_seconds": runtime,
               "peak_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}, sys.stdout)


if __name__ == "__main__":
    main()
