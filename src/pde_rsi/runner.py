"""Solve-call, wall-time, and per-process resource accounting."""

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np

from .evaluator import evaluate, failed_result
from .solvers import Solution
from .storage import project_path


def cpu_usage():
    parent = resource.getrusage(resource.RUSAGE_SELF)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    return parent.ru_utime+parent.ru_stime, child.ru_utime+child.ru_stime


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    max_solver_calls: int
    max_wall_seconds: float
    solver_calls: int = 0
    child_wall_seconds: float = 0.0
    # CPU of sandboxed grandchildren, which RUSAGE_CHILDREN does not see.
    sandbox_cpu_seconds: float = 0.0
    started: float = 0.0
    max_cpu_seconds: float | None = None
    ledger_path: Path | None = None
    phase: str = "unspecified"
    initial_cpu: tuple = field(init=False)

    def __post_init__(self):
        if self.max_cpu_seconds is None:
            self.max_cpu_seconds = self.max_wall_seconds
        if (self.max_solver_calls < 1 or not math.isfinite(self.max_wall_seconds) or self.max_wall_seconds <= 0
                or not math.isfinite(self.max_cpu_seconds) or self.max_cpu_seconds <= 0):
            raise ValueError("Budgets must be positive")
        self.started = time.perf_counter()
        self.initial_cpu = cpu_usage()
        if self.ledger_path is not None:
            self.ledger_path = project_path(self.ledger_path)

    @property
    def remaining_seconds(self):
        return self.max_wall_seconds-(time.perf_counter()-self.started)

    @property
    def remaining_cpu_seconds(self):
        used = sum(now-start for now, start in zip(cpu_usage(), self.initial_cpu))
        return self.max_cpu_seconds-used-self.sandbox_cpu_seconds

    def reserve(self, calls):
        if (self.solver_calls+calls > self.max_solver_calls or self.remaining_seconds <= 0
                or self.remaining_cpu_seconds <= 0):
            raise BudgetExceeded("Insufficient solve-call, CPU-time, or wall-time budget")

    def report(self):
        parent, child = (now-start for now, start in zip(cpu_usage(), self.initial_cpu))
        return {"max_solver_calls": self.max_solver_calls, "solver_calls": self.solver_calls,
                "max_wall_seconds": self.max_wall_seconds, "elapsed_wall_seconds": time.perf_counter()-self.started,
                "max_cpu_seconds": self.max_cpu_seconds, "parent_cpu_seconds": parent,
                "child_cpu_seconds": child, "sandbox_cpu_seconds": self.sandbox_cpu_seconds,
                "total_cpu_seconds": parent+child+self.sandbox_cpu_seconds,
                "child_wall_seconds": self.child_wall_seconds, "cpu_threads": 1, "gpus": 0,
                "within_resource_ceilings": (self.remaining_seconds >= 0 and self.remaining_cpu_seconds >= 0
                                             and self.solver_calls <= self.max_solver_calls)}


def run_task(task, settings, budget, solution_path=None):
    budget.reserve(1)
    budget.solver_calls += 1  # Failed and timed-out solves are charged too.
    timeout = min(task.max_seconds, budget.remaining_seconds)
    started = time.perf_counter()
    env = dict(os.environ)
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        env[variable] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    request = json.dumps({"task": task.to_dict(), "settings": vars(settings),
                          "cpu_limit_seconds": min(task.max_seconds, budget.remaining_cpu_seconds),
                          "solution_path": str(project_path(solution_path)) if solution_path is not None else None})
    child_cpu_start = cpu_usage()[1]
    result = None
    child_elapsed = None
    try:
        proc = subprocess.run([sys.executable, "-m", "pde_rsi.worker"], input=request, text=True,
                              capture_output=True, timeout=max(timeout, 0.001), env=env,
                              cwd=Path(__file__).resolve().parents[2])
        elapsed = time.perf_counter()-started
        child_elapsed = elapsed
        if proc.returncode:
            result = failed_result(task, "execution_failure", proc.stderr, elapsed)
        else:
            record = json.loads(proc.stdout)
            y = None if record.get("y") is None else np.asarray(record["y"])
            solution = Solution(np.asarray(record["x"]), np.asarray(record["t"]), np.asarray(record["u"]),
                                record["method"], y)
            result = evaluate(task, solution, record["runtime_seconds"], elapsed, record["peak_memory_mb"])
    except subprocess.TimeoutExpired:
        result = failed_result(task, "timeout", "Per-task or experiment wall-time ceiling reached", time.perf_counter()-started)
    except (ValueError, KeyError, FloatingPointError) as error:
        result = failed_result(task, "invalid_output", str(error), time.perf_counter()-started)
    finally:
        budget.child_wall_seconds += child_elapsed if child_elapsed is not None else time.perf_counter()-started
    result["cpu_seconds"] = cpu_usage()[1]-child_cpu_start
    if budget.remaining_seconds < 0 or budget.remaining_cpu_seconds < 0:
        result["resources_pass"] = result["passed"] = False
        result["budget_exceeded"] = True
    if budget.ledger_path is not None:
        with budget.ledger_path.open("a") as stream:
            json.dump({"call": budget.solver_calls, "phase": budget.phase,
                       "task_id": task.task_id, "family": task.family, "split": task.split,
                       "settings": vars(settings), "status": result["status"],
                       "passed": result["passed"], "relative_l2": result["relative_l2"],
                       "wall_seconds": result["wall_seconds"], "cpu_seconds": result["cpu_seconds"],
                       "peak_memory_mb": result["peak_memory_mb"]}, stream, allow_nan=False)
            stream.write("\n")
    return result


def run_suite(tasks, version, budget):
    budget.reserve(len(tasks))
    return [run_task(task, version.solver[task.family], budget) for task in tasks]
