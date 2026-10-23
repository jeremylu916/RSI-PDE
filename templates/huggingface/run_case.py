"""Run one shipped public baseline; never executes expressions or model code."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow.parquet as pq
from pde_baseline.policy import SolverSettings
from pde_baseline.solvers import solve
from pde_baseline.tasks import Task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nx", type=int)
    parser.add_argument("--nt", type=int)
    parser.add_argument("--grading", type=float)
    parser.add_argument("--advection", choices=["central", "upwind"])
    args = parser.parse_args()
    rows = [row for row in pq.read_table(args.data).to_pylist() if row["id"] == args.id]
    if len(rows) != 1:
        parser.error("The selected Parquet file must contain exactly one matching public ID")
    row = rows[0]
    task = Task.from_dict(row["task_record"])
    values = {key: row["solver_config"][key] for key in ("nx", "nt", "grading", "advection")}
    values.update({key: getattr(args, key) for key in values if getattr(args, key) is not None})
    settings = SolverSettings(**values).validate()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    solution = solve(task, settings)
    elapsed = time.perf_counter()-started
    solution.save_npz(output/"solution.npz")
    records = {
        "task.json": task.to_dict(), "settings.json": vars(settings),
        "run.json": {"id": task.task_id, "method": solution.method, "runtime_seconds": elapsed,
                     "peak_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                     "resources_certified": False, "numerical_accuracy_evaluated": False,
                     "nx": len(solution.x)-1, "nt": len(solution.t)-1,
                     "solver_source_sha256": hashlib.sha256(
                         (Path(__file__).resolve().parent/"pde_baseline/solvers.py").read_bytes()).hexdigest()},
    }
    for name, record in records.items():
        (output/name).write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(output), **records["run.json"]}, indent=2))


if __name__ == "__main__":
    main()
