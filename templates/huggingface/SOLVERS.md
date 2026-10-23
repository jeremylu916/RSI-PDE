# Portable numerical solvers

The `pde_baseline` package contains only public numerical methods, task/settings
schemas, forcing/IC functions, and NPZ I/O. It has no task generator, analytic
reference module, evaluator-private data, RSI policies, model API, or dependency on
the original PDE-RSI project directory.

From the dataset directory, with NumPy/SciPy/PyArrow installed:

```bash
python solver_bundle/run_case.py \
  --data data/heat/train.parquet --id heat-g0-v0 \
  --output /your/authorized/output/folder
```

Specify `--nx`, `--nt`, `--grading`, or `--advection` to override initial settings.
The command fixes BLAS/OpenMP libraries to one thread and writes `solution.npz`,
`task.json`, `settings.json`, and `run.json`. Existing output directories are never
overwritten. `run.json` reports solve runtime and peak memory but does not certify
an independently enforced benchmark budget or numerical accuracy. The owning
project's independent evaluator can check the public task and solution afterward.

| Family | Baseline method |
| --- | --- |
| Poisson | Centered finite differences and tridiagonal direct solve |
| Heat | Backward Euler and centered diffusion |
| Convection-diffusion | Backward Euler and centered or upwind transport |
| Burgers | Implicit diffusion, explicit nonlinear transport, and CFL checks |
| Caputo / multi-term | Nonuniform L1 history quadrature and graded meshes |

History quadrature costs O(Nt^2 Nx). Current methods require a uniform spatial
interval mesh, constant positive diffusivity, homogeneous Dirichlet boundaries,
and fractional orders in (0,1). The source task's family, profiles, parameters,
operator conventions, and IC/BC are fixed; this is not an arbitrary-expression solver.

Inside the originating project, source `scripts/env.sh` and use its dedicated
conda interpreter. Put every output and temporary file under that depot project.
