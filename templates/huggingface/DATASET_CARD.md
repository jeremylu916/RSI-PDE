# PDE and FPDE Problems

A collection of forward problem specifications and classical numerical solvers,
organized by PDE family. The current release contains **48 public manufactured
cases**. Each row is a problem specification, not a sampled solution array or an
agent trajectory. All six configurations use the same typed nested schema.

| Configuration | Train | Validation | Equation |
| --- | --- | --- | --- |
| `poisson` | 4 | 4 | `-kappa*u_xx = f(x)` |
| `heat` | 4 | 4 | `u_t - kappa*u_xx = f(x,t)` |
| `convection_diffusion` | 4 | 4 | `u_t + velocity*u_x - kappa*u_xx = f(x,t)` |
| `burgers` | 4 | 4 | `u_t + u*u_x - kappa*u_xx = f(x,t)` |
| `caputo` | 4 | 4 | `D_C^alpha u - kappa*u_xx = f(x,t)` |
| `multi_term` | 4 | 4 | `sum_j w_j D_C^alpha_j u - kappa*u_xx = f(x,t)` |

All cases are one-dimensional with constant positive diffusivity and zero Dirichlet
boundaries. Fractional time orders lie strictly between zero and one, with Caputo
history starting at zero and a local spatial Laplacian. The development fractional
cases include weaker initial-time regularity.

## Load a PDE family

After this directory has been uploaded to a dataset repository:

```python
from datasets import load_dataset

data = load_dataset("YOUR_NAMESPACE/YOUR_DATASET", "poisson")
row = data["train"][0]
print(row["id"])
print(row["oracle_config"]["pde"]["equation"])
print(row["problem_statement"])
```

Before upload, use the directory itself:

```python
from datasets import load_dataset

data = load_dataset("/path/to/this/dataset", "caputo")
```

The README configuration metadata associates each family with its own train and
validation Parquet files. JSONL mirrors in `records/` allow direct review and editing;
the viewer reads only the explicitly configured Parquet files. `preview.html` is an
offline table with family/split selectors, expandable JSON cells, and case details.

## Dataset columns

| Column | Contents |
| --- | --- |
| `id` | Original unique problem identifier |
| `pde_classification` | Equation family, mathematical type, linearity, dimension, temporal/fractional status, regularity |
| `oracle_config` | Equation, coefficients, forcing, operator conventions, domain, IC/BC, and reference availability |
| `evaluation_config` | Accuracy metrics/tolerances, sampling, interpolation, output format, and compute limits |
| `problem_statement` | Human-readable problem for a single-loop solving agent |
| `solver_config` | Shipped baseline method, implementation path, and initial numerical settings |
| `provenance` | Source identifier, original split/group/file/line, checksum, and reuse status |
| `task_record` | Native task parameters for the shipped numerical solver |

`oracle_config` describes the problem and evaluator contract; **it contains no exact
solution values, exact solution expression, or private evaluator source**.
See [SCHEMA.md](SCHEMA.md) for conventions and the machine-readable feature types in
[features.json](features.json).

## Numerical solvers

`solver_bundle/` includes standalone Poisson finite differences, backward Euler
for heat/transport, IMEX Burgers, and nonuniform L1 Caputo/multi-term methods.
See [solver instructions](solver_bundle/SOLVERS.md). Solvers use the problem's
specified forcing and IC/BC. No LLM or recursive improvement controller is included.

## Splits and provenance

- `train` maps to the original public **evolution** split.
- `validation` maps to the original public **development** split.
- Related template groups remain together: sine profiles are train data, polynomial
  profiles validation data. Parameter variants stay in the same group/split.
- The 24 reserved final cases, their generator, reference implementation, and final
  results are excluded. This release has no `test` split.
- IDs and original split names are retained in provenance and native task records.
- `manifest.json` records row counts and checksums for every included file.

Source: `PDE-RSI-manufactured-v1`, self-generated. The currently named literature
benchmarks and group-specific FPDE examples have not been imported. The variable
coefficient, quadrilateral, high-contrast, and no-exact-solution examples visible in
the user's layout reference are not part of this release.

## Intended use and limitations

This seed collection supports problem/solver organization, evaluator validation,
and development of a fixed-controller single-loop agent. It is small and uses
simple manufactured forcing; equation/forcing structure may reveal the solution
form. It does not measure realistic PDE diversity, unseen operator-family transfer,
or recursive self improvement. Validation queries are selection data.

Accuracy requires space-time and final-time relative L2 errors at most 0.002,
with IC/BC errors at most 1e-10. Executed benchmark solves additionally require
the declared time and memory limits. A solution file alone does not certify its
execution resources. Published benchmark imports should add source and solver
provenance and independently validated references before expanding this release.

No redistribution license has been selected by the dataset owner yet. No external
benchmark data or third-party solver implementation is copied into this package;
NumPy and SciPy remain dependencies installed separately.

## Packaging references

Family configurations and splits follow the official
[Hugging Face manual configuration](https://huggingface.co/docs/hub/datasets-manual-configuration).
The data uses typed Parquet and standard
[Hugging Face dataset features](https://huggingface.co/docs/datasets/about_dataset_features).
