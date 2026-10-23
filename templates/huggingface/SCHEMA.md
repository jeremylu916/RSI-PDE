# Dataset schema

All PDE configurations share the same nested Arrow/Hugging Face schema, stored in
`features.json` and in each Parquet file's metadata. Scalar fields are explicitly
typed as strings, booleans, int64, or float64. Orders/weights, axes, boundaries, and
metric names are typed lists. Nested objects are structs, rather than serialized
JSON strings. Empty lists retain their element type, and inapplicable fields are null.

## Problem identity and classification

`id` preserves the original task ID. `pde_classification.equation_type` selects the
PDE configuration. `math_type` is elliptic, parabolic, or time_fractional_diffusion;
linearity is recorded separately, so Burgers is nonlinear/parabolic. All current
spatial dimensions are one. Temporal regularity is not applicable to Poisson.

## Oracle configuration

- `pde.equation` is the equation with the original coefficient names;
  `diffusivity` in that string corresponds to the `coefficients.kappa` value.
- Coefficients use `{type, value, expr}` records. Current kappa values are constant.
  Velocity is present only for linear convection-diffusion; Burgers uses u itself.
- `forcing` and `initial_condition` use `{type, expr}`. Expressions use x and t,
  ordinary arithmetic, powers, and sin/cos. They are data, not trusted executable code.
- `time_operator` specifies derivative type, orders, weights, history start, and
  definition. Fractional weights multiply Caputo derivatives. Classical tasks use
  order/weight lists `[1.0]`; Poisson uses empty lists and type `none`.
- `space_operator` is the local second derivative, with no fractional exterior conditions.
- `domain.axes` describes the x interval. Poisson's `domain.time` is null.
- Boundaries are explicit Dirichlet records with coordinate and value.
- `reference` states manufactured analytic, evaluator-only, solution_provided=false.
  It is an access declaration, not an exported exact solution.

## Evaluation and solver configuration

`evaluation_config` declares thresholds, common output sampling, linear interpolation,
solution NPZ format, wall seconds, memory MB, CPU threads, and GPU count. Its values
are unchanged from the public source task. Poisson has one output time point.

`solver_config` describes the included baseline implementation and initial settings:
nx=32, nt=48, grading=1, advection=central. Poisson ignores nt; Burgers may increase
nt to satisfy its CFL bound. Fractional tasks use grading; classical tasks use a
uniform time grid. These are starting settings, not claims that every case passes.

## Provenance and native task record

`provenance.source_split` retains evolution/development while files are exposed as
train/validation. Source file/line/checksum identify the original public JSONL.
`template_group` is the grouping key used to prevent split overlap.
`task_record` preserves the original native fields for replay with the solver bundle.
Some native fields are unused for specific families; the normalized oracle fields
use null/empty values instead. The solver reads task_record, not arbitrary expressions.

Adding a new equation, coefficient expression, geometry, operator, or solver requires
extending and validating the actual numerical implementation. Changing metadata
alone does not add support for a new PDE.
