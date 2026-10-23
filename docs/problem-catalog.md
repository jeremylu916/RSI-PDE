# Problem and numerical solver catalog

Current scope: collect problems and numerical solvers, validate them, then build a
single-loop agent. RSI work is on hold. This catalog describes the existing local
pilot; it does not claim that the named literature benchmarks have been imported.

## Where the equations are stored

- `src/pde_rsi/tasks.py:72`: human-readable equation strings and task specifications.
- `src/pde_rsi/equations.py:23`: executable forcing functions; `initial` defines ICs.
- `src/pde_rsi/solvers.py:64`: numerical solve dispatcher and family-specific methods.
- `benchmarks/evolution.jsonl` and `benchmarks/development.jsonl`: complete public
  instances, one JSON object per line. Each includes `equation`, `forcing_expression`,
  `domain`, coefficients, `initial_expression`, `boundary_conditions`, operator
  convention, accuracy tolerances, and compute limits.
- `src/pde_rsi/reference.py:6`: evaluator-only analytic references. Do not give this
  module or reference solutions to a solving agent.

## Implemented families

All current cases use one spatial coordinate x in [0,L], homogeneous Dirichlet
boundaries u(0,t)=u(L,t)=0, and a constant positive diffusivity kappa.
Time-dependent cases run from t=0 to T. Fractional history starts at zero;
the spatial operator is a local Laplacian. Poisson has no initial condition.

| Family | Equation | Definition | Solver |
| --- | --- | --- | --- |
| Poisson | `-kappa u_xx = f(x)` | [tasks.py:89](../src/pde_rsi/tasks.py#L89) | [solvers.py:72](../src/pde_rsi/solvers.py#L72) — Centered second-order finite differences; tridiagonal direct solve |
| Heat | `u_t - kappa u_xx = f(x,t)` | [tasks.py:92](../src/pde_rsi/tasks.py#L92) | [solvers.py:101](../src/pde_rsi/solvers.py#L101) — Backward Euler in time; centered diffusion; tridiagonal solve |
| Convection-diffusion | `u_t + v u_x - kappa u_xx = f(x,t)` | [tasks.py:102](../src/pde_rsi/tasks.py#L102) | [solvers.py:105](../src/pde_rsi/solvers.py#L105) — Backward Euler; centered or upwind transport; tridiagonal solve |
| Viscous Burgers | `u_t + u u_x - kappa u_xx = f(x,t)` | [tasks.py:105](../src/pde_rsi/tasks.py#L105) | [solvers.py:115](../src/pde_rsi/solvers.py#L115) — Implicit diffusion and explicit nonlinear transport; CFL-limited stepping |
| Caputo subdiffusion | `D_t^alpha u - kappa u_xx = f(x,t), 0 < alpha < 1` | [tasks.py:95](../src/pde_rsi/tasks.py#L95) | [solvers.py:94](../src/pde_rsi/solvers.py#L94) — Nonuniform L1 history quadrature; graded time mesh; centered diffusion |
| Multi-term Caputo diffusion | `sum_j w_j D_t^alpha_j u - kappa u_xx = f(x,t), 0 < alpha_j < 1` | [tasks.py:95](../src/pde_rsi/tasks.py#L95) | [solvers.py:94](../src/pde_rsi/solvers.py#L94) — Weighted sums of nonuniform L1 history quadrature; centered diffusion |

## 2D wave family (added October 14, 2026)

The 2D wave family is stored separately so the published 1D benchmark and the
Hugging Face package keep their hashes:

| Item | Value |
| --- | --- |
| Equation | `u_tt - D (u_xx + u_yy) = f(x,y,t)` on `[0,L]^2`; `D` is the squared wave speed, stored in the `diffusivity` field |
| Boundary and initial data | `u = 0` on all four edges; `u(x,y,0)` given; `u_t(x,y,0) = 0` |
| Exact solution | `u = A s(x) s(y) (cos(omega t) + t^beta)`, where `omega = sqrt(D) pi sqrt(2) / L` is the frequency of the fundamental mode |
| Profiles and beta | Train (`g0`): sine, beta = 2.5. Validation (`g1`): quadratic, beta = 1.5, so `u_tt ~ t^-1/2` is singular at t = 0. Reserved (`g2`): mixed sine, beta = 1.25 |
| Variants v0–v3 | L = 1.0, 1.2, 1.0, 1.2; T = 0.5, 0.5, 0.75, 0.75; wave speed = 0.5, 0.75, 1.0, 1.25; amplitude = 0.3–0.6 |
| Evaluator | 65 x 65 spatial nodes x 17 times, trilinear interpolation, trapezoidal space-time L2. Pass criteria: relative L2 <= 0.002 (combined and final time) and IC/BC <= 1e-10 |
| Files | `benchmarks/wave2d/{evolution,development}.jsonl` plus a manifest; reserved cases in `evaluator_private/wave2d_sealed.jsonl` |
| Code | Generator `tasks.make_wave_pilot`; forcing `equations2d.py`; reference `reference.reference2d`; evaluator `evaluator.evaluate2d`; trusted solver `solvers.solve_wave2d` (explicit leapfrog, 5-point Laplacian, CFL number 0.5) |

The classical leapfrog (`NX = 32`) passes 3 of 4 training cases (relative L2
1.2e-3 to 2.2e-3). It fails all 4 validation cases at about 2e-2, because its
first step samples the singular source at a single point. The error in this
version converges only at about order 0.5. Integrating the source over each of
the first few time steps (hat-weighted, with a substitution that removes the
t^-1/2 singularity) reaches about 1e-5 in under 0.1 s. The validation cases
therefore reward correct treatment of the singular start, not only finer grids.

## Group FPDE examples (group-fpde-v2, October 15, 2026)

These four families come from the user's group examples. Every one has the exact solution
`u = t^p sin(x)` on `x in [0, pi]`, or `u = t^p` for the FODE, with zero initial and boundary data.
The paper instances (p = 3, order coefficient 0.5) are in the validation split alongside
self-generated variants. The variants change the order and the hidden power p, so a program
that hard-codes `t^3 sin(x)` fails them.

| Family | Equation | Paper instance |
| --- | --- | --- |
| `fode` | `D_t^alpha u(t) = f(t)`, `u(0) = 0`, t in [0, 1] | alpha = 0.5, u = t^3 |
| `fpde_caputo` | `D_t^alpha u = u_xx + f` | alpha = 0.5, u = t^3 sin x |
| `fpde_variable_order` | `D_t^{alpha(t)} u = u_xx + f`, alpha(t) = c sin^2 t | c = 0.5 |
| `fpde_variable_order_xt` | `D_t^{alpha(x,t)} u = u_xx + f`, alpha(x,t) = c sin(x) t^2 | c = 0.5 |

| Split | Order or coefficient c, and power p |
| --- | --- |
| Train (`g0`) | 0.3 with p = 2.5; 0.7 with p = 3.5 |
| Validation (`g1`) | 0.5 with p = 3 (paper); 0.9 with p = 4 |
| Reserved (`g2`) | 0.6 with p = 3.25 |

The forcing is `f = Gamma(p+1)/Gamma(p+1-alpha) t^(p-alpha) sin x + t^p sin x`, with alpha evaluated
at the current point. This is exact for the variable-order Caputo derivative whose order is frozen
at `(x, t)`: `1/Gamma(1-alpha(x,t)) int_0^t (t-s)^(-alpha(x,t)) u_s(x,s) ds`.
`tests/test_group_fpde.py` checks every forcing against Caputo derivatives computed by quadrature
with the singular kernel.

**Version history.** Version group-fpde-v1 had a two-term problem
(`D^alpha u + D^beta u = u_xx + f`, beta in (1, 2)) in place of `fpde_variable_order_xt`. The supplied
equation showed only `D^alpha`, while its forcing also contained a beta term. At the user's
request, v2 replaces it with the space-time variable-order problem. The v1 files are archived in
`benchmarks/group_fpde/archive-v1/` and `evaluator_private/group_fpde_sealed-v1.jsonl`. Their
manifest paths refer to the pre-archive location. Ten partial v1 agent runs were stopped and kept as
`runs/code-agents-group-*-20261015-aborted-groupv1/`.

Files: `benchmarks/group_fpde/{evolution,development}.jsonl` with their manifest, and
`evaluator_private/group_fpde_sealed.jsonl`. Code: generator `tasks.make_group_pilot`, forcing
`equations_group.py`, reference `reference.reference_group`, trusted solvers `solvers.solve_group`
(L1 with the order at the current step and grid point), and the evaluator path in
`evaluator.evaluate` (`manufactured-evaluator-v4-group`). The classical starting programs (32 x 48
grid) pass 7 of the 16 public cases.

## Public instance inventory

There are 48 public instances: four variants per family in each of two splits.
Evolution profiles are sine functions; development profiles are polynomials.
Beta specifies the manufactured temporal profile 1+t^beta for time-dependent
cases; it is unused for Poisson. The remaining 24 final cases stay evaluator-only
and are deliberately excluded from this catalog and from agent inputs.

| ID | Split | L | T | kappa | v (transport only) | Fractional orders | Weights | beta | Task file line |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| poisson-g0-v0 | evolution | 1 | — | 0.05 | — | — | — | — | [benchmarks/evolution.jsonl:1](../benchmarks/evolution.jsonl#L1) |
| poisson-g0-v1 | evolution | 1.2 | — | 0.1 | — | — | — | — | [benchmarks/evolution.jsonl:2](../benchmarks/evolution.jsonl#L2) |
| poisson-g0-v2 | evolution | 1 | — | 0.15 | — | — | — | — | [benchmarks/evolution.jsonl:3](../benchmarks/evolution.jsonl#L3) |
| poisson-g0-v3 | evolution | 1.2 | — | 0.2 | — | — | — | — | [benchmarks/evolution.jsonl:4](../benchmarks/evolution.jsonl#L4) |
| heat-g0-v0 | evolution | 1 | 0.25 | 0.05 | — | — | — | 1 | [benchmarks/evolution.jsonl:5](../benchmarks/evolution.jsonl#L5) |
| heat-g0-v1 | evolution | 1.2 | 0.25 | 0.1 | — | — | — | 1 | [benchmarks/evolution.jsonl:6](../benchmarks/evolution.jsonl#L6) |
| heat-g0-v2 | evolution | 1 | 0.35 | 0.15 | — | — | — | 1 | [benchmarks/evolution.jsonl:7](../benchmarks/evolution.jsonl#L7) |
| heat-g0-v3 | evolution | 1.2 | 0.35 | 0.2 | — | — | — | 1 | [benchmarks/evolution.jsonl:8](../benchmarks/evolution.jsonl#L8) |
| convection_diffusion-g0-v0 | evolution | 1 | 0.25 | 0.05 | 0.25 | — | — | 1 | [benchmarks/evolution.jsonl:9](../benchmarks/evolution.jsonl#L9) |
| convection_diffusion-g0-v1 | evolution | 1.2 | 0.25 | 0.1 | 0.35 | — | — | 1 | [benchmarks/evolution.jsonl:10](../benchmarks/evolution.jsonl#L10) |
| convection_diffusion-g0-v2 | evolution | 1 | 0.35 | 0.15 | 0.45 | — | — | 1 | [benchmarks/evolution.jsonl:11](../benchmarks/evolution.jsonl#L11) |
| convection_diffusion-g0-v3 | evolution | 1.2 | 0.35 | 0.2 | 0.55 | — | — | 1 | [benchmarks/evolution.jsonl:12](../benchmarks/evolution.jsonl#L12) |
| burgers-g0-v0 | evolution | 1 | 0.25 | 0.05 | — | — | — | 1 | [benchmarks/evolution.jsonl:13](../benchmarks/evolution.jsonl#L13) |
| burgers-g0-v1 | evolution | 1.2 | 0.25 | 0.1 | — | — | — | 1 | [benchmarks/evolution.jsonl:14](../benchmarks/evolution.jsonl#L14) |
| burgers-g0-v2 | evolution | 1 | 0.35 | 0.15 | — | — | — | 1 | [benchmarks/evolution.jsonl:15](../benchmarks/evolution.jsonl#L15) |
| burgers-g0-v3 | evolution | 1.2 | 0.35 | 0.2 | — | — | — | 1 | [benchmarks/evolution.jsonl:16](../benchmarks/evolution.jsonl#L16) |
| caputo-g0-v0 | evolution | 1 | 0.25 | 0.05 | — | 0.4 | 1 | 2 | [benchmarks/evolution.jsonl:17](../benchmarks/evolution.jsonl#L17) |
| caputo-g0-v1 | evolution | 1.2 | 0.25 | 0.1 | — | 0.55 | 1 | 2 | [benchmarks/evolution.jsonl:18](../benchmarks/evolution.jsonl#L18) |
| caputo-g0-v2 | evolution | 1 | 0.35 | 0.15 | — | 0.7 | 1 | 2 | [benchmarks/evolution.jsonl:19](../benchmarks/evolution.jsonl#L19) |
| caputo-g0-v3 | evolution | 1.2 | 0.35 | 0.2 | — | 0.85 | 1 | 2 | [benchmarks/evolution.jsonl:20](../benchmarks/evolution.jsonl#L20) |
| multi_term-g0-v0 | evolution | 1 | 0.25 | 0.05 | — | 0.3, 0.7 | 0.4, 1 | 2 | [benchmarks/evolution.jsonl:21](../benchmarks/evolution.jsonl#L21) |
| multi_term-g0-v1 | evolution | 1.2 | 0.25 | 0.1 | — | 0.35, 0.75 | 0.4, 1 | 2 | [benchmarks/evolution.jsonl:22](../benchmarks/evolution.jsonl#L22) |
| multi_term-g0-v2 | evolution | 1 | 0.35 | 0.15 | — | 0.4, 0.8 | 0.4, 1 | 2 | [benchmarks/evolution.jsonl:23](../benchmarks/evolution.jsonl#L23) |
| multi_term-g0-v3 | evolution | 1.2 | 0.35 | 0.2 | — | 0.45, 0.85 | 0.4, 1 | 2 | [benchmarks/evolution.jsonl:24](../benchmarks/evolution.jsonl#L24) |
| poisson-g1-v0 | development | 1 | — | 0.05 | — | — | — | — | [benchmarks/development.jsonl:1](../benchmarks/development.jsonl#L1) |
| poisson-g1-v1 | development | 1.2 | — | 0.1 | — | — | — | — | [benchmarks/development.jsonl:2](../benchmarks/development.jsonl#L2) |
| poisson-g1-v2 | development | 1 | — | 0.15 | — | — | — | — | [benchmarks/development.jsonl:3](../benchmarks/development.jsonl#L3) |
| poisson-g1-v3 | development | 1.2 | — | 0.2 | — | — | — | — | [benchmarks/development.jsonl:4](../benchmarks/development.jsonl#L4) |
| heat-g1-v0 | development | 1 | 0.25 | 0.05 | — | — | — | 2 | [benchmarks/development.jsonl:5](../benchmarks/development.jsonl#L5) |
| heat-g1-v1 | development | 1.2 | 0.25 | 0.1 | — | — | — | 2 | [benchmarks/development.jsonl:6](../benchmarks/development.jsonl#L6) |
| heat-g1-v2 | development | 1 | 0.35 | 0.15 | — | — | — | 2 | [benchmarks/development.jsonl:7](../benchmarks/development.jsonl#L7) |
| heat-g1-v3 | development | 1.2 | 0.35 | 0.2 | — | — | — | 2 | [benchmarks/development.jsonl:8](../benchmarks/development.jsonl#L8) |
| convection_diffusion-g1-v0 | development | 1 | 0.25 | 0.05 | 0.25 | — | — | 2 | [benchmarks/development.jsonl:9](../benchmarks/development.jsonl#L9) |
| convection_diffusion-g1-v1 | development | 1.2 | 0.25 | 0.1 | 0.35 | — | — | 2 | [benchmarks/development.jsonl:10](../benchmarks/development.jsonl#L10) |
| convection_diffusion-g1-v2 | development | 1 | 0.35 | 0.15 | 0.45 | — | — | 2 | [benchmarks/development.jsonl:11](../benchmarks/development.jsonl#L11) |
| convection_diffusion-g1-v3 | development | 1.2 | 0.35 | 0.2 | 0.55 | — | — | 2 | [benchmarks/development.jsonl:12](../benchmarks/development.jsonl#L12) |
| burgers-g1-v0 | development | 1 | 0.25 | 0.05 | — | — | — | 2 | [benchmarks/development.jsonl:13](../benchmarks/development.jsonl#L13) |
| burgers-g1-v1 | development | 1.2 | 0.25 | 0.1 | — | — | — | 2 | [benchmarks/development.jsonl:14](../benchmarks/development.jsonl#L14) |
| burgers-g1-v2 | development | 1 | 0.35 | 0.15 | — | — | — | 2 | [benchmarks/development.jsonl:15](../benchmarks/development.jsonl#L15) |
| burgers-g1-v3 | development | 1.2 | 0.35 | 0.2 | — | — | — | 2 | [benchmarks/development.jsonl:16](../benchmarks/development.jsonl#L16) |
| caputo-g1-v0 | development | 1 | 0.25 | 0.05 | — | 0.4 | 1 | 0.8 | [benchmarks/development.jsonl:17](../benchmarks/development.jsonl#L17) |
| caputo-g1-v1 | development | 1.2 | 0.25 | 0.1 | — | 0.55 | 1 | 0.8 | [benchmarks/development.jsonl:18](../benchmarks/development.jsonl#L18) |
| caputo-g1-v2 | development | 1 | 0.35 | 0.15 | — | 0.7 | 1 | 0.8 | [benchmarks/development.jsonl:19](../benchmarks/development.jsonl#L19) |
| caputo-g1-v3 | development | 1.2 | 0.35 | 0.2 | — | 0.85 | 1 | 0.8 | [benchmarks/development.jsonl:20](../benchmarks/development.jsonl#L20) |
| multi_term-g1-v0 | development | 1 | 0.25 | 0.05 | — | 0.3, 0.7 | 0.4, 1 | 0.8 | [benchmarks/development.jsonl:21](../benchmarks/development.jsonl#L21) |
| multi_term-g1-v1 | development | 1.2 | 0.25 | 0.1 | — | 0.35, 0.75 | 0.4, 1 | 0.8 | [benchmarks/development.jsonl:22](../benchmarks/development.jsonl#L22) |
| multi_term-g1-v2 | development | 1 | 0.35 | 0.15 | — | 0.4, 0.8 | 0.4, 1 | 0.8 | [benchmarks/development.jsonl:23](../benchmarks/development.jsonl#L23) |
| multi_term-g1-v3 | development | 1.2 | 0.35 | 0.2 | — | 0.45, 0.85 | 0.4, 1 | 0.8 | [benchmarks/development.jsonl:24](../benchmarks/development.jsonl#L24) |

## Collection queue

| Source | Present status | Next collection work |
| --- | --- | --- |
| Manufactured pilot | Equations, 48 public records, classical solver code, and analytic references present | Record convergence and resource results per problem/solver pairing |
| CFDLLMBench | Not imported | Verify primary source and reuse terms; inventory Python problem specifications and solver dependencies |
| PDEAgent-Bench | Not imported | Collect equation/task definitions and staged execution/accuracy/efficiency evaluation requirements |
| PINNacle benchmark | Not imported | Inventory PDE specifications, reference solutions, numerical baselines, and applicable neural methods |
| PINNACLE sampling method | Not imported | Treat as a training/sampling method; do not count it as the comprehensive PINNacle benchmark |
| Group FPDE examples | Not imported | Use user-designated source files; preserve operator conventions, IC/BC, and original numerical solvers |
| FEABench | Optional later track | Inventory requirements separately before COMSOL-dependent implementation |

The current task schema and forcing code support the manufactured pilot only.
General benchmark equations require explicit adapters; they must not be converted
silently into the manufactured forcing or the currently supported operators.
Diffusion-wave, fractional spatial operators, inverse problems, nonzero/general
boundaries, and higher-dimensional domains are not implemented yet.

## Single-loop agent target

Use a fixed prompt/controller to read one public problem, choose or write a solver,
execute it, obtain permitted evaluator diagnostics, and diagnose/retry within the
same session and resource budget. Return the final solution, solver source/settings,
metrics, and attempt/resource trace. Keep model and controller fixed.
The current `solve` command already supplies trusted single-solve execution and
NPZ output. A model-backed single-loop controller has not been connected.
Agent runtime tools use supplied project files without web or neighboring-project
search. Benchmark curation and numerical verification occur before agent episodes.

All files, environments, package/model caches, temporary artifacts, and outputs
remain under `/depot/lin491/data/Binghang_project/PDE-RSI`.
