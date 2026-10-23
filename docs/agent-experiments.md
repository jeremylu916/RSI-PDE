# Model-guided single-session and RSI experiments

This document covers the settings-selection protocol. In the newer protocol,
the model writes the solver code itself; see [code-writing agents](code-agents.md).

The implemented agents ask a fixed foundation model to propose numerical settings
for existing classical solvers. The model receives public equations and released
numerical diagnostics; the harness executes solvers, calculates L2 errors,
validates changes, and preserves their lineage. The RSI arm can also update the
instruction that guides the next improvement round and inherit a bounded memory
lesson. Foundation-model weights remain unchanged.

The executable protocol is in [`agent_runs.py`](../src/pde_rsi/agent_runs.py), the
HTTP adapter in [`model_backend.py`](../src/pde_rsi/model_backend.py), and the
independent numerical checks in [`evaluator.py`](../src/pde_rsi/evaluator.py).

## Problems and numerical solvers

The input is the local upload-ready Hugging Face package at
`datasets/huggingface/pde-problems`. Every native record is checked against its
matching local public benchmark record before evaluation. The experiment uses
48 one-dimensional manufactured forward problems: 24 train/evolution and 24
validation/development instances, with four instances per family in each split.
Reserved final cases are excluded.

| Family | Existing solver used by the agents |
| --- | --- |
| Poisson | Centered second-order finite differences and a tridiagonal solve |
| Heat | Backward Euler with centered spatial diffusion |
| Convection–diffusion | Backward Euler with centered or upwind transport |
| Viscous Burgers | Implicit diffusion and explicit nonlinear transport, with a CFL safeguard |
| Caputo subdiffusion | Nonuniform L1 history quadrature on a graded time mesh |
| Multi-term Caputo diffusion | Weighted sums of nonuniform L1 fractional derivatives |

Equations, coefficients, forcing, initial/boundary conditions, orders, and
provenance are indexed in the [problem catalog](problem-catalog.md).
These are self-generated pilot cases; published upstream benchmarks have not
been imported. Models choose settings for these solver implementations. Arbitrary
solver-code generation, MATLAB, and PINN/FPINN training are future extensions.

## Model configuration and request contract

Use either [`gpt_oss.json`](../configs/agent_models/gpt_oss.json) or
[`claude.json`](../configs/agent_models/claude.json):

| Configuration | Requested model route |
| --- | --- |
| `gpt_oss.json` | `openai.gpt-oss-120b-1:0` |
| `claude.json` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` |

Both use native Bedrock Converse in `us-east-1`, temperature 0, and a maximum of
4,000 output tokens. Credentials are loaded from a project-private file at call
time and excluded from request/response artifacts. Native text output is parsed
as JSON, including fenced JSON; reasoning blocks are excluded from proposals.
A single unambiguous fenced JSON proposal may have surrounding prose; multiple
blocks or conflicting outside JSON objects are rejected.
Each backend call makes one HTTP request, without automatic retries, redirects,
or fallback models. Missing provider token counts remain explicitly unknown.

The requested model route stays fixed within a comparison and its continuation
branches. When a provider returns a model identifier, identity changes stop that
arm. Bedrock usually omits a resolved model identifier; the audit records the
requested model and encoded Converse route rather than inventing a returned ID.
Cloud inference compute is unobserved; recorded CPU/GPU limits describe the local
numerical harness.

At each round the model receives:

1. Public specifications for all 48 problems and the required proposal schema.
2. The active numerical settings, improvement workflow, generation, and inherited memory.
3. Evolution and current validation diagnostics, including L2, IC/BC, residual, timing, and pass status.
4. Remaining numerical-call, wall-time, and local CPU allowances.
5. Prior proposals and acceptance/rejection feedback within the current conversation.

The output contains exactly `hypothesis`, `solver`, `next_improvement_policy`,
and `memory_update`. `solver` supplies exactly one setting row for each family:

| Field | Allowed proposed value |
| --- | --- |
| `nx` | Integer 8–128 spatial intervals |
| `nt` | Integer 8–384 requested time steps |
| `grading` | 1–4; fractional time mesh is `t_j = T (j/nt)^grading` |
| `advection` | `central` or `upwind` |

Poisson ignores temporal settings. Grading applies to fractional solvers.
Burgers may raise actual time steps above the requested count to satisfy its CFL
bound, with a solver safeguard of 1,024 steps. `per_case_l2.csv` records the proposed
settings; `frozen_results.json` and the saved solution grids record actual steps.
The hypothesis is bounded to 3,000 characters, memory lesson to 1,200, and proposed
workflow to 2,048. Models have no code execution, filesystem access, tools, or
external search during these experiments.

## Single-session and recursive arms

Both arms begin with `nx=32`, `nt=48`, `grading=1`, centered transport, the same
initial workflow, and empty persistent memory. They first evaluate the 24
validation problems. Each improvement round then evaluates the active solver on
24 evolution problems, obtains one model proposal, evaluates the candidate on
24 validation problems, and applies the independent acceptance gate.

The single-session arm keeps the improvement workflow fixed. It can revise
solver settings and use prior conversation feedback, but its proposals must set
`next_improvement_policy` to `null`; generation and persistent memory remain
unchanged.

The RSI arm may propose an updated workflow and a reusable memory lesson.
Accepted numerical candidates inherit these changes, advance the generation,
and record the parent version. Rejected candidates remain archived without
changing the active agent. Later model proposals receive the accepted workflow
and memory, so the improvement procedure can influence subsequent searches.

After the requested rounds, each arm freezes its selected agent and evaluates
all 48 public problems. These repeated public measurements are development and
selection evidence. They do not provide sealed-test generalization evidence.

## Declared budgets and paired continuation

The default main comparison uses three rounds. Every main arm has identical
declared allowances:

| Allowance | Main arm | Each continuation branch |
| --- | ---: | ---: |
| Model requests | 3 | 1 |
| Numerical solves | 216 | 120 |
| Wall time | 900 seconds | 900 seconds |
| Local CPU time | 600 seconds | 600 seconds |
| Numerical-library threads | 1 | 1 |
| Local GPUs | 0 | 0 |
| Output tokens per model request | 4,000 | 4,000 |
| Input characters per model request | 180,000 | 180,000 |

Main numerical accounting is `24 + 3 × (24 + 24) + 48 = 216`: initial validation,
evolution/candidate validation each round, and frozen public evaluation.
For a one-round continuation it is `24 + (24 + 24) + 48 = 120`.
Failed or timed-out numerical solves and attempted failed model requests are
charged. Invalid proposals can produce fewer completed candidate evaluations;
the manifest reports actual usage and whether all requested rounds completed.
The input limit counts all message-content characters in the request.

The two continuation branches copy the same frozen RSI solver, memory,
generation, and model into fresh sessions. One restores the initial workflow;
the other retains the evolved workflow. Both use fixed workflows during the
probe. Their budgets and starting numerical capability match, allowing a paired
comparison of future improvement.

For each continuation branch, the improvement is initial validation macro
relative L2 minus selected validation macro relative L2. The recorded
`continuation_policy_advantage` is evolved-branch improvement minus
initial-branch improvement. A positive value supports an advantage on this
public continuation; a stronger current solver alone does not. Multiple seeds,
independent final tasks, and further generations remain necessary for stronger
scientific conclusions. Equal ceilings do not imply equal actual CPU use, token
use, model inference cost, or completed search; inspect the recorded accounting.

## L2 errors and evaluator checks

The independent evaluator compares the numerical solution with an analytic
manufactured reference on **65 spatial points and 17 uniform output times**.
Poisson uses 65 spatial points at its single stationary output. Numerical
solutions are interpolated to this fixed evaluation grid, then integrated with
trapezoidal quadrature. This reports errors of the interpolated numerical output
on the evaluator grid rather than a certified continuum norm.

For a time-dependent problem with error `e = u_numerical - u_reference`, the
reported metrics are:

| Output column | Definition |
| --- | --- |
| `absolute_l2` | `sqrt(integral_0^T integral_0^L e(x,t)^2 dx dt)` |
| `relative_l2` | `absolute_l2 / sqrt(integral_0^T integral_0^L u_reference(x,t)^2 dx dt)` |
| `final_absolute_l2` | `sqrt(integral_0^L e(x,T)^2 dx)` |
| `final_relative_l2` | `final_absolute_l2 / sqrt(integral_0^L u_reference(x,T)^2 dx)` |

Poisson omits the time integral, so its combined and final spatial metrics agree.
Relative denominators use the evaluator's small numerical floor. The scalar
absolute norms retain dependence on the physical domain, time horizon, and
solution amplitude; relative norms allow comparison across those scales.

An individual public case passes when both relative norms are at most `0.002`,
IC and boundary errors satisfy their task tolerances (currently `1e-10`), and
per-task resources fit the specification (currently 15 seconds and 1,024 MB).
The evaluator also checks real finite output, dimensions, monotonic grids, and
complete domain/time coverage. An independent final-time physical residual is a
diagnostic, not an acceptance certificate.

A proposed improvement is accepted only when:

- Every candidate evaluation executes successfully and passes IC/BC and resource checks.
- Validation macro relative L2 decreases by more than 0.5%.
- Each family's mean relative L2 increases by at most 5%, and its pass rate does not decrease.
- No paired instance's combined or final relative L2 increases by more than 25%, subject to the evaluator's small numerical floors.
- The paired task IDs and evaluator version match, without duplicate instances.

Macro relative L2 averages each family's mean, giving families equal weight.
The gate includes penalties for failed evaluations. Reported CSV metric means
use numerical records with `status="ok"`; failures remain visible in statuses,
pass rates, completeness checks, and the raw result files. Failed evaluations
have no measured absolute L2 and use a large relative-error penalty for selection.

## Verified public results — October 14, 2026

The completed v3 experiments produced the following arithmetic means across all
48 public problems per arm. Combined norms include time for evolutionary problems
and reduce to a spatial norm for Poisson.

| Model | Arm | Passes | Absolute L2 | Relative L2 | Final absolute L2 | Final relative L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GPT-OSS | Single session | 48/48 | 1.60307e-4 | 6.58692e-4 | 3.11488e-4 | 7.52955e-4 |
| GPT-OSS | RSI | 48/48 | 7.31164e-5 | 3.00900e-4 | 1.48960e-4 | 3.54163e-4 |
| Claude | Single session | 47/48 | 2.10984e-4 | 8.66179e-4 | 3.75574e-4 | 8.84309e-4 |
| Claude | RSI | 48/48 | 1.94230e-4 | 8.09481e-4 | 3.78798e-4 | 9.07095e-4 |

Every main arm completed all three rounds using exactly 216 numerical solves and
three model requests. Both continuation branches for each model used exactly
120 solves and one request. Actual call counts matched across paired arms, and
local resource ceilings were met. All 384 saved frozen solutions reproduced their
reported evaluator metrics; source, coverage, routing, and budget checks passed.

GPT-OSS showed no future-improvement gain in either continuation, giving a paired
policy advantage of zero. Claude's evolved workflow gained
`3.0003947443674097e-05` in validation macro relative L2 beyond the initial workflow.
These are exploratory results on repeatedly queried public cases; the positive
Claude continuation does not establish unseen-task generalization or scientific
RSI by itself.

The [consolidated report](../runs/model-agent-results-20261014/summary.md) includes
actual resources, tokens, continuation gains, and separately retained aborted
preparation calls, which are excluded from the final comparison budgets.
Inspect [per-case errors](../runs/model-agent-results-20261014/per_case_l2.csv),
[per-family errors](../runs/model-agent-results-20261014/per_family_l2.csv), and
[protocol checks](../runs/model-agent-results-20261014/protocol_checks.json).
The consolidated per-case table includes both requested and actual numerical grids.

## Running and inspecting an experiment

```bash
cd /depot/lin491/data/Binghang_project/PDE-RSI
bash scripts/run.sh agents --model-config configs/agent_models/gpt_oss.json \
  --rounds 3 --probe-rounds 1 --max-model-calls 3 --max-calls 216 \
  --wall-seconds 900 --cpu-seconds 600 --output runs/my-gpt-oss-agents
```

Replace the model configuration with `configs/agent_models/claude.json` for the
Claude comparison, and use a fresh output directory. `--probe-rounds 0` runs only
the main comparison. The command executes hosted inference using the configured
credential file.

| Artifact | Purpose |
| --- | --- |
| `comparison.csv` | Combined frozen public errors, pass rate, actual calls, and local resources |
| `per_family_l2.csv` | Error metrics and pass rates by family and public split |
| `per_case_l2.csv` | Four L2 metrics, status, pass indicator, and requested settings for every public case |
| `manifest.json` | Model configuration, source/data hashes, task IDs, allowances, identity/completeness checks |
| `single/report.json`, `rsi/report.json` | Initial, selected, frozen, and resource summaries |
| `single/frozen_results.json`, `rsi/frozen_results.json` | Raw per-case evaluator results, including actual grid sizes |
| `single/solutions/`, `rsi/solutions/` | Frozen numerical grids and solutions in NPZ format |
| Per-arm `model-call-*/` | Sanitized request/response audit; native request includes the pinned model route |
| Per-arm proposals, checkpoints, `events.jsonl` | Proposed settings, lineage, acceptance decisions, and failures |
| Per-arm `solve_calls.jsonl` | Every numerical solve, settings, status, timing, and memory |
| `continuation/comparison.csv` | Initial-workflow versus evolved-workflow future performance |

Results should be interpreted only after checking complete frozen evaluation,
completed candidate rounds, matched actual call counts, and resource ceilings.
The older `compare`/`probe` commands run the historical deterministic scaffold;
its records are preserved in [validation](validation.md). The current
model-backed command is `agents`.
