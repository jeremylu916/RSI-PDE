# Agent workflow and evaluation

## Inputs, outputs, and local workspace

Each public JSONL task states the equation, explicit forcing expression, spatial
domain and time horizon, coefficients, initial and Dirichlet boundary conditions,
permitted observations, requested output grid, accuracy tolerances, provenance, and
compute limits. FPDE records also state Caputo orders, positive weights, history
starting at zero, and the absence of exterior conditions for the local Laplacian.

An agent version contains a per-family solver policy, improvement-policy procedure
and parameters, persistent memory, parent identifier, generation, and frozen-model
identifier. The current model identifier is `none: deterministic numerical scaffold`.

Every entrypoint uses the dedicated conda interpreter in `envs/pde-rsi/` under
`/depot/lin491/data/Binghang_project/PDE-RSI`. Package and model caches, conda catalog
metadata, and temporary files stay in `.runtime/`; outputs stay under the project.
Use `scripts/setup_env.sh`, `scripts/conda.sh`, `scripts/run.sh`, and `scripts/test.sh`.
The `doctor` command verifies the interpreter, storage paths, disabled user-site
packages, and disabled bytecode writes. Output paths are resolved before execution,
including symlinks, and rejected if outside the project. HOME and global config
are untouched. These path controls do not provide isolation for arbitrary code.

```text
public task + agent version + remaining budget
    -> trusted numerical worker -> numerical solution
    -> evaluator -> evolution/development diagnostics
    -> active improvement policy -> candidate solver + policy + memory
    -> paired development gate -> accept or reject
    -> accepted improvement policy drives the next proposal
```

The project and episode workspace are the information boundary. Agent prompts in
`prompts/` prohibit searching the web, neighboring projects, hidden references,
sealed task files, or final results. The v0.1 worker is trusted shipped Python code;
it does not enforce a filesystem/network sandbox. Do not run untrusted generated
code with this worker. Later model execution must use an isolated copied workspace
containing only public tasks, the active solver/skills/memory, approved tool adapters,
and an evaluator RPC that never returns hidden solutions.

Numerical outputs are `(x,t,u,method)` internally. The external adapter uses
`solution.npz` with x, t, u and a scalar method description. The v0.2 `solve` command
writes this archive, public task, settings, measured metrics, resource ledger, and
solver-source hash. `evaluate-solution` independently reloads a project-local NPZ
without pickle and checks numerical accuracy and IC/BC. A submitted archive carries
no trusted execution measurements, so resources and overall pass are reported as
unknown. Solver and improvement prompts are contract templates, not evidence that
a foundation model has been run.

## Pilot and splits

Six families each have three spatial template groups and four parameter variants:
72 tasks total. Sine templates are evolution data, polynomial templates development
data, and mixed-sine templates reserved final data. All parameter variants of one
template stay together. The manufactured temporal profile is `1+t^beta`. Fractional
beta is 2.0 on evolution, 0.8 on development, and 0.55 on final data, reserving a
stronger initial-time singularity for final evaluation. Amplitudes, domains, time
horizons, diffusivities, velocities, and fractional orders vary within groups.

These are deliberately simple evaluator-validation cases, not an unbiased sample
of real-world PDE difficulty. Shared family and manufactured structure remain
visible across splits; grouping reduces instance leakage but does not demonstrate
transfer to unseen operator families. References are analytic, implemented separately
from numerical solvers, and checked against numerical differentiation and the
defining Caputo integral. No noisy observations or inverse tasks are included yet.

The task files store no reference arrays. The solver sees the forcing and IC/BC,
which may allow inference of a manufactured solution; that is an additional reason
not to present this pilot as an adversarial or realistic agent benchmark.

## Numerical methods

Poisson uses centered second-order finite differences. Heat and linear
convection–diffusion use implicit backward Euler and a tridiagonal solve. Transport
can be centered or upwind. Burgers treats diffusion implicitly and transport
explicitly, enforces a specified-envelope CFL limit, and checks the returned field.
Caputo and multi-term diffusion use nonuniform L1 history quadrature with a graded
mesh `t_n=T*(n/N)^r`; each fractional weight includes `Gamma(2-alpha)` normalization.
History costs O(Nt^2 Nx); fast history algorithms are a later improvement target.

Diffusion-wave orders above one, fractional spatial operators, arbitrary boundaries,
and general geometries are unsupported and must not be silently approximated with
the implemented operators.

## Metrics and gates

Metrics use common evaluator nodes (65 spatial points and 17 temporal points;
Poisson has one time sample) with linear interpolation of returned solutions.
Space-time relative L2 integrates squared error and squared reference using
trapezoidal weights. Also report final-time relative L2, absolute L-infinity,
native-grid IC and BC error, pass rate, per-family accuracy, solver runtime,
subprocess wall time, and Linux peak RSS. The accuracy threshold is 0.002 for both
space-time and final L2. IC and BC limits are 1e-10.

A final-time physical residual uses centered spatial differentiation; its local
time derivative differs from backward Euler, and its fractional derivative uses
PCHIP plus Gauss–Jacobi quadrature rather than solver L1 weights. It is diagnostic,
not an acceptance certificate. Coarse differentiation can have its own error.

Acceptance requires at least 0.5% improvement in the macro-average of per-family
relative L2, no family mean-error regression above 5%, no individual error
regression above 25%, no family pass-rate reduction, and successful execution,
IC/BC, and resource checks. A zero-error numerical floor protects the regression
comparison. Accepted candidates need not pass every accuracy tolerance; the gate
permits gradual progress toward passing. Efficiency is measured but is not optimized
by the initial accuracy-only gate; an accuracy-constrained efficiency gate is TODO.

## Budgets and comparison arms

Every task runs in a child process with one BLAS/OpenMP thread, no GPU use, a
15-second subprocess wall ceiling, a 1-GiB address-space cap, and a CPU-time cap.
Peak RSS is reported separately from the address-space ceiling. Failed and timed-out
solves still consume a solve call and enter `solve_calls.jsonl`. Each record includes
task/split, active settings, phase, status, wall time, child CPU time, and peak RSS.
Parent wall accounting includes proposal,
evaluation, serialization, and child startup. A solve receives the lesser of its
task time limit and the remaining arm time. The budget is sequential, not parallel.

Default ceilings are 240 solve calls, 600 wall seconds, and 600 aggregate CPU seconds
per arm. `--cpu-seconds` changes the CPU ceiling independently. Linux resource usage
measures parent CPU (proposal, evaluation, and orchestration) and child CPU (including
startup and failed solves). The worker's CPU limit receives the lesser of task and
remaining arm limits. Aggregate limits are checked before calls and after each
result; child CPU signals have integer-second granularity, and parent evaluation
can finish across a ceiling. Such overruns are reported and cannot pass resources.
This is a sequential allowance, not scheduler-enforced aggregate CPU isolation.

Initial development
costs 24 calls; each candidate cycle costs 24 evolution plus 24 development calls;
24 calls are reserved for final tests. Four complete updates plus final evaluation
therefore cost 240 calls. The smoke protocol uses one variant per family, two
updates, and 36 calls per arm including the reserved final pass.

| Arm | Procedure | Persistent version updates |
| --- | --- | --- |
| `single` | Fixed search in one session using all requested candidate slots | Best solver retained; no reusable memory, policy, or generation updates |
| `fixed` | Initial procedure refines all families each cycle | Solver and memory only |
| `rsi` | Accepted procedure ranks families from evolution errors and changes future refinement/grading | Solver, memory, and improvement procedure |

All three receive the same initial development evaluation and evolution/development
candidate slots. `single` keeps working solver settings inside the session and
selects the best candidate under the same gate; its persistent memory, improvement
policy, parent, and generation stay fixed. Its proposal/checkpoint files are audit
traces, not inherited future agent generations. Because the current proposal
procedure does not consult memory, single and fixed should reach the same solver
when evaluations succeed. `accepted_candidates` counts session selections separately
from accepted agent generations. The explicit `--single-mode static` option retains
the old initial-solver control, which spends fewer calls and is labeled accordingly.
Insufficient call allowances for requested updates plus final reservation are rejected.

Equal ceilings do not imply equal actual compute. All three use identical numbers
of evaluation slots when they complete the same candidate cycles; grid costs can
differ. Manifest flags report matched calls, completed searches, and resource checks.
Wall time is hardware- and load-dependent. Full matched-compute LLM experiments will need pinned model identity,
token and tool budgets, measured CPU/GPU usage, repeated seeds, and a single-run arm
that can spend its complete tool allowance in one uninterrupted session without
transmitting evolved state between generations.

## Evidence specifically for recursive improvement

The `probe` command begins two continuations from the exact same solver and memory.
Only the improvement policy changes: initial uniform refinement versus the accepted
evolved procedure. Both policy definitions stay fixed during these future updates
to isolate their effect. Both receive the same solve-call and wall-time ceiling.
Outputs record whether actual solve counts match and whether both continuations
complete all requested updates. The difference in future development-error reduction
is an exploratory policy-advantage estimate; it may be negative.

Because probes reuse development data, they are selection measurements. A scientific
RSI claim requires repeated controlled continuation experiments, a predeclared
budget, held-out continuation tasks, paired uncertainty estimates, and a final
release that was never used for meta-policy selection. `generation` counts accepted
updates, while `attempted_updates` counts evaluations; probe reports begin from the
source snapshot generation, so those accepted-generation values are cumulative.

## Frozen final tests

`compare` and `probe` never load sealed task contents. `finalize` verifies source,
environment lock and Python/NumPy/SciPy versions, benchmark, and agent hashes and
evaluates the frozen three-arm comparison once.
An exclusive `FINAL_CONSUMED.json` marker records consumption even if evaluation
fails after starting. Results cannot feed back into versions. Source changes
require a fresh comparison; do not edit the evaluator between freezing and final
evaluation.

The directory name and mode 0600 enforce only protocol separation: the project owner
can still read reserved data. For genuine sealing, use another account/evaluator
service and hide task generators, reference source, and final-case parameters from
agent workspaces. The marker is per run, not a guarantee that final data have never
been observed through another run. Later experiments need new held-out cases after
the pilot final release.
