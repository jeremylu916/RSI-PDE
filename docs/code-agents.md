# Code-writing single-run and RSI agents

In the earlier settings agents, the model chose grid sizes for trusted solvers.
In this protocol, the frozen model **writes the solver programs**: one standalone
Python module per PDE family, each defining `solve(problem)`. The harness checks
each module, runs it in a sandbox, scores it with the unchanged independent
evaluator (`manufactured-evaluator-v2-l2`), and applies the same acceptance gate.

Implementation: [`code_agents.py`](../src/pde_rsi/code_agents.py) (agent loop and
comparison), [`codegen.py`](../src/pde_rsi/codegen.py) (static policy, sandbox,
metering), [`sandbox_worker.py`](../src/pde_rsi/sandbox_worker.py) (in-sandbox
runner), and [`classical_programs/`](../src/pde_rsi/classical_programs/).

## Starting programs

Runs launched before the rename on October 14, 2026 record the classical start as
`seed` in their manifests. The report script displays these as `classical`.

| `--start` | Starting program for both arms |
| --- | --- |
| `classical` | Standalone versions of the six classical solvers (`NX=32`, `NT=48`, `GRADING=1`, centered transport). In the sandbox they reproduce the trusted baseline: 45/48 passes, mean relative L2 1.01879e-3. |
| `scratch` | A stub per family that raises `NotImplementedError`. The agent must write every solver itself. |

A classical start tends to produce constant tuning (the preflight round-1 proposals
from both models only changed `NT` and `GRADING`). The scratch start makes the
numerical method itself the agent's output.

## Families

`--families` selects which families a comparison covers: `all` (the six 1D
families, `wave2d`, and the four group FPDE families), `1d`, `wave2d`, `group`, or a
comma-separated list. The default is `all`. Group families are described in the
problem catalog. Their programs receive `problem.alpha(t)`, and the FODE returns
`{"t", "u"}` with a one-dimensional `u`. The proposal schema, solver contract, starting programs, gate, and
tables cover only the selected families. Two-dimensional programs return
`{"x", "y", "t", "u"}` with `u[t, y, x]`. They receive `problem.forcing(x, y, t)`,
`problem.initial(x, y)`, and `problem.initial_velocity(x, y)`, all of which accept
broadcastable arrays. Adding the 2D path bumped the evaluator to
`manufactured-evaluator-v3-l2-2d`. The 1D formulas are unchanged, and a test checks
that 1D metrics match the v2 values. The sandbox runner is `sandbox_worker.py`
(`bwrap-v2-2d`).

## Solver contract and information boundary

`solve(problem)` returns `{"x", "t", "u"}`. The rules are:

- `x` is uniform on `[0, L]`.
- Poisson returns `t=[0]`. Other families return increasing `t` on `[0, T]`.
- `u` has shape `(len(t), len(x))`.
- `u[0]` matches the initial data and the boundaries are zero (within 1e-10).

`problem` exposes the family, equation, coefficients (`length`, `final_time`,
`diffusivity`, `velocity`, `orders`, `weights`), tolerances, limits, and
evaluator node counts. It also provides `forcing(x, t)` and `initial(x)` as
**black-box callables**.

The manufactured-profile parameters (`profile`, `amplitude`, `mode`, `beta`) are
hidden from both the model and the program, along with the closed-form forcing
and initial expressions and the template group. Together these would reveal the
exact solution. Task IDs appear only in the model's diagnostics; the program
never receives them. The full solver contract text is `SOLVER_CONTRACT` in
`code_agents.py`.

## Isolation

Each solve runs as a fresh process in four layers:

1. **Static policy.** Every module is checked before execution. Imports are
   limited to `math`, `cmath`, `functools`, `itertools`, `collections`,
   `dataclasses`, `typing`, NumPy, and selected SciPy submodules. The check
   rejects `eval`, `exec`, `open`, `getattr`, `type`, and `dir`; any name or
   attribute starting with `_`; NumPy file I/O; and frame/code introspection.
   A module needs exactly one top-level `solve` and at most 30,000 characters.
2. **Restricted builtins.** The program sees only safe builtins and a runtime
   import allowlist. All allowed modules are imported before the resource limits
   take effect.
3. **Bubblewrap sandbox.** The sandbox has no network and its own namespaces.
   The only read-only mounts are `/usr` and the project interpreter; `/tmp` is
   private. Project source, evaluator, references, credentials, home, and run
   outputs are absent. `sandbox_self_test()` checks this before every
   comparison and refuses to run if isolation fails.
4. **Resource limits.** Address space is capped at `memory_mb`. CPU time is
   capped at `max_seconds`. File-size writes are capped at zero. One BLAS
   thread, zero GPUs, and at most 4,000,000 returned values.

The parent process records the following as charged failures:

- crashes, with the traceback filtered to the program's own frames;
- timeouts and signals;
- invalid output;
- non-finite error integrals from unstable schemes.

## Data flow for one task

One task, for example `caputo-g1-v3`, exists in three views. Only the harness holds all three.

| View | Built by | Contents | Goes to |
| --- | --- | --- | --- |
| Full record | `benchmarks/development.jsonl` → `Task` (hash-checked against the manifest) | All fields, including `profile`, `amplitude`, `mode`, `beta`, and the closed-form forcing | Harness only |
| Model view | `codegen.agent_problem_record` | `task_id`, split, family, equation string, coefficients, tolerances, limits, evaluator node counts | Prompt, once, in `public_problems` |
| Program view | `codegen.public_problem`, then `sandbox_worker.build_problem` | The same numbers without `task_id`/split, plus `forcing(x, t)` and `initial(x)` closures over the hidden fields | `solve(problem)` inside the sandbox |
| Exact solution | `reference.py` | `u(x, t)` on the evaluator nodes | `evaluator.evaluate` in the harness |

Each solve is one call to `codegen.run_program_task(task, code, budget)`:

```
harness                                   sandbox (bwrap: no network, no project files)
-------                                   ---------------------------------------------
budget.reserve(1); solver_calls += 1
request JSON ──stdin──────────────────▶  build problem view (hidden fields stay in closures)
  code = program[task.family]            restricted builtins and imports; rlimits on CPU, memory, file size
  problem = public numbers               exec(code); solve(problem) -> {x, t, u}
  hidden = profile/amplitude/mode/beta   check real arrays, size <= 4,000,000 values
  cpu limit, value limit                 stdout: NPZ(x, t, u, runtime, peak memory, cpu)
                       ◀──stdout/stderr── on error: traceback of agent frames only, exit 3
parse NPZ -> Solution
evaluate(task, solution):  grid checks -> interpolate to 65 x 17 nodes -> compare to exact
  -> relative/absolute L2 (combined and final), IC/BC error, residual, resource checks, passed
non-finite metrics -> invalid_output; crash -> execution_failure; > max_seconds -> timeout
append one line to solve_calls.jsonl; frozen phase also saves solutions/<task_id>.npz
```

Where the result goes depends on the task's split and the phase:

| Phase (`solve_calls.jsonl`) | Program | Result used for |
| --- | --- | --- |
| `evolution_k` (training task) | Active program | `evolution_diagnostics` in round k's prompt |
| `self_test_k-a` (training task) | Candidate, attempt a | Broken if status is not ok, IC/BC or resources fail, or relative L2 >= 1. Broken candidates go back to the model for revision; a clean one is submitted |
| `initial_validation`, `candidate_validation_k` (validation task) | Starting program / submitted candidate | One of 24 inputs to the acceptance gate, then `candidate_validation_diagnostics` in the prompt. An accepted candidate's results become the next round's `validation_diagnostics` |
| `frozen_public_evaluation` | Final agent | `frozen_results.json`, `per_case_l2.csv`, `solutions/<task>.npz`, re-verified by the report script |

The model sees one diagnostics row per task (`released_diagnostics`): status, the four L2
metrics, IC/BC error, residual, timing, memory, pass flags, grid sizes, and any error message.
Reference values, hidden fields, and reserved cases never leave the harness, and the program
never receives `task_id`.

Example from `runs/code-agents-v2-classical-opus55-20261014/rsi` (validation task `caputo-g1-v3`, 8 solves):

| Phase | Program | Relative L2 | Gate |
| --- | --- | ---: | --- |
| initial_validation | `25c729bd` (classical start) | 2.81e-3 (fails) | — |
| candidate_validation_001 | `4b62f162` | 7.37e-5 | accept |
| candidate_validation_002 | `0eecb6f2` | 1.66e-5 | accept |
| candidate_validation_003 | `c83b27cb` | 1.40e-6 | accept |
| candidate_validation_004 | `c83b27cb` (Caputo program unchanged) | 1.40e-6 | accept (gain in other families) |
| candidate_validation_005 | `ac4f132e` | 7.28e-7 | accept |
| candidate_validation_006 | `ac4f132e` | 7.28e-7 | accept |
| frozen_public_evaluation | `ac4f132e` | 7.28e-7, 128 x 1047 grid, 2.8 s | — |

Training task `caputo-g0-v3` in the same arm was solved 15 times: 6 evolution evaluations,
8 self-tests (including revisions), and 1 frozen evaluation.

## Rounds, arms, and gate

The structure matches the settings protocol. Both arms start from the same
program, workflow, and empty memory. Each round:

1. Run the active program on the 24 evolution problems.
2. Make one model request.
3. Statically check the proposal. A rejected proposal is not executed.
4. Run the candidate on the 24 validation problems.
5. Apply the gate.

A proposal contains `hypothesis`, `program_updates` (1–6 complete replacement
modules, each with a `change_summary`), `next_improvement_policy`, and
`memory_update`. Families that are not listed keep their active program.

- **Single run** (since 2026-10-20): **one model request**. It carries what RSI sees in its first round:
  the task, the solver contract, the starting programs, and their training and validation diagnostics. The
  returned programs are frozen as they are, with no feedback, revision, self-test, or gate. An answer that
  fails the static policy or the schema leaves the starting programs in place. Only provider failures
  (transport, HTTP 429/5xx) are re-sent. Implementation: `request_one_shot` / `finish_one_shot` in
  `code_agents.py`.
- **Fixed workflow** (the old "single session", replaced on 2026-10-20): it iterated gated rounds under a
  fixed workflow with no inherited memory. Each run keeps this arm as `<run>/fixed_workflow/` for reference;
  reports no longer show it. `scripts/redo_single_one_shot.py` produced the single runs of all earlier runs.
- **RSI:** an accepted candidate also inherits the proposed workflow
  instruction and a memory lesson. These then guide later rounds.

The gate is unchanged. A candidate is accepted only when all of these hold:

- no validation case fails on execution, IC/BC, or resources;
- macro relative L2 drops by more than 0.5%;
- no family mean rises by more than 5%, and no pass rate drops;
- no case's combined or final-time relative L2 rises by more than 25%.

For a scratch start, the first acceptance therefore requires all six families
to run.

The transcript is compacted between rounds. Older contexts and proposals keep
code hashes and short per-case rows with truncated failure messages. The latest
context, proposal, and decision stay complete.

## Budgets

The run configuration in October 2026 used 6 main rounds and 2 continuation
rounds.

| Allowance | Each main arm | Each continuation branch |
| --- | ---: | ---: |
| Model requests | 6 | 2 |
| Sandboxed solves | 24 + 6×48 + 48 = 360 | 24 + 2×48 + 48 = 168 |
| Wall / local CPU seconds | 3,600 / 2,400 | 3,600 / 2,400 |
| Output tokens per request | 32,000 | 32,000 |
| Input characters per request | 500,000 | 500,000 |
| Threads / GPUs | 1 / 0 | 1 / 0 |

Every run also evaluates the starting programs once on all 48 cases. This
reference run is outside every arm's budget.

The continuation branches copy the frozen RSI agent (same programs, memory,
and generation). One branch restores the initial workflow; the other keeps the
evolved workflow. Both run with fixed workflows.
`continuation_policy_advantage` = (evolved-branch validation gain) −
(initial-branch validation gain).

## Opt-in RSI options (October 21, 2026)

Three options change only the RSI arm. The single run stays one request. With
the defaults, prompts, schemas, events, files, and reports are unchanged; this
was checked by running the earlier `code_agents.py` and the current one on the
same fake model responses. Options that differ from their defaults are recorded
in `manifest.json` under `configuration.rsi_options` and in the arm's
`report.json` under `rsi_options`. The options work with the
harness v2 `--modular` programs.

| Flag | Default | Effect |
| --- | --- | --- |
| `--best-of K` | 1 | Each attempt sends K requests in parallel from the same context. Each request carries a sample-index note, so temperature-0 configurations still differ. Every valid sample is self-tested on the training cases. The passing sample with the lowest training macro relative L2 is submitted, and only that sample enters later context. If every sample is broken, the least broken one is returned as self-test feedback with an overview of all samples. A `best_of` event records every sample. |
| `--gate pass-first` | `standard` | Passes come first. A candidate that passes more validation cases and makes no passing case fail is accepted even when the macro gain is under 0.5%, a family mean rises, or a case failing in both versions gets worse. Execution, IC/BC and resource failures still reject. Cases passing in both versions keep the 25% instance-regression check, which protects their margin to the tolerance. Accepted gate decisions record `gate.pass_first = {gained, waived}`. The harness v2 per-part gate applies the same rule within each part. |
| `--experiments-per-round E` | 0 | A proposal may add `experiment = {"cases": [...], "question": "..."}`. The harness applies the updates to the active program, runs the candidate on the named training cases (all training cases for an empty list), and returns its diagnostics beside the active program's on the same cases. An experiment is never gated and uses no attempt. It costs one model request plus the requested solves. Validation and unknown case IDs are refused. After E experiments in a round, a proposal is self-tested and submitted as usual (`experiment_ignored` in the self-test event). Each experiment writes an `experiment` event and `experiment-NNN-AxM.json`. |

`--best-of` and `--experiments-per-round` cannot be combined. Default model and
solver allowances scale with the options: model calls are
rounds × (attempts × K + E), and each round reserves
training × (1 + attempts × K + E) solves. Under a time budget, K samples
finish in about the time of one request but cost K times as much. The spending
cap is checked before each batch, so the last batch can overshoot it by up to
K − 1 responses.

The `pass-first` rule is a selection rule in `acceptance_gate`. Scoring and
tolerances are unchanged, so `EVALUATOR_VERSION` is unchanged.

## Running

```bash
cd /depot/lin491/data/Binghang_project/PDE-RSI
bash scripts/run.sh code-agents --model-config configs/agent_models/claude_code.json \
  --start scratch --rounds 6 --probe-rounds 2 --output runs/my-code-agents
```

The opt-in options above are added to the same command, for example
`--best-of 3 --gate pass-first` or `--experiments-per-round 2 --gate pass-first`.

Use `gpt_oss_code.json` for GPT-OSS-120B and `--start classical` to start from the classical solvers.
Each output directory contains:

- `summary.md`, `comparison.csv`, `per_family_l2.csv`, `per_case_l2.csv`, and
  `manifest.json` (configuration, hashes, sandbox self-test, budgets, checks).
- Per arm: `programs/{initial,candidate-NNN,frozen}/<family>.py`, proposals,
  sanitized model request/response audits, `events.jsonl` (gate decisions),
  `solve_calls.jsonl`, and the frozen solution NPZ files.
- `continuation/` with the paired branches.

## Results: October 14, 2026

These October 14–15 sets (this section and the next two) are superseded by the fine-grid runs below. Their
"Single" column is the old iterated single-session arm (a fixed workflow over 6 gated rounds), not the one-request
single run. They were not redone: the evaluator and classical solvers have changed since then.

The consolidated report is in `runs/code-agent-results-20261014/` (summary, CSVs, per-round
curve, and protocol checks). Mean relative L2 over all 48 public cases:

| Start | Model | Single | RSI | Continuation advantage |
| --- | --- | ---: | ---: | ---: |
| classical | Claude Sonnet 4.5 | 3.83e-4 (48/48) | 5.08e-4 (48/48) | +1.0e-5 |
| classical | GPT-OSS-120B | 3.86e-4 (48/48) | 7.74e-4 (46/48) | +8.3e-5 |
| scratch | Claude Sonnet 4.5 | 0.194 (16/48) | none accepted (0/48) | n/a (identical branches) |
| scratch | GPT-OSS-120B | none accepted (0/48) | 0.044 (16/48) | 0 |

The starting classical programs score 1.02e-3 (45/48). This single exploratory run per condition
shows no RSI advantage in the main comparison. With a classical start, the accepted changes were
constant retuning. With a scratch start, progress was limited by the rule that a candidate is
accepted only if all six families run. The summary file lists the failure modes.

## Results with the self-test, five models (October 14, 2026)

These results come from 20 comparisons: 5 models × {1D, wave2d} × {classical, scratch}.
The consolidated report is `runs/code-agent-results-v2-20261014/`. It contains the summary,
`comparison.csv`, `per_family_l2.csv`, `per_round.csv`, `continuation.csv`, and
`protocol_checks.json`. All 1,120 saved solutions reproduce their metrics.

Each cell gives the mean relative L2 of the frozen agent over the public cases, with passes in parentheses.

| Benchmark, start | Model | Single | RSI | Continuation advantage |
| --- | --- | ---: | ---: | ---: |
| 1D, classical | Sonnet 4.5 | 5.29e-4 (48) | 4.04e-4 (48) | −1.7e-5 |
| 1D, classical | GPT-OSS-120B | 5.99e-4 (48) | 5.29e-4 (48) | −1.8e-5 |
| 1D, classical | Opus 5.5 | 1.94e-8 (48) | 8.08e-8 (48) | +2.1e-8 |
| 1D, classical | GPT-6.1 | 2.40e-6 (48) | 7.28e-7 (48) | ≈0 |
| 1D, classical | GPT-5.5 | 4.40e-5 (48) | 4.71e-5 (48) | −1.0e-6 |
| 1D, scratch | Sonnet 4.5 | 0.292 (22) | 0.129 (14) | −0.117 |
| 1D, scratch | GPT-OSS-120B | 0.325 (16) | 0.0133 (14) | 0 |
| 1D, scratch | Opus 5.5 | 2.22e-9 (48) | 1.81e-8 (48) | −1.0e-8 |
| 1D, scratch | GPT-6.1 | 2.86e-7 (48) | 1.70e-7 (48) | ≈0 |
| 1D, scratch | GPT-5.5 | 2.41e-6 (48) | 7.86e-6 (48) | +6.9e-8 |
| wave2d, classical | Sonnet 4.5 | 5.47e-3 (4) | 7.63e-3 (4) | −4.5e-4 |
| wave2d, classical | GPT-OSS-120B | 5.75e-3 (4) | 1.89e-3 (4) | +1.3e-3 |
| wave2d, classical | Opus 5.5 | 2.66e-7 (8) | 5.01e-8 (8) | −9.9e-8 |
| wave2d, classical | GPT-6.1 | 1.30e-7 (8) | 1.49e-7 (8) | −1.4e-7 |
| wave2d, classical | GPT-5.5 | 6.62e-4 (6) | 2.78e-4 (8) | −1.7e-4 |
| wave2d, scratch | Sonnet 4.5 | 1.21e-2 (0) | 2.40e-3 (4) | +2.8e-3 |
| wave2d, scratch | GPT-OSS-120B | 0.111 (0) | 458 (4) | 0 |
| wave2d, scratch | Opus 5.5 | 2.66e-7 (8) | 2.06e-7 (8) | 0 |
| wave2d, scratch | GPT-6.1 | 2.66e-7 (8) | 1.69e-7 (8) | 0 |
| wave2d, scratch | GPT-5.5 | 4.21e-5 (8) | 6.49e-4 (6) | +2.7e-5 |

Model choice changes errors by up to five orders of magnitude. Single session vs RSI has no
consistent winner: RSI does better in 11 of 20 comparisons and single in 9, with several
near-ties. Continuation advantages are 5 positive, 9 negative, and 6 zero. Strong models pass
every case, so this benchmark no longer discriminates among them. Model IDs:
`us.anthropic.claude-opus-5-5`, `us.openai.gpt-6.1-sol`, and `us.openai.gpt-5.5`. These three
reject the `temperature` field, and their configurations set `"temperature": null`.

## Results on the group FPDE examples (group-fpde-v2, October 15, 2026)

These are 10 comparisons (5 models × classical and scratch starts) on the 16 public group cases:
FODE, Caputo FPDE, time-variable order, and space-time variable order. The report is
`runs/code-agent-results-group-20261015/` (with `score_vs_round.png`). All 304 saved solutions
reproduce their metrics. CPU metering is valid in these runs (`bwrap-v4-group`). The classical
starting programs pass 7 of 16 cases (mean relative L2 6.3e-3).

Each cell gives the mean relative L2 over the 16 public cases, with passes in parentheses.

| Start | Model | Single | RSI | Continuation advantage |
| --- | --- | ---: | ---: | ---: |
| classical | Sonnet 4.5 | 1.48e-3 (13) | 8.96e-4 (15) | −9.0e-5 |
| classical | GPT-OSS-120B | 3.66e-4 (16) | 5.82e-4 (16) | −3.8e-4 |
| classical | GPT-5.5 | 1.81e-5 (16) | 1.34e-5 (16) | +1.1e-5 |
| classical | GPT-6.1 | 5.32e-6 (16) | 6.16e-6 (16) | +8.6e-7 |
| classical | Opus 5.5 | 1.68e-8 (16) | 7.87e-11 (16) | +4.8e-12 |
| scratch | Sonnet 4.5 | 0.545 (2) | 4.40e-4 (16) | −1.7e-4 |
| scratch | GPT-OSS-120B | none accepted (0) | 4.72e-2 (4) | 0 |
| scratch | GPT-5.5 | 1.08e-5 (16) | 1.88e-5 (16) | +6.3e-6 |
| scratch | GPT-6.1 | 6.51e-6 (16) | 5.39e-5 (16) | +2.8e-5 |
| scratch | Opus 5.5 | 8.80e-8 (16) | 9.92e-8 (16) | −7.0e-9 |

- **RSI vs single session.** Each arm is better in 5 of the 10 comparisons. Continuation
  advantages are 5 positive, 4 negative, and 1 zero.
- **Model strength.** As with the other benchmarks, model strength dominates. Opus 5.5 reaches
  1e-11 to 1e-16 on the paper instances.
- **FODE method.** The FODE with u(0) = 0 has the exact representation u = I^alpha f, the
  Riemann-Liouville integral of the forcing. Opus 5.5 and GPT-6.1 evaluate it by Gauss-Jacobi
  quadrature on samples of f, which reaches machine precision. These programs were inspected and
  do not fit or hard-code a closed form.
- **Saturation.** Every case passes for GPT-5.5, GPT-6.1, and Opus 5.5, so this set also
  saturates for strong models.

## Fine evaluation grid (evaluator v5, October 16, 2026)

Evaluator v5 measures errors on finer node grids:

| Family | Grid |
| --- | --- |
| 1D families, including the group FPDEs | 512 x-nodes x 500 times |
| Poisson | 512 x-nodes |
| FODE | 500 times |
| wave2d | 128 x 128 x 100 |

Before v5, every family used 65 x 17 (wave 65 x 65 x 17; FODE 65 times). The grid is defined in
`tasks.evaluation_points`, and agents receive it as `output_space_points` and `output_time_points`
in their problem view. The task records keep their original `output_*` fields so that the published
benchmark hashes are unchanged. `evaluate(..., grid="legacy")` reproduces v2–v4 scores exactly, and
the report script uses it to re-verify older runs.

Re-scoring the saved frozen solutions on the fine grid (`runs/fine-grid-rescore-20261016.csv`)
shows that the coarse grid hid most of the strong models' error:

| Model | 1D | wave2d |
| --- | --- | --- |
| Opus 5.5 | 1e-8 to 2e-9 becomes about 2e-5 and 5e-6 | about 1e-7 becomes about 3e-3, with 4/8 passing |
| GPT-6.1 | 1e-7 to 1e-6 becomes 1e-5 to 1e-4 | about 1e-7 becomes about 3e-3, with 4/8 passing |

These models had matched their output to the 65 x 17 nodes. The wave programs returned only the 17
evaluator time levels, so linear interpolation between them is now inaccurate. Errors for Sonnet 4.5
and GPT-OSS change by less than 10%. All earlier results are therefore measurements on the legacy
grid. The fine-grid comparisons are `runs/code-agents-fine-*-20261016`.

## Results on the fine grid, 30 comparisons (October 16, 2026)

The final report is `runs/code-agent-results-fine-20261016/`. It contains `per_equation_table.md`
(single vs RSI for every equation, model, and start), `score_vs_round_1d_wave.png`,
`score_vs_round_group.png`, CSVs, and `protocol_checks.json`. The interim report
`code-agent-results-fine-interim-20261016/` is superseded.

The single column is the **single run**: one model request, with the solver frozen as returned (redone on
2026-10-20). The earlier iterated fixed-workflow arm is archived in `runs/archive-fixed-workflow-20261020/`.

All saved solutions reproduce their metrics, the sandbox was isolated, and no reserved cases were used.

Over the 30 comparisons, ranked by passes and then mean relative L2, RSI is better in 21 and the single run in 9.
On passes alone, RSI passes more cases in 9 comparisons, the single run in 3, and 18 are tied.

- **wave2d:** RSI wins all 10 comparisons. Five of the ten single-run answers were rejected by the static policy:
  imports of `time` or `numpy.polynomial.legendre`, or use of `getattr`.
  - A classical start then keeps the classical solver (2/8 passes).
  - A scratch start keeps the stub (0/8).
- **1D, from the classical solvers:** the single run wins 4 of 5; one well-aimed edit already lowers the error.

Continuation advantages are 12 positive, 10 negative, and 8 zero. They concern the RSI arm only.

Model strength dominates:

- GPT-6.1 and Opus 5.5 reach 1e-9 to 1e-14 on most equations in at least one arm.
- Sonnet 4.5 and GPT-OSS stay at 1e-4 to 1e-3, or fail from scratch.
- Poisson still reaches about 1e-13 for strong models, so it does not separate them.

## CPU metering correction (October 14, 2026)

Before sandbox version `bwrap-v3-cpu`, CPU time used by agent-written code was not
metered. The solver process runs as a grandchild in bubblewrap's own PID namespace,
so the parent's `RUSAGE_CHILDREN` counter never saw it. Recorded per-solve and
per-arm CPU was therefore about zero, and the per-arm CPU ceiling was not enforced
for sandboxed code.

These limits were still enforced: per-solve CPU and memory limits inside the
sandbox, per-solve and per-arm wall-time ceilings, and solve-call counts.

From `bwrap-v3-cpu` on, the sandboxed process reports its own CPU time, failed or
killed solves are charged their wall time, and `Budget.sandbox_cpu_seconds` counts
toward the arm's ceiling. The report script marks earlier runs with
`cpu_metering_valid = False` and gives `sandbox_solve_seconds` (summed in-sandbox
`solve()` time) as a proxy.

## Limitations

- These results are on the 48 public cases. The validation cases are queried
  repeatedly during selection, there is one run per model and start, and the
  reserved final cases are not used.
- The static policy and sandbox block file, network, and process access and
  hide the closed-form parameters. A program could still try to fit a closed
  form from forcing samples. The contract forbids this but cannot detect it;
  inspect the frozen programs.
- Seeds, held-out continuation sets, efficiency gates, benchmark imports, and
  PINN/FPINN tracks are not part of this protocol.
