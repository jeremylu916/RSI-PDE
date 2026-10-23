# pde_benchmarks.tex suite (student protocol)

`pde_benchmarks.tex` defines 11 problems (20 cases). `src/pde_rsi/pdebench.py` turns them into a student-style task.
Each case gives the agent the following:

- 256 noisy samples of the exact solution (1 % RMS Gaussian noise);
- the domain and which coordinates exist (`x`, `x,t`, `t`, or `x,y,t`);
- black-box `forcing`, `initial`, `initial_velocity` (2-D only), and `boundary(t)` (1-D Dirichlet values, possibly
  nonzero);
- the library of 10 equation classes.

The agent picks the class and coefficients, chooses a method (numerical or PINN on the GPU), and writes one solver
program from scratch. The program returns the solution and `model = {family, parameters, method}`.

Run: `bash scripts/run.sh code-agents --protocol pdebench --start scratch --model-config <cfg> --rounds 6
--probe-rounds 2 --attempts-per-round 3 --wall-seconds 14400 --cpu-seconds 14400 --output runs/<name>`.

## Cases and splits

Manufactured cases (1a, 2a, 3a, 4a, 5, 7, 8) use the tex profiles by split:

| tex profile | split | role |
| --- | --- | --- |
| training (sine) | evolution | labeled feedback |
| validation (polynomial) | development | acceptance gate |
| reserved (mixed sine) | sealed test | scored only after freezing |

Their L, T, A, D, β, v, α and w values are drawn per case. The literature cases use different numbers in each split:

| Case | Training | Validation | Sealed test |
| --- | --- | --- | --- |
| 1b multiscale Poisson | | D = 1 | |
| 1c arctan layer | α = 20 | | α = 1000 |
| 2b free decay | L = 2, D = 0.3 | | L = 1.5, D = 0.1 |
| 2c periodic source | | γ = 2π, D = 0.2, T = 2 | γ = 10, D = 0.05 |
| 3b steady layer, f = 0 | Pe = 20 | | Pe = 200 |
| 3c steady layer, f = 1 | | Pe = 50 | Pe = 100 |
| 3d advected Gaussian | v = 1, D = 0.02 | v = −0.8, D = 0.015 | |
| 4b Burgers −sin(πx), D = 0.01/π | | ✓ | |
| 4c viscous shock | u_l, u_r = 1, 0; D = 0.02 | | u_l, u_r = 1.5, 0.5; D = 0.005 |
| 6 FODE | α = 0.4, p = 2 | α = 0.7, p = 0.8 | α = 0.9, p = 1.2 |
| 9 Caputo, zero IC | α = 0.5, p = 2 | α = 0.8, p = 1.5 | α = 0.3, p = 0.6 |
| 10 α(t) = c sin²t | c = 0.8, p = 2 | c = 1.3, p = 2.5 | c = 1.0, p = 1.5 |
| 11 α(x,t) = c t² sin x | c = 0.5, p = 2 | c = 0.9, p = 3 | c = 0.7, p = 1.5 |

That gives 16 training, 16 validation and 17 sealed-test cases. Every class appears in both public splits.

Case IDs (`case-t*`, `case-v*`, `case-r*`) are hash-ordered, so they reveal neither the class nor the tex case. The
following are hidden from the agent:

- the class and coefficients;
- the tex case number;
- every manufactured profile parameter.

The self-test runs one training case of each output kind (1-D evolution, steady, ODE, 2-D) before validation.

## Evaluation

Each evaluation is unchanged in kind from the other protocols. The output is linearly interpolated onto the
evaluation grid:

| Case type | Grid |
| --- | --- |
| 1-D time-dependent | 512 x × 500 t |
| Steady | 512 |
| ODE | 500 |
| 2-D | 128 × 128 × 100 |

On that grid:

- **Pass condition:** the trapezoidal space-time relative L2 and the final-time relative L2 are both ≤ 2e-3, IC and
  Dirichlet data hold to 1e-10, and the solve fits in 90 s of wall/CPU time and 16 GB.
- **Exact solutions:** closed forms from the tex. Case 4b uses Cole–Hopf with a 4001-node trapezoid in z (with
  log-sum-exp scaling). It reproduces the known peak slope |u_x| = 152.0 at t = 1.6037/π. `tests/test_pdebench.py`
  checks every forcing against the operator applied to the exact solution.
- **Identification:** class accuracy and relative parameter errors are reported but never gated. For 3b only v/D is
  identifiable.

## Harness fixes made for this suite

- `codegen.run_program_task` takes an `evaluator`, so this protocol scores against its own exact solutions.
- The sandbox mounts `equations_bench.py` (sandbox `bwrap-v6-bench`).
- **Import hook fix:** the sandbox import hook now lets numpy, scipy, and torch import their own submodules lazily
  from C code. Examples are `torch.cuda` on first CUDA use and `numpy._core._methods`. Before this fix, any agent code
  that created a CUDA tensor failed with "Import of 'torch.cuda' is not permitted". The earlier student runs never
  tried CUDA, but GPT-OSS hit the numpy case 15 times. Agent-written imports are still checked statically against
  the allowlist.

## Results — October 17, 2026 (h006, H100)

Consolidated in `runs/code-agent-results-pdebench-20261017/`:

| File | Contents |
| --- | --- |
| `pdebench_tables.md` | overview and per-case tables for every split |
| `summary.md`, `*.csv` | shared report |
| `score_vs_round.png` | validation score of each class vs round |

The task folders are in `tasks/pdebench/`. Re-scoring every saved frozen solution reproduces the recorded metrics
(288 checked, 0 mismatched). No sealed case was used by any arm.

Settings: RSI ran 6 rounds with 3 attempts per round, from a scratch start. Opus 5.5 ran with a 64k output cap;
the other models used 32k. With 32k, all 18 of Opus's calls in the old iterated single arm ended at the cap
while still reasoning; that run is kept as `runs/aborted-pdebench-opus55-32k-20261017` and is excluded.

The **single run** is one model request with the information RSI sees in round 1, and its solver is frozen as
returned (redone on 2026-10-20; see `docs/code-agents.md`). The Opus single run used Opus 5.5's 128k output
maximum, because at 64k the answer ended before any code. Its 128k answer imported `time`, which the static
policy rejects, so it scores 0. The earlier iterated fixed-workflow arm is archived in
`runs/archive-fixed-workflow-20261020/`.

Passed cases, single run / RSI:

| Model | Training (16) | Validation (16) | Sealed test (17) |
| --- | --- | --- | --- |
| Opus 5.5 | 0 / 14 | 0 / 16 | 0 / 16 |
| GPT-6.1 | 14 / 13 | 11 / 16 | 14 / 13 |
| GPT-5.5 | 15 / 12 | 2 / 11 | 1 / 8 |
| Sonnet 4.5 | 0 / 3 | 0 / 3 | 0 / 2 |
| GPT-OSS | 0 / 0 | 0 / 0 | 0 / 0 |

The same comparison for the 2 h runs (`runs/code-agent-results-pdebench2h*-20261017/pdebench_tables.md`):

| Model | 2 h | 2 h, 100 % noise | 2 h, hidden time dependence |
| --- | --- | --- | --- |
| Opus 5.5 | 15/15/16 vs 14/16/16 | 0/0/0 vs 3/7/2 (single: `time` import) | 15/15/16 vs 14/16/15 |
| GPT-6.1 | 14/15/16 vs 14/16/14 | 5/2/2 vs 7/8/8 | 13/10/11 vs 14/15/13 |
| GPT-5.5 | 10/5/2 vs 14/14/13 | 3/1/1 vs 6/4/2 | 9/4/6 vs 12/9/8 |
| Sonnet 4.5 | 0/0/0 vs 0/0/1 | 0/0/0 vs 0/0/0 | 0/0/0 vs 0/0/0 |
| GPT-OSS | 0/0/0 vs 0/0/0 | 0/0/0 vs 0/0/0 | 0/0/0 vs 0/0/0 |

Each cell is single run vs RSI, as training/validation/sealed passes (of 16/16/17).

Two patterns stand out:

- A capable model (Opus 5.5, GPT-6.1) often solves the clean benchmark in one request; RSI's gain is on
  the validation and sealed cases of GPT-5.5 and on the noisy data.
- One-shot failures are mostly code-policy violations (forbidden imports such as `time` or `warnings`, `getattr`,
  over-long programs) and malformed JSON. RSI recovers from these through retries and feedback.

Findings:

- **No frozen solver ever chose PINN.**
- **The 2-D wave case fails for every model.** Its space-time error is about 1e-3, but the final-time error is about
  3e-3, above the 2e-3 tolerance.
- **The steep layers are the hardest cases.** Case 1c with α = 1000 is solved only by GPT-6.1 (both arms) and Opus
  RSI. Case 3b at Pe = 200 is solved only by Opus.
- **Continuation test:** the evolved workflow beats the initial one only for Opus, and by very little (advantage
  2.9e-4 in macro relative L2). For the other four models the advantage is 0.
