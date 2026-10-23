# RSI-PDEbench v1

A benchmark of recursive self-improvement on scientific code.

An entrant is a **self-improvement harness**. For one track and one fixed foundation model, it develops a PDE
solver program under a fixed budget, using only the track's training and validation cases. The benchmark then
scores the solver the harness retained after every round on a **hidden test split**. No entrant, and no
development decision, ever sees that split. Code: `src/pde_rsi/bench/`, `scripts/bench.py`.

## Task

Each case is an identify-then-solve problem from the project's student protocol:

- 256 noisy samples (1 % RMS noise) of the solution of an unknown equation from a class library;
- black-box forcing, initial and boundary data;
- the domain and the evaluation grid.

The program `solve(problem)` must identify the class and its coefficients and return the solution on the whole
domain. A case passes when all of the following hold:

- the space-time and final-time relative L2 errors are at most the tolerance (2e-3; inventory 4-D elliptic: 5e-2);
- the initial and boundary data hold to 1e-10;
- the solve fits `max_seconds` and the memory limit.

## Tracks

| Track | Source | Templates | Train | Validation | Hidden test |
| --- | --- | ---: | ---: | ---: | ---: |
| `tex` | pde_benchmarks.tex Problems 1–11 | 20 | 20 | 20 | 100 |
| `ext` | pde_benchmarks.tex Problems 12–16 | 10 | 10 | 10 | 50 |
| `inventory` | PDE Data Inventory | 12 | 12 | 12 | 60 |

- **Templates:** a template is one tex case or inventory family. Every split has each template once (train and
  validation) or five times (hidden test).
- **Generator:** the numbers come from the samplers of the 1001-case expansion (`src/pde_rsi/expanded.py`), so the
  three splits share one distribution.
- **Manufactured families:** the code-protocol families with a given equation are not a track. The calibration
  probes solve 10 of their 11 templates on every variant (see Calibration).

## What an entrant gets

Each run gets:

- **Workspace:**
  - the task text and class library;
  - the public view and noisy data of every training and validation case;
  - a stub solver.
- **Training feedback:** unlimited. Each call runs a candidate on chosen training cases and returns its errors,
  pass/fail, and the true class and coefficients.
- **Validation submissions:** at most 30, each scored on all validation cases (errors only) and passed through the
  project's acceptance gate:
  - the macro relative L2 must drop by more than 0.5 %;
  - no class may lose passes or rise more than 5 %;
  - no case may rise more than 25 %;
  - every case must execute with exact IC/BC.

  The **retained solver** is the last accepted submission. A round is one validation submission.
- **One fixed model** through the metering proxy, under the budget below.

The run never sees the hidden test cases, their references, their seed, or their DiffusionPDE instances. All of
these live in `evaluator_private/bench/`.

## Budget (the same for every entrant and model)

| Resource | Cap |
| --- | ---: |
| Improvement wall-clock | 2 h |
| Model calls | 400 |
| Prompt tokens (input + cache read + cache write) | 40 M |
| Output tokens | 600 k |
| Validation submissions | 30 |
| Solver sandbox | 1 CPU core, a GPU share, the per-case `max_seconds` |

The caps sit above what the 24 matched 2-hour runs of 2026-10-18/20 used:

- harness RSI: 10–35 calls;
- Claude Code and Codex: 39–155 calls, up to 22 M prompt and 152 k output tokens.

A run is therefore limited by time and validation submissions, as before, and the token caps bind only a harness
that uses far more.

The **metering proxy** (`pde_rsi.bench.proxy`) is the only place model use is counted:

- The entrant gets a loopback Bedrock endpoint and a dummy credential. The proxy holds the real token, relays every
  request, and reads usage from the response (Converse, InvokeModel streams, OpenAI-compatible Responses).
- It refuses calls to other models, and any call once a cap is reached.
- Tests: `tests/test_bench_proxy.py`. A live check through the proxy matched each client's own count exactly:
  Claude Code, Codex, and this harness's model client.

## Scores

| Number | Definition |
| --- | --- |
| **Held-out pass rate** | Hidden-test cases passed by the final retained solver, after N rounds. |
| **Gain over single run** | Held-out pass rate minus that of the single-run entrant with the same model. The single run makes one model call and is frozen as returned. |
| **Dev–test gap** | Validation pass rate minus held-out pass rate of the same solver. Large values mean overfitting to the validation cases. |
| Curve | Held-out pass rate of the retained solver after each round. |

`python scripts/bench.py score` scores every distinct retained solver on the hidden split. Per-case hidden results
stay in `evaluator_private/bench/scores`.

## Splits and the hidden test

- **Train and validation:** seeded by `rsi-pde-bench-v1:train` and `:val`.
- **Hidden test:**
  - seeded by `rsi-pde-bench-v1:test:<secret>`, where the secret is a random 128-bit value in
    `evaluator_private/bench/secret.txt`;
  - its five DiffusionPDE instances per dataset family are secret draws from 200–999. Training and validation use
    instances 100 and 101; the 1001-case release uses 4–23.
- **Ids:** case ids are opaque (`case-bt07`, `case-bv12`) and ordered by a hash; task ids carry only track, split
  and a hash.
- **Overlap:** `tests/test_bench.py` checks that no bench case repeats any other case of the project, public,
  sealed or expanded.
- **Refresh:** a new release changes the version or the secret, which draws a fresh hidden test.
- **References:**
  - closed forms where they exist;
  - otherwise computed (12b and 14b: ETDRK4; inventory: the suite's reference solvers) and checked against a
    half-resolution solve. The hidden ones are in `evaluator_private/bench/references`.

## Retired splits (public since 2026-10-23)

When the code was published (github.com/jeremylu916/RSI-PDE), the earlier sealed splits became reproducible
from the public generators, because their seeds are their split names. They are therefore **retired**:

- pdebench `test`, `test2`, `test3`;
- inventory `inventory_test`, `inventory_test2`;
- the pdebench-ext `test` split;
- the manufactured reserved cases.

They keep their role in the results reported before that date. They are no longer held-out tests. The RSI-PDE
Bench hidden test is unaffected: its seed contains a secret that is never published. It is the project's only
held-out test from now on.

## Calibration (2026-10-23)

Fixed probe solvers, with no model calls, solved all 1001 cases of the expansion: 2,856 sandboxed solves on two
H100 nodes. The probes are the frozen solvers of earlier 2-hour runs:

- harness v2 RSI with Opus 5.5 and with GPT-5.5;
- Claude Code (Opus 5.5) and Codex (GPT-5.5);
- for the manufactured families, the code-protocol RSI solvers.

Code: `scripts/bench_calibrate.py`. Per-template table: `data/bench/calibration/summary.json`, from
`scripts/bench_report_calibration.py`.

**Pass rate over all variants:**

| Track | Cases | Opus RSI v2 | Claude Code | Codex | GPT RSI v2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| tex | 400 | 0.91 | 0.87 | 0.50 | 0.19 |
| inventory | 240 | 0.91 | 0.92 | 0.44 | 0.35 |
| manufactured (code protocol) | 161 | 0.98 (Opus RSI) | — | — | 0.93 (GPT RSI) |

**Tiers:** easy means a mean probe pass rate of at least 0.9; medium 0.3–0.9; hard below 0.3.

- **Identification tracks:** 5 of 42 templates are easy (tex 1a, 1b; inventory Burgers-random, Darcy, Poisson).
  The other 37 are medium, and the probes differ widely on them; for example, inventory convection–diffusion is
  0.80 for both Opus probes and 0.00 for both GPT probes. The easy templates stay in as anchors.
- **Manufactured families:** 10 of 11 are easy, so they are not a track.
- **Hardening:** seven templates were solved on every variant by the strongest probes. Their bench variants are
  harder:

  | Template | Release range | Bench range |
  | --- | --- | --- |
  | 3b | Péclet 5–150 | 50–600 |
  | 3c | Péclet 5–120 | 50–500 |
  | 6, 9, 11 | time power p ≥ max(alpha, 0.5) | p just above the order: low regularity at t = 0 |
  | Gaussian Burgers | D 0.05–0.2 | 0.02–0.05 |
  | Helmholtz | k 0.5–2 | 2.5–4.2, below the first Dirichlet eigenvalue |

  - **Feasibility check:** a separate public calibration draw (10 per hardened template; DiffusionPDE instances
    110–119) was scored by the same probes. Every hardened template is still solved by at least one probe, and
    every exact solution passes its evaluator (190 tex bench cases checked).
  - **Effect:** hardening lowers the GPT probes markedly (for example, 3c 0.50 and 0.00), but the Opus probes
    still solve most hardened variants (0.8–1.0). The remaining headroom for the strongest systems is about 10 %
    of the cases, mostly in 1c, 2b, 2c, 3d, 4b, 4c, 5 and 8 and in inventory Kolmogorov, convection–diffusion,
    ADR and Allen–Cahn.
- **ext track:** no frozen ext solver existed before the bench. Its tiers come from the step-5 baseline runs.

## Baselines

(Filled in by step 5.)

## Running

```bash
source scripts/env.sh
# one entrant run (GPU node; two cores per run)
python scripts/bench.py run --track tex --entrant pde-rsi-v2 --model opus55 --cores 42,44 --output runs/bench-v1-tex-pde-rsi-v2-opus55
python scripts/bench.py run --track tex --entrant claude-code --model opus55 --cores 46,48 --output runs/bench-v1-tex-claude-code-opus55
# hidden-test scoring of every round, then the table
python scripts/bench.py score --runs runs/bench-v1-* --cores 50,52,54,...
python scripts/bench.py report
```

`RSI_PDE_BENCH_SMOKE=1` shrinks the budget for an integration test; its runs are never results.
