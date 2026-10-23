# PDE-RSI

Single-session and recursive improvement-policy agents now use fixed foundation
models through Amazon Bedrock. The user resumed this phase on October 14, 2026.
The models select numerical settings for trusted classical Python solvers; the
independent evaluator calculates absolute and relative L2 errors. See the
[agent experiment protocol](docs/agent-experiments.md) for inputs, budgets,
acceptance criteria, and interpretation.

A **2D wave family** (`u_tt - D(u_xx+u_yy) = f` on a square, 4 train, 4 validation, and 4 reserved
cases) is stored separately in `benchmarks/wave2d/`; see the [problem catalog](docs/problem-catalog.md).
It is not part of the uploaded Hugging Face dataset. The **group FPDE examples** (FODE,
Caputo FPDE, time-variable-order FPDE, space-time-variable-order FPDE; 8 train, 8 validation, and 4
reserved cases) are in `benchmarks/group_fpde/` and are selected with `--families group`.

The 1D experiments use the **48 public manufactured problems** in
[JeremyLu1017/PDE-RSI](https://huggingface.co/datasets/JeremyLu1017/PDE-RSI): Poisson,
heat, convection–diffusion, viscous Burgers, Caputo subdiffusion, and multi-term
Caputo diffusion. Each family has four train/evolution and four
validation/development cases. The [problem and solver catalog](docs/problem-catalog.md)
indexes their equations and implementations. Published benchmark imports and
PINN/FPINN solvers remain future work.

All project storage belongs in `/depot/lin491/data/Binghang_project/PDE-RSI`,
including the conda environment, downloaded packages, model caches, temporary
files, and experiment artifacts.

## RSI-PDE Bench v1

[RSI-PDE Bench](docs/rsi-pde-bench.md) (2026-10-23) scores self-improvement harnesses.

- **What runs:** for one track and one fixed model, a harness develops a solver under a fixed budget: 2 h, 400
  model calls, 40 M prompt and 600 k output tokens, and 30 gated validation submissions. Every model call goes
  through a metering proxy.
- **What is scored:** the solver retained after each round, on a hidden test split.
- **Headline numbers:** the held-out pass rate after N rounds, the gain over a single run, and the dev–test gap.
- **Tracks:** `tex` (Problems 1–11), `ext` (Problems 12–16) and `inventory`, with 294 cases in all. The hidden 210
  are kept in `evaluator_private/bench`.
- **More:** [datasheet](docs/rsi-pde-bench-datasheet.md); containers in `containers/`; the public-split evaluator
  is `rsi-pde-bench-eval`.
- **Retired splits:** the earlier sealed splits (pdebench test/test2/test3, inventory_test/test2, the ext test
  split) became public with this repository. They are retired, and only the bench's secret hidden test is held out
  ([details](docs/rsi-pde-bench.md#retired-splits-public-since-2026-10-23)).

```bash
python scripts/bench.py run --track tex --entrant pde-rsi-v2 --model opus55 --cores 42,44 --output runs/bench-v1-...
python scripts/bench.py score --runs runs/bench-v1-* --cores ...   # hidden test, every round
python scripts/bench.py report
```

## Run the code-writing agents

In the current protocol, the frozen model writes the solver programs: one Python
`solve(problem)` module per family. Each module runs in a bubblewrap sandbox,
is scored by the unchanged evaluator, and is accepted or rejected by the same
gate. The single run (one model request, solver frozen as returned) is compared with the RSI arm
(gated rounds with inherited workflow and memory); see [code agents](docs/code-agents.md).

```bash
cd /depot/lin491/data/Binghang_project/PDE-RSI
bash scripts/run.sh code-agents --model-config configs/agent_models/claude_code.json \
  --start scratch --rounds 6 --probe-rounds 2 --output runs/my-code-agents
```

`--families` selects the six 1D families, the 2D wave family (`wave2d`), or `all`.
With `--start classical`, both arms begin from the classical solvers. With
`--start scratch`, the agent must write every solver itself. See
[code-writing agents](docs/code-agents.md) for the solver contract, isolation,
budgets, and outputs.

## PDE identification from data

With `--protocol identification`, the agent instead receives noisy data from a hidden equation and
a library of candidate equations. It must choose the equation, estimate its coefficients, and write
the solver. Evaluation is unchanged. See [identification](docs/identification.md).

With `--protocol pdebench`, the same student-style task (scratch only, numerical or PINN on the GPU)
runs on the 20 cases of `pde_benchmarks.tex`, with a sealed test scored after freezing. See
[pdebench](docs/pdebench.md); case folders are in `tasks/pdebench/`.

## Run the settings-selection model agents

```bash
cd /depot/lin491/data/Binghang_project/PDE-RSI
bash scripts/run.sh agents --model-config configs/agent_models/gpt_oss.json \
  --rounds 3 --probe-rounds 1 --max-model-calls 3 --max-calls 216 \
  --wall-seconds 900 --cpu-seconds 600 --output runs/my-gpt-oss-agents
bash scripts/run.sh agents --model-config configs/agent_models/claude.json \
  --rounds 3 --probe-rounds 1 --max-model-calls 3 --max-calls 216 \
  --wall-seconds 900 --cpu-seconds 600 --output runs/my-claude-agents
```

The [verified October 14 results](runs/model-agent-results-20261014/summary.md)
report all four model/arm comparisons, per-case L2 errors, matched actual
budgets, and paired continuations; see the
[results table and interpretation](docs/agent-experiments.md#verified-public-results--october-14-2026).

Use a fresh output directory for each comparison. The configured models are
`openai.gpt-oss-120b-1:0` and `us.anthropic.claude-sonnet-4-5-20250929-v1:0`.
Each comparison holds its requested model fixed across the single-session arm,
RSI arm, and two continuation branches. Configurations read the project-private
credential file at request time; credentials are excluded from artifacts.
The HTTP backend issues one request per call, with no retries, redirects, or
fallback models. These commands invoke hosted model inference.

`single` retains solver revisions and feedback within one session while keeping
its improvement workflow fixed. `rsi` may also inherit a proposed workflow and
memory lesson after the numerical candidate passes validation. The continuation
branches start from the same frozen RSI solver and memory, comparing the initial
and evolved workflows under equal declared allowances.

Each main arm allows three model requests and 216 numerical solves. Each
continuation allows one request and 120 solves. All arms have 900 wall seconds,
600 local CPU seconds, one numerical-library thread, and zero local GPUs.
Actual calls, token usage, elapsed time, and CPU usage are recorded separately.
Model requests cap output at 4,000 tokens and input at 180,000 characters.

The output directory contains:

- `comparison.csv`, `per_family_l2.csv`, and `per_case_l2.csv` for frozen public evaluation.
- `continuation/comparison.csv` for the paired future-improvement comparison.
- `manifest.json` with configuration, source/data hashes, budgets, and completeness checks.
- Per-arm proposals, gate decisions, agent snapshots, model request/response audits,
  `solve_calls.jsonl`, numerical results, and solution NPZ files.

All 48 final public measurements are selection/development evidence. Reserved
final cases are outside this experiment. A lower current L2 error alone does not
establish recursive improvement; the paired continuation tests the future
improvement procedure.

## Environment and storage

```bash
cd /depot/lin491/data/Binghang_project/PDE-RSI
bash scripts/setup_env.sh
bash scripts/run.sh doctor
bash scripts/test.sh
```

The dedicated interpreter is `envs/pde-rsi/bin/python`.
`configs/conda-explicit.txt` records installed builds and checksums.
`scripts/env.sh` routes caches and temporary files into `.runtime/`, disables
user-site packages and new bytecode writes, and fixes numerical libraries to one
thread. Launchers always use the project interpreter.

Use `bash scripts/conda.sh … --prefix "$PWD/envs/pde-rsi"` for later conda changes.
The wrapper stores conda's environment catalog in `.runtime/conda/` and handles
conda 25.5.1 touching the home catalog even when registration is disabled.
All CLI paths must resolve inside the project; paths and symlinks leading outside
are rejected before execution. Current model agents use bounded settings for
trusted solver code. Arbitrary generated code would require a separate isolated
worker and private evaluator service.

## One-task numerical workflow

```bash
bash scripts/run.sh solve caputo-g1-v0 --split development \
  --nx 64 --nt 96 --grading 2 --output runs/my-solution
bash scripts/run.sh evaluate-solution caputo-g1-v0 \
  runs/my-solution/solution.npz --split development \
  --output runs/my-solution-check
```

`solve` writes the public task, settings, `solution.npz` (`x`, `t`, `u`, scalar
`method`), numerical metrics, source hash, resource report, and call ledger.
The evaluator checks dimensions, domain/time coverage, IC/BC, and accuracy without
pickle. Evaluating a submitted NPZ checks numerical accuracy; execution-resource
usage remains unknown. The model agent contract and L2 definitions are documented
in [agent experiments](docs/agent-experiments.md).

## Historical deterministic scaffold

Earlier experiments used a deterministic numerical policy scaffold. Their
`compare` and `probe` commands and historical artifacts remain available:

```bash
bash scripts/run.sh compare --limit-per-family 1 --generations 2 \
  --max-calls 36 --wall-seconds 120 --cpu-seconds 120 \
  --output runs/my-historical-comparison
bash scripts/run.sh probe runs/my-historical-comparison/rsi/frozen_agent.json \
  --limit-per-family 1 --generations 2 --max-calls 30 \
  --wall-seconds 120 --cpu-seconds 120 --output runs/my-historical-probe
```

This scaffold has `single`, `fixed`, and `rsi` arms. Its `single` searches within
one session, `fixed` persists solver and memory with a fixed procedure, and `rsi`
also proposes deterministic policy changes. `--single-mode static` reproduces the
older initial-solver baseline. The earlier numerical continuation found no RSI
advantage; [validation records](docs/validation.md) preserve that evidence.

For a frozen historical `compare` run, the separate once-per-run final step is:

```bash
bash scripts/run.sh finalize runs/my-historical-comparison
```

It verifies source, environment, benchmark, and snapshot hashes and consumes the
reserved budget without updating policies or memory. Source/environment changes
require a fresh comparison. Controlled evaluator release is required for final
scientific claims; owner-readable directories alone do not seal cases.

## Dataset and project files

The [upload-ready Hugging Face package](datasets/huggingface/pde-problems/README.md)
contains six configurations and a [local table preview](datasets/huggingface/pde-problems/preview.html).
[Packaging instructions](docs/huggingface-dataset.md) describe the uploaded layout.
The public package excludes reserved final cases. Source identifiers and pending
reuse reviews are in `benchmarks/source_registry.json`.

See the [workflow](docs/workflow.md), [research plan](docs/research-plan.md), and
[validation](docs/validation.md) for earlier methods and remaining research work.

| Location | Contents |
| --- | --- |
| `src/pde_rsi/` | Task interface, numerical solvers, model backend, agents, evaluator, CLI |
| `src/pde_rsi/classical_programs/` | Standalone classical solver programs that code-writing agents can start from |
| `envs/pde-rsi/` | Dedicated conda environment |
| `.runtime/` | Project-owned caches, conda data, temporary files, private credentials |
| `benchmarks/`, `evaluator_private/` | Public tasks and evaluator-only final cases |
| `prompts/`, `configs/`, `docs/` | Agent contracts, model/environment settings, research protocol |
| `scripts/`, `tests/` | Entrypoints, setup, Slurm template, verification |
| `runs/` | Experiment artifacts and validation evidence |
