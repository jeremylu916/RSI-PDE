"""RSI-PDEbench v1: what an entrant submits, what it may use, and how it is scored.

An entrant is a self-improvement harness. For one track and one fixed foundation model it develops a solver
program, solve(problem), under a fixed budget; it sees only the track's training and validation cases. The
benchmark scores the solver it retains after every validation submission on a hidden test split that no entrant
and no development decision ever sees.

Headline numbers, per track and model:
    held-out pass rate      fraction of hidden-test cases passed by the final retained solver (after N rounds)
    gain over single run    held-out pass rate minus that of the single-run entrant (one model call, same model)
    dev-test gap            validation pass rate minus held-out pass rate of the same solver (overfitting)
A round is one gated validation submission; the curve gives the held-out pass rate after each round.
"""

import os

BENCH_VERSION = "rsi-pde-bench-v1"

# Tracks: identification-then-solve suites. The manufactured code-protocol families (equation given) are not a
# track: the calibration probes solve 10 of their 11 templates on every variant (docs/rsi-pde-bench.md).
TRACKS = {
    "tex": {"protocol": "pdebench", "title": "pde_benchmarks.tex Problems 1-11"},
    "ext": {"protocol": "pdebench-ext", "title": "pde_benchmarks.tex Problems 12-16"},
    "inventory": {"protocol": "inventory", "title": "PDE Data Inventory"},
}

# Cases per template in each split. train: labels and errors on request (unlimited); val: gated submissions
# (errors only); test: hidden, drawn from a secret seed, scored by the benchmark only.
PER_TEMPLATE = {"train": 1, "val": 1, "test": 5}

# The budget of one entrant run, the same for every entrant and model. Chosen above the usage of the 24 matched
# 2-hour runs of 2026-10-18/20 (harness RSI: 10-35 calls; Claude Code / Codex: 39-155 calls, up to 22M prompt
# and 152k output tokens), so that it binds only a run that uses far more than they did.
BUDGET = {
    "wall_seconds": 7200,             # improvement time; final scoring is outside it
    "model_calls": 400,               # requests that reach the model (metering proxy)
    "prompt_tokens": 40_000_000,      # input + cache read + cache write
    "output_tokens": 600_000,
    "validation_submissions": 30,     # gated submissions of the full validation split
    "solver_cpu_cores": 1,            # one core and one GPU share for the solver sandbox
}

# The acceptance gate of validation submissions is the harness gate (pde_rsi.evaluator.acceptance_gate):
# macro relative L2 drop > 0.5 %, no class mean +5 % or lost passes, no case +25 %, every case executes.
# The retained solver is the last accepted submission; it starts as the stub that raises NotImplementedError.

if os.environ.get("RSI_PDE_BENCH_SMOKE"):  # integration test of the pipeline only; never a benchmark result
    BUDGET = {**BUDGET, "wall_seconds": 600, "model_calls": 8, "prompt_tokens": 2_000_000, "output_tokens": 100_000,
              "validation_submissions": 2}

MODELS = {"opus55": "us.anthropic.claude-opus-5-5", "gpt55": "us.openai.gpt-5.5"}
