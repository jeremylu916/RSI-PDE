# Harness v2: modular programs, patch edits, per-part gate, run-time self-checks (2026-10-20)

Enable it with `code-agents --protocol pdebench|inventory --modular`. It applies to both arms (single run and RSI).
Harness v1 is unchanged without the flag. Code: `src/pde_rsi/modular.py`; tests: `tests/test_modular.py`.

## Why

The matched studies showed where RSI lost to Claude Code and Codex:

| Finding | Evidence |
| --- | --- |
| Solvers had no accuracy margin | Opus RSI's inventory sealed misses were just above the tolerance (final-time 2.4e-3 against 2e-3), using 2–8 s of 300 s. |
| GPT-5.5 picked the wrong class more often | 29 sealed misses against 21 for Codex. |
| The gate rejected most candidates | 71 % for GPT-5.5. |
| Whole-file rewrites | They caused 64k output truncations and 30,000-character rejections, and cost $13–19 per run. |

## What changes

1. **Modular program.** The program is a dict of parts, assembled in this order:
   1. a trusted prelude (`SOLVERS = {}`);
   2. `common` (optional helpers);
   3. one part per library class (`solve_<class>(problem, model)`, registered in `SOLVERS`);
   4. the router `identify`, which defines `solve(problem)` and dispatches.

   Each part is checked by the static policy separately and may be up to 30,000 characters. Top-level names must be
   unique across parts.
2. **Patch edits.** A proposal item either replaces or creates a part (`code`) or patches it (`edits`: exact
   old/new pairs, each old text occurring once). Parts not listed stay unchanged.
3. **Per-part gate.** When a candidate changes only class parts, routing is unchanged, because identification lives
   in the router. Each changed part is gated on the validation cases routed to it (by the reported class), using the
   unchanged acceptance gate. Accepted parts are merged into the active program, and the merged validation results
   are exact (no re-run). A change to `identify` or `common` is gated on all cases, as before.
   - A still-failing case elsewhere no longer vetoes an improvement to an unrelated class.
4. **Run-time self-checks.**
   - Solvers get `problem.elapsed_seconds()` (sandbox `bwrap-v8-clock`; `time` stays banned).
   - The contract recommends:
     - choosing the class and coefficients by data misfit (m ≈ 1 for the true equation);
     - refining until the estimated error is below tolerance/5, or about half of `max_seconds` is used.
   - The starting workflow asks for targeted patches and a safety margin.

The evaluator, tolerances, splits, budgets and model settings are unchanged.

## Fresh hidden splits

`test`, `test2` and `inventory_test` were used to diagnose the harness, so v2 is evaluated on new splits:

| Split | Cases | How it was made |
| --- | ---: | --- |
| pdebench `test3` | 20 | Every tex case once. New numbers for the literature cases; manufactured cases drawn from the split name. |
| inventory `inventory_test2` | 12 | DiffusionPDE instance 3 for the dataset families, new draws for the generated ones, Burgers-Gaussian D = 0.12, convection–diffusion pulse at x₀ = 7. |

References for `inventory_test2` come from `scripts/build_inventory_references.py --splits inventory_test2`. Neither
split is shown to any agent or used for any development decision.

## Test

- `scripts/run_v2_study.sh` runs v2 for Opus 5.5 and GPT-5.5 on both suites, 3 runs each, with the matched budgets
  (2 h, $25 cap, 30 validation submissions).
- `scripts/score_fresh_splits.py` scores every v1 harness arm, every native-agent solver, and every v2 arm on the
  fresh splits.
- `scripts/report_fresh_comparison.py` writes the comparison.
- **Caveat:** the native agents were not given the v2 self-check guidance. A stricter comparison would also re-run
  them with the same contract text.

## Results (2026-10-20, h008)

Each cell is the mean number of hidden cases passed per run (3 runs), with the cost per run. Every configuration's
frozen solver was scored on the same fresh split.

**pde_benchmarks.tex, `test3` (20 cases):**

| Model | v1 single | v1 RSI | v2 single | **v2 RSI** | Native agent |
| --- | ---: | ---: | ---: | ---: | ---: |
| Opus 5.5 | 19.7 · $1.6 | 20.0 · $18.9 | 19.3 · $1.9 | **20.0 · $14.9** | 19.7 · $6.3 (Claude Code) |
| GPT-5.5 | 7.7 · $0.4 | 9.7 · $17.8 | 3.3 · $0.4 | **11.0 · $12.2** | 12.0 · $14.9 (Codex) |

**Inventory, `inventory_test2` (12 cases):**

| Model | v1 single | v1 RSI | v2 single | **v2 RSI** | Native agent |
| --- | ---: | ---: | ---: | ---: | ---: |
| Opus 5.5 | 0.0 · $2.3 | 11.3 · $14.0 | 0.0 | **12.0 · $13.5** | 12.0 · $6.0 (Claude Code) |
| GPT-5.5 | 5.3 · $0.35 | 9.3 · $13.0 | 5.3 · $0.4 | **9.0 · $5.8** | 9.7 · $9.7 (Codex) |

What the results show:

- **Opus 5.5:** v2 RSI ties Claude Code on both suites. It passes every hidden case in all six runs: 20/20 and 12/12.
- **GPT-5.5, tex suite:**
  - v2 RSI improves on v1 RSI (11.0 against 9.7) at about 30 % lower cost.
  - It is 1 case behind Codex on average. Its runs scored 6/15/12 against Codex's 12/14/10, so the difference is
    within the run-to-run spread.
- **GPT-5.5, inventory:**
  - v2 RSI is level with v1 RSI (9.0 against 9.3) at less than half the cost.
  - It is 0.7 behind Codex, and it is the most consistent configuration (9/9/9).
- **Cost:** v2 RSI is the cheaper RSI on both suites.
- **The per-part gate is working:** GPT-5.5 accepted about 60 % of candidates, against 29 % under v1.
- **Modular single runs are worse.** One request must write every part:
  - GPT-5.5 tex: 3.3 against 7.7.
  - Opus inventory: the answer was cut off even at the 128k output cap.
  - Modular programs pay off only with iteration.
- **Not yet better than the native agents.** The remaining GPT-5.5 gap is in identification and accuracy margin on
  heat, convection–diffusion and fractional problems.

Limits:

- There are only 3 runs per configuration.
- The native agents did not get the v2 self-check guidance.
- The Opus single runs used the 128k output cap, both v1 (re-requested) and v2; the RSI arms and native agents used
  64k.

Tables: `runs/v2-fresh-comparison-20261020/fresh_comparison.md` and `fresh_comparison_long.csv`.

## Single run with one rule repair (2026-10-21)

Every strict Opus 5.5 single run on the inventory suite scored 0, because its one answer never ran:

- at the 64k output cap, every answer was cut off (v1 and v2);
- at 128k, v1 answers broke the 30,000-character program limit or used `import time`;
- at 128k, v2 answers were cut off again, because one answer must write every class part.

`code_agents.repair_one_shot` (`redo_single_one_shot.py repair` / `score-repair`) adds the single run with one rule
repair, written to `<run>/single_repair/`:

- If the single answer was rejected before execution, or was cut off, only that rejection message is sent back once.
- One corrected answer is accepted.
- No solver runs before the repair, so it carries no information about results.
- The strict single run stays in `<run>/single/`.
- All 18 other single runs in the comparison had valid first answers, so only the 6 Opus inventory runs change.

Opus 5.5, inventory hidden cases passed per run (of 12):

| Configuration | Fresh `inventory_test2` | Earlier `inventory_test` |
| --- | ---: | ---: |
| v1 single (strict) | 0 / 0 / 0 | 0 / 0 / 0 |
| v1 single + repair | 10 / 8 / 9 (mean 9.0) | 9 / 7 / 8 |
| v2 single (strict) | 0 / 0 / 0 | — |
| v2 single + repair | 11 / 10 / 11 (mean 10.7) | — |
| v2 RSI / Claude Code | 12 / 12 / 12 | — |

Diagnostic, outside the protocol (`scripts/diagnostic_score_rejected.py`): the three over-long v1 programs, run exactly
as written, pass 12, 0 and 7 fresh cases. Run 2 also imports `time`, so it crashes when run. Results page:
`reports/results-page/` (collect.py, then build.py).
