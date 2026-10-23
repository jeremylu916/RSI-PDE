# Initial implementation validation

Verified on Gautschi on October 14, 2026. Python 3.13.5, NumPy 2.1.3,
SciPy 1.15.3. Small local checks used one numerical thread; no Slurm job or
foundation-model API call was made. This is setup validation, not a research result.

`bash scripts/test.sh` passed **17 tests**. The test suite covers:

- Analytic forcing verified using independently differentiated references for all
  local families and spatial templates.
- Fractional forcing checked against the defining Caputo integral, including
  endpoint singularities; L1 exactness on a linear function on graded meshes.
- Error reduction under refinement in all six families; improved fractional
  initial-singularity accuracy with a graded mesh.
- Detection of nonfinite output, wrong IC/BC, family/instance regressions, and
  resource failures; solve-call and wall ceilings, failure accounting, and
  resource-limited child execution.
- Template grouping and deduplication, version round trips and parent hashes,
  final-budget reservation, comparison without readable sealed tasks, identical
  starting solver/memory in policy probes, and once-only frozen final evaluation.

All 48 open pilot cases were also evaluated directly to check parameter variants.
The initial solver passed 24/24 evolution cases and 21/24 development cases.
Their macro relative L2 values were 0.00080613249 and 0.0012314430, respectively.
This direct accuracy check did not meter subprocess resources, so these pass counts
are numerical validation rather than a complete resource-certified benchmark result.

The budgeted smoke comparison selected one variant per family from evolution and
development, ran two candidate cycles, and reserved six final calls per arm.
No final-test scores were generated.

| Arm | Accepted updates | Development macro relative L2 | Consumed solve calls |
| --- | --- | --- | --- |
| Initial single baseline | 0 | 0.0010041837 | 6 |
| Fixed improvement | 2 | 0.00012805306 | 30 |
| Recursive policy scaffold | 2 | 0.00020166509 | 30 |

The paired continuation started from the same accepted RSI solver and memory,
held each procedure fixed, and ran two future candidate cycles in each branch.
Both completed and consumed exactly 30 solve calls under the same 120-second ceiling.
The initial procedure finished with macro relative L2 0.000048252171; the evolved
procedure finished with 0.00017954610. Future error reductions were 0.00015341292
and 0.000022118993. Evolved-policy advantage was **-0.00013129392**.

Thus, the initial numerical policies improve solvers, while this small development
continuation offers **no evidence of an RSI advantage**. Actual grid costs and
elapsed time differ despite equal solve counts and ceilings. No statistical or
sealed-test claim follows from this deterministic smoke check.

Artifacts within the project:

- `runs/test-validation.log`
- `runs/open-pilot-evolution-accuracy.json`
- `runs/open-pilot-development-accuracy.json`
- `runs/validated-smoke/development_comparison.csv`
- `runs/validated-smoke/manifest.json` and per-arm snapshots/events/metrics
- `runs/validated-continuation/continuation_comparison.csv`
- `runs/validated-continuation/manifest.json` and paired continuation traces

Source hashes in the validated smoke manifest record the v0.1 numerical package;
the v0.2 continuation below has new source hashes and a separate environment lock.
The earlier `runs/smoke-comparison/` folder is an implementation-time preliminary
trace; use the `validated-*` runs for review. Final-test protocol tests used
temporary mocked solver results; they did not score the project final split.

## Depot environment and v0.2 continuation

Verified on October 14, 2026 using the dedicated conda interpreter
`/depot/lin491/data/Binghang_project/PDE-RSI/envs/pde-rsi/bin/python`.
Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3 match the original numerical stack;
43 conda packages form the installed dependency closure. Exact builds and archive
checksums are recorded in `configs/conda-explicit.txt`. The portable environment
specification permits compatible versions; scientific runs freeze their actual lock.

`bash scripts/test.sh` passes **25 tests**. Added checks cover a single-session
control using identical candidate slots, an explicit static control, insufficient
requested-call budgets, aggregate CPU exhaustion, success/timeout CPU accounting
and ledgers, NPZ round trips and invalid/object arrays, the dedicated interpreter,
project-local temporary files, and home/symlink output rejection.

A real minimal offline conda transaction additionally checks that the home
environment catalog's contents and modification timestamp stay identical. Conda
25.5.1 initially touched that existing 204-byte file even with `register_envs=false`;
`scripts/conda.sh` now redirects transaction verification to the project catalog.
The wrapper's transaction check passes without changing global conda or shell files.

The home audit found approximately 9.2 GiB: VS Code server files account for
approximately 4.0 GiB and the pre-existing `muon` environment for 3.9 GiB. No
PDE-RSI-specific home environment was found. These unrelated files were preserved;
this work does not claim to have reclaimed their space. Project conda packages,
environment, caches, and temporary files are now on depot.

The new smoke comparison uses one public variant per family, two candidate cycles,
36 allowed calls, 120 wall seconds, and 120 aggregate CPU seconds per arm.
Every arm consumes 30 calls, reserves six final calls, completes both candidate
cycles, and passes resource checks. All selected development tasks pass at freeze.
Source hashes and environment-lock hashes match the current run.

| Arm | Accepted candidates | Accepted agent updates | Development macro relative L2 | Calls | CPU seconds | Wall seconds |
| --- | --- | --- | --- | --- | --- | --- |
| Single-session numerical search | 2 | 0 | 0.00012805306 | 30 | 6.839437 | 10.616733 |
| Fixed improvement policy | 2 | 2 | 0.00012805306 | 30 | 6.808040 | 10.904810 |
| Recursive policy scaffold | 2 | 2 | 0.00020166509 | 30 | 6.815437 | 11.219113 |

The session and fixed arms retain the same solver, as expected for a deterministic
procedure that does not consult persistent memory. The single arm retains empty
reusable memory, its original improvement policy, and generation zero. This repairs
the old comparison's unequal evaluation allowances; it does not establish an LLM
single-run baseline or prove a memory effect.

A paired continuation starts from the accepted RSI solver and memory. Both branches
complete two future updates and consume 30 calls under equal 120-second wall/CPU
ceilings. The initial-policy branch reaches 0.000048252171 development macro L2
using 8.213734 CPU seconds; the evolved-policy branch reaches 0.00017954610 using
7.357213 CPU seconds. The future error-reduction advantage of the evolved procedure
is **-0.00013129392377**. This again provides no RSI advantage in the pilot;
it is an exploratory development result with different actual compute use.

The end-to-end `solve caputo-g1-v0 --nx 64 --nt 96 --grading 2` check produces a
metered, passing solution archive. Independent `evaluate-solution` reloading passes
numerical accuracy and constraints; resource certification is correctly unknown
for the submitted file alone. No final tests, model APIs, or Slurm submissions were
performed during this continuation.

Current artifacts:

- `runs/storage-doctor.json`, `runs/environment-install.log`
- `runs/test-validation-v02.log`
- `runs/depot-v02-comparison/development_comparison.csv` and `manifest.json`
- `runs/depot-v02-comparison/{single,fixed,rsi}/solve_calls.jsonl`
- `runs/depot-v02-continuation/continuation_comparison.csv` and `manifest.json`
- `runs/depot-v02-solution/solution.npz`, `metrics.json`, and `report.json`
- `runs/depot-v02-solution-check/report.json`
