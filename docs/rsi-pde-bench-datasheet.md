# Datasheet: RSI-PDE Bench v1

This datasheet follows the questions of Gebru et al., *Datasheets for Datasets*. The benchmark itself is described
in `docs/rsi-pde-bench.md`.

## Motivation

- **Purpose:** to measure whether a self-improvement harness, given a fixed foundation model and a fixed budget,
  turns validation feedback into a solver that generalizes to unseen cases. It does this for identify-then-solve
  PDE problems.
- **Prior state:** the earlier sealed splits of the project were small (12–20 cases). They saturated: the
  strongest configurations passed 20/20 and 12/12. They had also been inspected during harness development.
- **Creators:** the PDE-RSI project (Purdue). The project was assembled with Claude Code (Opus 5.5) at the
  project owner's direction.

## Composition

- **Case:** an instance is one PDE case, containing:
  - a hidden true equation (class and coefficients), domain, forcing, initial and boundary data;
  - 256 noisy samples of the solution (1 % RMS Gaussian noise);
  - a reference solution on the evaluation grid.
- **Size:** 294 cases. Training and validation are public (84 in all); the hidden test (210) is evaluator-only.

  | Track | Templates | Train | Validation | Hidden test |
  | --- | ---: | ---: | ---: | ---: |
  | tex | 20 | 20 | 20 | 100 |
  | ext | 10 | 10 | 10 | 50 |
  | inventory | 12 | 12 | 12 | 60 |

- **Sampling:** each split has every template of its track once (train, validation) or five times (hidden
  test). The numbers come from the 1001-case generator (`src/pde_rsi/expanded.py`).
- **Hardening:** seven templates were saturated in calibration (tex 3b, 3c, 6, 9, 11; inventory Gaussian Burgers,
  Helmholtz). Their bench variants use harder ranges (`bench/splits.py`, `harden_tex`, `harden_inventory`).
- **Labels:** the true class and coefficients are labels for training cases only. They are returned by the
  training feedback. Validation and test labels are never released.
- **Relations:** none between cases. No case repeats any other case of the project, public, sealed or expanded
  (`tests/test_bench.py`).
- **Noise and errors:** the noise is synthetic and seeded. The references are of two kinds:
  - closed forms, which are exact;
  - computed solutions, checked against a half-resolution solve. The worst differences are 2.4e-5 for 12b and 14b;
    for inventory, up to 4.2e-4 except 4-D elliptic at 2.7e-2, whose tolerance is 5e-2.
- **Confidential data:** none. There is no personal data.

## Collection

- **Synthetic sources:** pde_benchmarks.tex Problems 1–16 (manufactured and literature cases), and the PDE Data
  Inventory families (X. Kong, 19 October 2026).
- **Real inputs:** for five inventory families (Burgers with random initial data, Darcy, Poisson, Helmholtz,
  Navier–Stokes), the initial states or coefficient and forcing fields are instances of the DiffusionPDE test set
  (Huang et al. 2024). Solutions are recomputed, never copied.
- **Instance use:** training and validation use instances 100 and 101. The hidden test uses five secret draws
  from 200–999 per family. Earlier splits used 0–23, and calibration used 110–119.
- **Seeds:** every split is seeded. The hidden test seed includes a 128-bit secret.

## Preprocessing

- **DiffusionPDE orientation:** fields are transposed where the dataset stores x first (Navier–Stokes), and
  checked against the dataset's own trajectory (`scripts/extract_inventory_inputs.py`).
- **References:** computed by `scripts/build_inventory_references.py` and `scripts/build_pdebench_ext_references.py`.

## Uses

- **Intended:** benchmarking self-improvement harnesses under the fixed budget, one track and one model at a time.
  A run is judged by:
  - the held-out pass rate after N rounds;
  - the gain over the single-run entrant with the same model;
  - the dev–test gap.
- **Not intended:**
  - training on the hidden test;
  - tuning a harness on hidden-test results;
  - comparing runs made under different budgets, evaluator versions or bench versions.
- **Limitation:** the cases are synthetic and low-dimensional (1-D, 2-D, one 4-D family). They do not represent
  industrial PDE workloads.

## Distribution

- **Code:** in the PDE-RSI repository. Public splits are generated from it.
- **Hidden material:** the hidden test (secret, DiffusionPDE inputs, references, per-case scores) lives only in
  `evaluator_private/bench/`. It is never uploaded, and is scored by the maintainers on request.
- **Licence:** the DiffusionPDE inputs are under **CC BY-NC-SA 4.0**, as is the DiffusionPDE code
  (`external/DiffusionPDE/README.md`). Inventory cases derived from them inherit non-commercial, share-alike use.
  The project's own code and the synthetic tex and ext cases follow the repository's licence.

## Maintenance

- **Maintainer:** the PDE-RSI project.
- **Versions:** a new version changes `BENCH_VERSION` or the secret, which draws a fresh hidden test. An evaluator
  change bumps the evaluator version, and invalidates comparisons across versions.
- **Errata:** reported through the repository. Reference errors found later are fixed by a new version, never in
  place.
