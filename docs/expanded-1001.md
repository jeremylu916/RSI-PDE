# Expanded release: 1001 cases (`src/pde_rsi/expanded.py`)

Every benchmark problem of the project with new coefficients, initial data and boundary data (added 2026-10-23). A
variant keeps its template's equation class and geometry type. It redraws:

- the coefficients;
- the initial data: profile, modes, amplitudes, DiffusionPDE instance, or random field;
- the boundary values, where the problem has Dirichlet data.

Variants come from a seeded generator (`expanded.VERSION = "expanded-1001-v1"`), so the release is reproducible.

| Suite | Templates | Variants each | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pdebench` (tex Problems 1–11) | 20 | 20 | 280 | 60 | 60 | 400 |
| `pdebench_ext` (tex Problems 12–16) | 10 | 20 | 140 | 30 | 30 | 200 |
| `inventory` | 12 | 20 | 168 | 36 | 36 | 240 |
| `manufactured` 1-D families | 6 | 15 | 66 | 12 | 12 | 90 |
| `manufactured` 2-D wave | 1 | 15 | 11 | 2 | 2 | 15 |
| `manufactured` group FPDE | 4 | 14 | 40 | 8 | 8 | 56 |
| **all** | **53** | | **705** | **148** | **148** | **1001** |

Within each template, the first variants go to train, the next to validation, and the last to test (`SPLIT_SIZES`).

## What varies

- **pdebench, manufactured cases:**
  - The domain length, amplitude, D, v, beta, the fractional orders and the spatial profile.
  - Dirichlet data: in 3 of 4 variants of 1a, 2a, 3a, 4a, 7 and 8, the solution gets a linear lift
    `b0 + (b1 − b0)(x − a)/L`. `equations_bench.LIFTED` and `lift()` adjust the forcing, which covers the extra
    `v·slope` and Burgers terms.
- **pdebench, literature cases:** they redraw their physical numbers:
  - the Peclet number and the boundary-layer amplitude `U` (3b, U ≠ 1 in 2 of 3 variants);
  - the front sharpness and position (1c);
  - the Gaussian pulse (3d);
  - the viscosity and the shock states (4b, 4c);
  - the time power p and the order alpha (6, 9–11).
- **pdebench_ext:**
  - KS: the domain, amplitude and horizon. For 12b, also the amplitude and shift (`ic_amp`, `ic_shift`) of the
    Kassam–Trefethen data.
  - Taylor–Green: k and nu, with e^(−2νk²T) ≥ e^(−4). Kovasznay: Re from 10 to 100.
  - Cahn–Hilliard: M, epsilon and L. For 14b, five new initial modes.
  - Maxwell: the cavity size, the mode and the wave speed.
  - Curl-curl: the mode numbers and k. For 16b, k lies inside a gap between Maxwell eigenvalues.
- **inventory:**
  - The DiffusionPDE families (Burgers random, Darcy, Poisson, Helmholtz, Navier–Stokes) use test instances 4–23:
    `scripts/extract_inventory_inputs.py --instances 4-23` writes `data/expanded/inventory_inputs.npz`. The spec
    records the instance as `dataset_instance`.
  - The Gaussian Burgers data take `ic_center` and `ic_width`.
  - The other families redraw their coefficients and random fields from a per-variant seed.
- **manufactured:** the length, horizon, coefficients, amplitude, mode, profile, time exponent and orders.

## Checks

`tests/test_expanded.py` checks:

- 1001 cases;
- splits of 705 / 148 / 148;
- no variant repeats any existing case of the project, public or sealed (`sealed_overlap()`; gridded inputs are
  compared by content hash);
- all variants are distinct;
- the lifted Dirichlet cases satisfy their PDE by finite differences.

Exact submissions pass the evaluators for every closed-form case.

## References

Cases without a closed form get computed references:

```bash
python scripts/build_expanded_references.py --suite ext --workers 4          # 12b, 14b: ETDRK4, about 30 min
python scripts/build_expanded_references.py --suite inventory --workers 14   # 240 cases, about 14 CPU-hours
```

Each reference is checked against a half-resolution solve. The measured differences go to
`data/expanded/reference_report-*.json`.

## Release

The 1001 cases are **not** on Hugging Face (decision of 2026-10-23). The repository keeps the 148 public cases.
`EXPANDING_TO_1001.md` gives the generation commands (commit 00ff39d3; source in
`datasets/huggingface/generator-guide/`).

`scripts/build_expanded_hf.py` still builds and validates the full package locally, in
`datasets/huggingface/expanded-1001`, about 3 GB, with float64 references. With `--push`, it would upload the
package under `expanded/` with the configurations `expanded_pdebench`, `expanded_pdebench_ext`,
`expanded_inventory`, `expanded_manufactured` and `expanded_index`.

The release's test split is not one of the project's sealed test splits, which stay private.
