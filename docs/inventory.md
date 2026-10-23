# PDE Data Inventory suite (`--protocol inventory`)

This suite adds the PDEs of *PDE Data Inventory* (`PDE_Data_Inventory_Overleaf.pdf`, X. Kong, 19 October 2026) to the
benchmark under the same student protocol as `pdebench`.

- **Task.** Each case gives 256 noisy samples (1 % RMS) of the true solution, the domain, and black-box `forcing`,
  `initial`, `boundary`, and `coefficient` functions, plus the raw gridded input arrays. The agent identifies the
  equation class and its scalar coefficients from an extended library, then returns the solution. It can use a
  numerical method or a PINN.
- **Splits.** 12 families, with one case each in training (`case-a*`), validation (`case-b*`), and a fresh sealed
  split (`case-c*`, `inventory_test`).
- **Code.**
  - `src/pde_rsi/inventory.py`: cases, view, evaluator, protocol.
  - `src/pde_rsi/equations_inventory.py`: inputs, shared with the sandbox.
  - `src/pde_rsi/inventory_reference.py`: reference solvers.
  - `scripts/extract_inventory_inputs.py` and `scripts/build_inventory_references.py`.
  - `tests/test_inventory.py`.

## Families

| Family | Library class | Layout | Inputs | Coefficients identified |
| --- | --- | --- | --- | --- |
| Burgers, fixed Gaussian IC (inv. 1.1) | burgers (periodic) | 1-D | exp(−(x−3)²/2) on [−8, 8), periodic representation; T = 10 | D: 0.1 / 0.07 / 0.15 |
| Burgers, random IC (inv. 1.2) | burgers (periodic) | 1-D | DiffusionPDE `burgers.mat` input, instances 0 / 1 / 2 (trigonometric interpolation); T = 1 | D = 0.01 |
| Linear conv.-diff. (inv. 10) | convection_diffusion | 1-D Dirichlet | Gaussian pulse (x₀, σ₀) = (5, 1) / (4, 0.8) / (6, 1.2) on [0, 20], T = 5; Dirichlet data from the whole-line solution | v = 1, D = 1 |
| Darcy (inv. 3) | darcy_2d | 2-D steady | DiffusionPDE `darcy.mat` thresholded permeability (3 / 12, cell-constant), f = 1 | — |
| Poisson (inv. 4.1) | poisson_2d | 2-D steady | DiffusionPDE `poisson.mat` forcing (bicubic spline of node values), Δu = f | — |
| Helmholtz (inv. 4.2) | helmholtz_2d | 2-D steady | DiffusionPDE `helmholtz.mat` forcing, Δu + k²u = f | k = 1 |
| Navier–Stokes (inv. 5) | vorticity_ns | 2-D periodic | DiffusionPDE `ns-nonbounded.mat` ω₀ (unit torus), q = 0.1[sin 2π(x+y) + cos 2π(x+y)], T = 1 | ν = 1e−3, μ = 0 |
| Kolmogorov (inv. 6) | vorticity_ns | 2-D periodic | ω₀ ~ N(0, 7^{3/2}(−Δ+49)^{−5/2}) on the 2π torus (128²), q = −4 cos 4y, T = 1 | ν = 1e−3, μ = 0.1 |
| Allen–Cahn (inv. 7) | allen_cahn_2d | 2-D periodic | ξ ~ U[−1, 1]⁵⁷, κ(x, y, ξ), inventory u₀, T = 1 | λ = 0.8 + 0.2ξ₅₇ |
| 2-D stochastic Burgers (inv. 2) | burgers_2d | 2-D periodic | ξ ~ U[−1, 1]²⁰, κ, f, u₀ from the inventory, T = 0.25 | c_x, c_y |
| ADR (inv. 8) | adr_2d | 2-D periodic | m = 4 (d_ξ = 12) with the inventory's mixed law, κ_m, f = γ f₀, T = 2 | c_x, c_y, λ |
| 4-D elliptic (inv. 9) | elliptic_4d | 4-D points | â from a Matérn-3/2 field on 65², a(x) = â(\|x′\|², z), s₀; output at the 103,845 cell centres | — |

## Choices not fixed by the inventory or DiffusionPDE

Each choice is recorded in the case spec:

- **Kolmogorov:** horizon T = 1, against 10 s on a 2048² grid in the inventory. Long horizons are chaotic, so no
  solver can match a reference pointwise.
- **Linear convection–diffusion:** Gaussian-pulse initial data. The inventory leaves the profile unspecified.
- **2-D Burgers:** forcing modes normalised to unit L² norm on the torus.
- **4-D elliptic:** â is a unit-variance, zero-mean Matérn field.
- **ADR:** m = 4.
- **Validation and sealed cases:** these use different coefficients or instances from training, as in `pdebench`.

## References and tolerances

References are computed at high resolution from the same continuous inputs the agent sees (data/inventory/references,
evaluator only):

| Family | Reference solver |
| --- | --- |
| 1-D Burgers | Pseudospectral ETDRK4, 1024–2048 modes |
| Darcy | Finite volumes on the 8× refined grid (1024²) |
| Poisson and Helmholtz | Sine series on 1020² nodes |
| Navier–Stokes and Kolmogorov | Pseudospectral integrating-factor RK4 |
| Allen–Cahn, 2-D Burgers, ADR | Pseudospectral RK4 |
| 4-D elliptic | Cell-centred finite volumes on the 63⁴ masked grid |
| Linear convection–diffusion | Closed form |

Each reference is checked against a coarser solve (`data/inventory/reference_report.json`). The pass tolerance is
2e-3 unless that check requires a looser one, in which case the tolerance is at least 5× the reference error.

Measured (coarse vs fine): Burgers ≤ 1e−8, Poisson ≤ 1.9e−5, Helmholtz ≤ 2e−7, Darcy ≤ 3.5e−4, Navier–Stokes
≤ 1.4e−7, Kolmogorov (512² vs 256²) ≤ 4.4e−6, Allen–Cahn ≤ 1e−12, 2-D Burgers ≤ 4e−16, ADR ≤ 1.1e−6. These
families use 2e−3.

The 4-D problem is the exception. The staircased 63⁴ reference differs from 21⁴ by 2.1–2.8 % and from an
interpolated 45⁴ solve by 0.9 %, so its own error is about 1 % at most. Its tolerance is 5 %. A correct solve on
the inventory's 21⁴ grid passes, so this case tests the setup: the masked domain, the coefficient mapping, the
source, and the Dirichlet condition.

Checks against the published data:

- **Burgers:** the reference matches DiffusionPDE's Burgers trajectories to 7e-7.
- **Navier–Stokes:** it matches to 4.6e-4 at t = 1, which is the dataset's own time-stepping error. This holds once
  the dataset's axis 0 is read as x.
- **Poisson:** it matches to 1.4e-4, the dataset's 128² finite-difference error.
- **Darcy:** it differs by about 1 %. The dataset spline-smooths the 3 / 12 permeability before solving; this suite
  uses the piecewise-constant field, as given to the agent.

## Matched study (October 20, 2026)

The study compares a single run (one model request, solver frozen as returned), the RSI harness, and the
native coding agent. It uses Opus 5.5 with Claude Code and GPT-5.5 with Codex, 3 runs each. RSI and the native
agents get 2 h, a $25 spending cap and 30 gated validation submissions.

Results are in `runs/matched-inv-study-results-20261020/`:

- `summary.md`: findings;
- `matched_study.md`: per configuration and per run;
- `per_family.md`: per family.

Sealed cases passed (of 12), mean over 3 runs:

| Model | Single run | RSI | Native agent |
| --- | ---: | ---: | ---: |
| Opus 5.5 | 0.0 | 10.7 | 12.0 (Claude Code) |
| GPT-5.5 | 5.7 | 8.0 | 9.0 (Codex) |

The Opus single run fails for a structural reason. At the harness's 64k output cap its answer ended before any
code. At Opus's own 128k maximum each program exceeded the 30,000-character limit that every configuration has.
