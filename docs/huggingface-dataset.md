# Local Hugging Face dataset package

The dataset is prepared locally at
`datasets/huggingface/pde-problems/`, with a downloadable archive at
`datasets/huggingface/pde-problems.zip`. The reviewed package was uploaded on
October 14, 2026 to [JeremyLu1017/PDE-RSI](https://huggingface.co/datasets/JeremyLu1017/PDE-RSI),
preserving its existing private visibility. All 38 uploaded files were verified
against their remote Git/LFS hashes. Upload evidence is in `runs/huggingface-upload.json`.
The viewer API returned HTTP 501 because this private repository is not supported
by its current viewer access. Its message states that private dataset viewing is
supported for PRO users and Enterprise Hub organizations. Dataset files are uploaded
and verified; the local preview remains available. Visibility was not changed.

Open `preview.html` in a browser to inspect the table. Select a PDE family and
train/validation split; expand JSON cells, open a complete case by its ID, or
download the selected rows. The uploaded Hub viewer will use the Parquet files and
the README configuration metadata, rather than this local HTML preview.

Six configurations are present: Poisson, heat, convection-diffusion, Burgers,
Caputo subdiffusion, and multi-term fractional diffusion. Each has four training
and four validation cases: 48 public instances total. Training maps to evolution;
validation maps to development. Original IDs, group identifiers, and source splits
are preserved. The current release is the manufactured pilot, not imported
literature data or the variable-coefficient/2D cases in the screenshot.

## Build and validate

```bash
cd /depot/lin491/data/Binghang_project/PDE-RSI
bash scripts/setup_hf.sh
bash scripts/run.sh export-hf --output datasets/huggingface/new-release
bash scripts/run.sh validate-hf datasets/huggingface/new-release
```

Export refuses to overwrite an existing directory. Keep reviewed releases intact
and use a new directory for a rebuild. Dataset tooling, pip caches, Hugging Face
caches, and temporary files use the project conda environment and `.runtime/`.
The numerical versions remain NumPy 2.1.3/SciPy 1.15.3. Installed dataset dependencies
are recorded in `configs/huggingface-requirements-lock.txt`.

## Contents and local loading

```text
pde-problems/
  README.md                         dataset card and six HF configurations
  data/<family>/train.parquet        typed viewer data
  data/<family>/validation.parquet
  records/<family>/*.jsonl           reviewable mirrors
  features.json                     common nested feature schema
  SCHEMA.md                         field conventions
  solver_bundle/                    portable public numerical code and runner
  preview.html                      offline family/split table
  manifest.json                     row counts and file checksums
```

```python
from datasets import load_dataset

package = "/depot/lin491/data/Binghang_project/PDE-RSI/datasets/huggingface/pde-problems"
poisson = load_dataset(package, "poisson")
print(poisson["train"][0]["oracle_config"]["pde"]["equation"])
```

The first columns follow the supplied screenshot: `id`, `pde_classification`,
`oracle_config`, and `evaluation_config`. Additional columns contain the readable
problem statement, solver configuration, source provenance, and native task record.
All nested values have explicit Arrow types. Nulls and empty lists remain typed.

## Numerical replay and information boundary

```bash
source scripts/env.sh
"$PDE_RSI_ENV_PREFIX/bin/python" \
  datasets/huggingface/pde-problems/solver_bundle/run_case.py \
  --data datasets/huggingface/pde-problems/data/heat/train.parquet \
  --id heat-g0-v0 --output runs/my-hf-heat-solution
```

The portable bundle includes task/settings classes, forcing/IC functions, numerical
solvers, and NPZ I/O. It excludes the pilot task generator, evaluator references,
reserved final data, and all improvement/RSI code. It can run without importing the
originating `pde_rsi` package. Its direct execution reports runtime/memory, but does
not independently certify resource limits or numerical accuracy.

`oracle_config` contains the problem/evaluator contract and reference availability,
not exact solution arrays or expressions. The 24 reserved final cases are omitted
from all Parquet, JSONL, preview, solver, and archive files. There is no public test
split. The source benchmark and sealed files are unchanged by export.

## Verification and later upload

The package loads through `datasets.load_dataset` for all six configurations.
Six targeted tests cover typed loading and grouped splits, public-only export,
fractional/null-field conventions, coherent problem fields, integrity checks, and
standalone numerical replay. One validation case per family, using nx=64, nt=96,
and grading=2, passes the project's independent accuracy and IC/BC checks.
Evidence is stored in `runs/huggingface-tests.log` and
`runs/huggingface-validation.log`; `runs/huggingface-preview.png` records the rendering.

Upload only the contents of the reviewed dataset package to the dataset repository
you choose later. The package README already declares the configurations and splits
needed by the Hub. A repository identity, visibility, and redistribution license
were not selected during local preparation. The user subsequently authorized
`JeremyLu1017/PDE-RSI`; upload preserved the existing private visibility and left
redistribution-license selection unchanged. For future uploads after local edits:

```bash
bash scripts/push_hf.sh --repo-id JeremyLu1017/PDE-RSI
```

The upload script requires a project-local Hugging Face login, revalidates the package,
uploads only its manifest-listed files, guards against concurrent remote commits,
preserves unrelated repository files, and verifies each committed file's hash.

The layout follows the official
[manual configuration guide](https://huggingface.co/docs/hub/datasets-manual-configuration)
and [dataset feature documentation](https://huggingface.co/docs/datasets/about_dataset_features).
