"""Build (and optionally push) every PDE benchmark of this project to the Hugging Face dataset repo.

Packages, each with training and validation cases only (sealed test splits stay evaluator-only):
    pdebench/        pde_benchmarks.tex Problems 1-11 (scripts/build_pdebench_hf.py; the tex now covers 1-16)
    pdebench_ext/    pde_benchmarks.tex Problems 12-16 (Kuramoto-Sivashinsky, Navier-Stokes, Cahn-Hilliard,
                     Maxwell TE_z, time-harmonic electromagnetic)
    inventory/       PDE Data Inventory, 12 families
    manufactured_2d/ the 2-D wave and group FPDE sets of the code protocol
The 48-case 1-D manufactured package at the repository root is already published and kept unchanged.
Usage (after sourcing scripts/env.sh):
    python scripts/build_all_benchmarks_hf.py            # build and validate datasets/huggingface/all-benchmarks
    python scripts/build_all_benchmarks_hf.py --push     # also commit it to JeremyLu1017/PDE-RSI
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np

from pde_rsi import inventory as inv
from pde_rsi import pdebench_ext as px
from pde_rsi.storage import PROJECT_ROOT

REPO = "JeremyLu1017/PDE-RSI"
PACKAGE = PROJECT_ROOT/"datasets/huggingface/all-benchmarks"
SPLITS = {"evolution": "train", "development": "validation"}
CONFIGS = {"pdebench_ext": "pdebench_ext/data", "inventory": "inventory/data"}


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT/f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def floats(values):
    return None if values is None else np.asarray(values, dtype=float).tolist()


def write_parquet(root, rows_by_split):
    import pyarrow as pa
    import pyarrow.parquet as pq
    for name, rows in rows_by_split.items():
        (root/f"records/{name}.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in rows))
        pq.write_table(pa.Table.from_pylist(rows), root/f"data/{name}.parquet")


def ext_row(case, name, folder="pdebench_ext"):
    task, data, view = case.truth, case.data(), case.public_view()
    return {
        "case_id": case.case_id, "split": name, "tex_case": task.spec["problem"], "title": px.CASES[task.spec["problem"]][3],
        "equation_class": task.family, "equation": px.LIBRARY[task.family]["equation"],
        "true_parameters": json.dumps(px.true_parameters(task.spec)), "hidden_spec": json.dumps(task.spec),
        "layout": task.layout, "components": list(task.components), "domain": json.dumps(view["domain"]),
        "boundary_description": view["boundary_description"], "noise_std": floats(data["noise_std"]),
        "data_x": floats(data["x"]), "data_y": floats(data["y"]), "data_t": floats(data["t"]),
        "data_u": floats(data["u"]), "evaluation_grid": json.dumps(view["evaluation_grid"]),
        "relative_l2_tolerance": task.relative_l2_tolerance, "boundary_tolerance": task.boundary_tolerance,
        "initial_tolerance": task.initial_tolerance, "max_seconds": task.max_seconds, "memory_mb": task.memory_mb,
        "reference_file": f"{folder}/reference/{case.case_id}.npz"}


def inventory_row(case, name, folder="inventory"):
    task, data, view = case.truth, case.data(), case.public_view()
    spec = {k: v for k, v in task.spec.items() if k != "fields"}
    fields = task.spec.get("fields") or {}
    return {
        "case_id": case.case_id, "split": name, "family_key": task.spec["family_key"],
        "title": inv.TITLES[task.spec["family_key"]], "equation_class": task.family,
        "equation": inv.library()[task.family]["equation"],
        "true_parameters": json.dumps(inv.true_parameters(task.spec)), "hidden_spec": json.dumps(spec, default=float),
        "layout": task.layout, "domain": json.dumps(view["domain"]), "boundary_description": view["boundary_description"],
        "noise_std": float(data["noise_std"]), "data_x": floats(data["x"]), "data_y": floats(data["y"]),
        "data_t": floats(data["t"]), "data_points": floats(data["points"]), "data_u": floats(data["u"]),
        "gridded_inputs": sorted(fields), "inputs_file": f"{folder}/inputs/{case.case_id}.npz" if fields else None,
        "evaluation_grid": json.dumps(view["evaluation_grid"]),
        "relative_l2_tolerance": task.relative_l2_tolerance, "max_seconds": task.max_seconds,
        "memory_mb": task.memory_mb, "reference_file": f"{folder}/reference/{case.case_id}.npz"}


def build_ext(root):
    for sub in ("data", "records", "reference"):
        (root/sub).mkdir(parents=True)
    rows = {}
    for split, name in SPLITS.items():
        rows[name] = []
        for case in sorted(px.make_cases(split), key=lambda c: c.case_id):
            axes = px.evaluation_axes(case.truth)
            np.savez_compressed(root/f"reference/{case.case_id}.npz", u=px.reference(case.truth),
                                **{k: v for k, v in axes.items()})
            rows[name].append(ext_row(case, name))
    write_parquet(root, rows)
    (root/"library.json").write_text(json.dumps(px.LIBRARY, indent=2)+"\n")
    shutil.copyfile(PROJECT_ROOT/"src/pde_rsi/equations_ext.py", root/"equations_ext.py")
    shutil.copyfile(PROJECT_ROOT/"pde_benchmarks.tex", root/"pde_benchmarks.tex")
    lines = ["# pde_benchmarks.tex Problems 12-16 (`pdebench_ext` configuration)", "",
             "Identify-then-solve cases for the fourth-order, incompressible-flow, phase-field and electromagnetic",
             "problems of `pde_benchmarks.tex`: Kuramoto-Sivashinsky (12a manufactured, 12b chaotic), 2-D incompressible",
             "Navier-Stokes (13a Taylor-Green, 13b Kovasznay), Cahn-Hilliard (14a manufactured, 14b spinodal decomposition),",
             "Maxwell TE_z (15a cavity mode, 15b current-driven) and time-harmonic electromagnetic (16a, 16b). Each case",
             "gives 256 noisy samples of every solution component (1% RMS noise per component), the domain, black-box",
             "forcing / initial / boundary data, and the library in `library.json`.", "",
             "This release has the training and validation cases (10 each). Sealed test cases are withheld.", "",
             "| Path | Contents |", "| --- | --- |",
             "| `data/{train,validation}.parquet`, `records/*.jsonl` | agent inputs, hidden truth (`hidden_spec`, `true_parameters`) and evaluation settings |",
             "| `reference/<case_id>.npz` | solution on the evaluation grid: `u` with components on the last axis (`[t, x, c]` or `[t, y, x, c]`) and the grid axes |",
             "| `equations_ext.py` | forcing, initial, boundary and closed-form solutions of a case's `hidden_spec` |",
             "| `pde_benchmarks.tex` | the specification (Problems 12-16) |", "",
             "Scoring: interpolate linearly onto the evaluation grid; space-time and final-time relative L2 over all",
             "components (pressure after removing its spatial mean) must both be at most 2e-3, initial and boundary",
             "conditions must hold to 1e-10, within `max_seconds` (90 s 1-D, 300 s 2-D). Cases 12b and 14b have no closed",
             "form; their references are Fourier-spectral ETDRK4 solutions checked against a half-resolution solve.", "",
             "| Case | Split | tex case | Title | Class | True coefficients |", "| --- | --- | --- | --- | --- | --- |"]
    for name, items in rows.items():
        for r in items:
            lines.append(f"| `{r['case_id']}` | {name} | {r['tex_case']} | {r['title']} | {r['equation_class']} | "
                         f"{', '.join(f'{k}={v:.4g}' for k, v in json.loads(r['true_parameters']).items()) or '—'} |")
    (root/"README.md").write_text("\n".join(lines)+"\n")
    return {name: len(items) for name, items in rows.items()}


def build_inventory(root):
    for sub in ("data", "records", "reference", "inputs"):
        (root/sub).mkdir(parents=True)
    rows = {}
    for split, name in SPLITS.items():
        rows[name] = []
        for case in sorted(inv.make_cases(split), key=lambda c: c.case_id):
            fields = case.truth.spec.get("fields") or {}
            if fields:
                np.savez_compressed(root/f"inputs/{case.case_id}.npz", **fields)
            shutil.copyfile(inv.reference_path(case.truth), root/f"reference/{case.case_id}.npz")
            rows[name].append(inventory_row(case, name))
    write_parquet(root, rows)
    (root/"library.json").write_text(json.dumps(inv.library(), indent=2)+"\n")
    shutil.copyfile(PROJECT_ROOT/"src/pde_rsi/equations_inventory.py", root/"equations_inventory.py")
    shutil.copyfile(PROJECT_ROOT/"docs/inventory.md", root/"SPECIFICATION.md")
    lines = ["# PDE Data Inventory suite (`inventory` configuration)", "",
             "Twelve families from *PDE Data Inventory* (X. Kong, 19 October 2026), with inputs from the DiffusionPDE",
             "test set (Huang et al. 2024) where the inventory leaves gaps: 1-D Burgers (two initial-data families),",
             "linear convection-diffusion, Darcy, Poisson, Helmholtz, 2-D Navier-Stokes, Kolmogorov flow, Allen-Cahn,",
             "2-D stochastic Burgers, advection-diffusion-reaction and a 4-D elliptic problem. Each case gives 256 noisy",
             "samples (1% RMS noise), black-box forcing / initial / boundary / coefficient data and gridded inputs.", "",
             "This release has the training and validation cases (12 each). Sealed test splits are withheld.", "",
             "| Path | Contents |", "| --- | --- |",
             "| `data/{train,validation}.parquet`, `records/*.jsonl` | agent inputs, hidden truth and evaluation settings |",
             "| `inputs/<case_id>.npz` | gridded inputs (DiffusionPDE fields, coefficient fields) |",
             "| `reference/<case_id>.npz` | high-resolution reference on the evaluation grid and the noise-free samples |",
             "| `equations_inventory.py`, `SPECIFICATION.md` | input definitions and the suite specification |", "",
             "Tolerance 2e-3 (4-D elliptic: 5e-2); 90 s for 1-D and 300 s for 2-D and 4-D cases."]
    (root/"README.md").write_text("\n".join(lines)+"\n")
    return {name: len(items) for name, items in rows.items()}


def build_manufactured(root):
    root.mkdir(parents=True)
    counts = {}
    for name, source in (("wave2d", "benchmarks/wave2d"), ("group_fpde", "benchmarks/group_fpde")):
        (root/name).mkdir()
        for split, label in SPLITS.items():
            text = (PROJECT_ROOT/source/f"{split}.jsonl").read_text()
            (root/name/f"{label}.jsonl").write_text(text)
            counts[f"{name}/{label}"] = sum(1 for line in text.splitlines() if line.strip())
        shutil.copyfile(PROJECT_ROOT/source/"manifest.json", root/name/"manifest.json")
    lines = ["# Manufactured 2-D wave and group FPDE sets (code protocol)", "",
             "Manufactured cases of the code protocol, where the equation is given and the solver is written for it.",
             "`wave2d/`: u_tt - D (u_xx + u_yy) = f on [0, L]^2 with zero Dirichlet data, 4 training and 4 validation cases.",
             "`group_fpde/`: fractional ODE, Caputo FPDE and time- and space-time-variable-order FPDEs on [0, pi] x [0, 1],",
             "8 training and 8 validation cases. Each JSON line is one task with its forcing expression, domain, initial",
             "and boundary data and limits. Reserved cases are withheld. Pass tolerance 2e-3 (relative L2, space-time and",
             "final time), 15 s and 1 GB per solve."]
    (root/"README.md").write_text("\n".join(lines)+"\n")
    return counts


def build():
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    PACKAGE.mkdir(parents=True)
    tex = load_script("build_pdebench_hf")
    tex.build()
    counts = {"pdebench": tex.validate(tex.make_cases("evolution")+tex.make_cases("development"))}
    shutil.copytree(tex.PACKAGE/"pdebench", PACKAGE/"pdebench")
    counts["pdebench_ext"] = build_ext(PACKAGE/"pdebench_ext")
    counts["inventory"] = build_inventory(PACKAGE/"inventory")
    counts["manufactured_2d"] = build_manufactured(PACKAGE/"manufactured_2d")
    sealed = [p for p in PACKAGE.rglob("*") if any(tag in p.name for tag in ("case-r", "case-er", "case-c", "case-d"))]
    if sealed:
        raise RuntimeError(f"Sealed test files present: {sealed[:5]}")
    files = {str(p.relative_to(PACKAGE)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(PACKAGE.rglob("*")) if p.is_file()}
    (PACKAGE/"release_manifest.json").write_text(json.dumps(
        {"counts": counts, "sealed_test_cases_included": False, "files": files,
         "created_utc": datetime.now(timezone.utc).isoformat()}, indent=2)+"\n")
    return counts


def validate():
    from datasets import load_dataset
    for config, path in CONFIGS.items():
        root = PACKAGE/path.split("/")[0]
        loaded = load_dataset("parquet", data_files={n: str(root/f"data/{n}.parquet") for n in SPLITS.values()},
                              cache_dir=str(PROJECT_ROOT/f".runtime/cache/huggingface/validation-{config}"))
        for name in SPLITS.values():
            mirror = [json.loads(line) for line in (root/f"records/{name}.jsonl").read_text().splitlines()]
            if list(loaded[name]) != mirror:
                raise ValueError(f"{config}: parquet rows differ from records/{name}.jsonl")


def verify_commit(api, oid, operations):
    """Check every committed file against its local source (LFS sha256 or git blob sha1); return the count."""
    remote = api.repo_info(REPO, repo_type="dataset", revision=oid, files_metadata=True)
    siblings = {item.rfilename: item for item in remote.siblings}
    for operation in operations:
        local = Path(operation.path_or_fileobj).read_bytes()
        sibling = siblings[operation.path_in_repo]
        lfs = getattr(sibling, "lfs", None)
        if lfs is not None:
            matched = (lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256) == hashlib.sha256(local).hexdigest()
        else:
            matched = hashlib.sha1(f"blob {len(local)}\0".encode()+local).hexdigest() == sibling.blob_id
        if not matched:
            raise RuntimeError(f"Committed file does not match the package: {operation.path_in_repo}")
    return len(operations)


def push():
    from huggingface_hub import CommitOperationAdd, HfApi, get_token, hf_hub_download
    token = get_token()
    if not token:
        raise SystemExit("No Hugging Face login. Source scripts/env.sh, run hf auth login, then retry.")
    api = HfApi(token=token)
    info = api.repo_info(REPO, repo_type="dataset")
    readme = Path(hf_hub_download(REPO, "README.md", repo_type="dataset", revision=info.sha, token=token,
                                  cache_dir=PROJECT_ROOT/".runtime/cache/huggingface/upload-check")).read_text()
    _, header, body = readme.split("---", 2)
    for config, path in CONFIGS.items():
        if f"config_name: {config}" not in header:
            header = header.rstrip("\n")+(f"\n- config_name: {config}\n  data_files:\n  - split: train\n    path: {path}/train.parquet"
                                          f"\n  - split: validation\n    path: {path}/validation.parquet\n")
    sections = {
        "## pde_benchmarks.tex Problems 12-16": "\n## pde_benchmarks.tex Problems 12-16 (`pdebench_ext`)\n\n20 identify-then-solve cases (10 train, 10 validation): Kuramoto-Sivashinsky, 2-D incompressible Navier-Stokes, Cahn-Hilliard, Maxwell TE_z and time-harmonic electromagnetic problems, with vector-valued solutions. See [`pdebench_ext/README.md`](pdebench_ext/README.md).\n",
        "## PDE Data Inventory": "\n## PDE Data Inventory (`inventory`)\n\n24 identify-then-solve cases (12 train, 12 validation) over 12 families with DiffusionPDE-derived inputs. See [`inventory/README.md`](inventory/README.md).\n",
        "## Manufactured 2-D wave and group FPDE": "\n## Manufactured 2-D wave and group FPDE sets\n\nCode-protocol JSONL task files. See [`manufactured_2d/README.md`](manufactured_2d/README.md).\n",
    }
    for marker, text in sections.items():
        if marker not in body:
            body = body.rstrip("\n")+"\n"+text
    staged = PACKAGE/".root-README.md"
    staged.write_text("---"+header.rstrip("\n")+"\n---"+body)
    sources = [p for p in sorted(PACKAGE.rglob("*")) if p.is_file() and p != staged and p.name != "release_manifest.json"]
    operations = [CommitOperationAdd(path_in_repo=str(p.relative_to(PACKAGE)), path_or_fileobj=p) for p in sources]
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=staged))
    print(f"Uploading {len(operations)} files to {REPO} (private={info.private})", flush=True)
    commit = api.create_commit(
        repo_id=REPO, repo_type="dataset", operations=operations, parent_commit=info.sha, num_threads=4,
        commit_message="Add pde_benchmarks.tex Problems 12-16, the PDE Data Inventory suite and the 2-D manufactured sets",
        commit_description="Training and validation cases only; sealed test splits are withheld.")
    verified = verify_commit(api, commit.oid, operations)
    staged.unlink()
    report = {"repository": REPO, "url": f"https://huggingface.co/datasets/{REPO}", "commit": commit.oid,
              "commit_url": commit.commit_url, "private": info.private, "files_uploaded": len(operations),
              "verified_files": verified, "sealed_test_cases_uploaded": False,
              "time_utc": datetime.now(timezone.utc).isoformat()}
    (PROJECT_ROOT/"runs/huggingface-upload-all-benchmarks.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    counts = build()
    validate()
    size = sum(p.stat().st_size for p in PACKAGE.rglob("*") if p.is_file())
    print(json.dumps({"package": str(PACKAGE), "counts": counts, "megabytes": round(size/1e6, 1)}, indent=2), flush=True)
    if args.push:
        push()


if __name__ == "__main__":
    main()
