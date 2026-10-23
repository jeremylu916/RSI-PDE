"""Build (and optionally push) the public pde_benchmarks.tex package for the Hugging Face dataset repo.

Only training and validation cases are included; the 17 sealed-test cases stay evaluator-only.
Usage (after sourcing scripts/env.sh):
    python scripts/build_pdebench_hf.py                      # build and validate datasets/huggingface/pdebench
    python scripts/build_pdebench_hf.py --push               # also commit it to JeremyLu1017/PDE-RSI

Repository layout added under pdebench/ (existing files are kept; the root card gains a "pdebench" config):
    pdebench/data/{train,validation}.parquet   one row per case: agent inputs, hidden truth, evaluation settings
    pdebench/records/{train,validation}.jsonl  reviewable mirror of the parquet rows
    pdebench/reference/<case_id>.npz           exact solution on the evaluation grid (x, t, u; y for 2-D)
    pdebench/problems/<case_id>.tex            the true equation and exact solution with the case's numbers
    pdebench/library.json, pde_benchmarks.tex, equations_bench.py, README.md, manifest.json
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np

from pde_rsi.pdebench import (EVALUATOR_VERSION, GRID, LIBRARY, NOISE_FRACTION, PROTOCOL_VERSION, TITLES,
                              evaluation_nodes, make_cases, reference, true_parameters)
from pde_rsi.storage import PROJECT_ROOT, project_path

REPO = "JeremyLu1017/PDE-RSI"
PACKAGE = PROJECT_ROOT/"datasets/huggingface/pdebench"
SPLITS = {"evolution": "train", "development": "validation"}
ORDER = ["1a", "1b", "1c", "2a", "2b", "2c", "3a", "3b", "3c", "3d", "4a", "4b", "4c", "5", "6", "7", "8", "9", "10", "11"]
CONFIG = """- config_name: pdebench
  data_files:
  - split: train
    path: pdebench/data/train.parquet
  - split: validation
    path: pdebench/data/validation.parquet
"""


def exporter():
    spec = importlib.util.spec_from_file_location("export_pdebench_cases", PROJECT_ROOT/"scripts/export_pdebench_cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(case, tex):
    task, s = case.truth, case.truth.spec
    data, view = case.data(), case.public_view()
    lists = {f"data_{k}": (None if data[k] is None else [float(v) for v in data[k]]) for k in ("x", "y", "t")}
    return {
        "case_id": case.case_id, "split": SPLITS[case.split], "tex_case": task.problem, "title": TITLES[task.problem],
        "equation_class": task.family, "equation": LIBRARY[task.family]["equation"],
        "true_parameters": json.dumps(true_parameters(task)), "hidden_spec": json.dumps(s),
        "coordinates": view["coordinates"], "domain": json.dumps(view["domain"]),
        "spatial_dimension": task.dimension, "time_dependent": not task.steady,
        "boundary_description": view["boundary_description"], "noise_std": float(data["noise_std"]),
        **lists, "data_u": [float(v) for v in data["u"]],
        "evaluation_space_points": view["evaluation_space_points"],
        "evaluation_time_points": view["evaluation_time_points"],
        "relative_l2_tolerance": task.relative_l2_tolerance, "boundary_tolerance": task.boundary_tolerance,
        "initial_tolerance": task.initial_tolerance, "max_seconds": task.max_seconds, "memory_mb": task.memory_mb,
        "problem_tex": tex.problem_tex(case), "reference_file": f"pdebench/reference/{case.case_id}.npz",
    }


def card(cases):
    lines = ["# pde_benchmarks.tex suite (`pdebench` configuration)", "",
             "Student-style inverse-then-forward tasks built from `pde_benchmarks.tex`. Each case is one tex case with",
             "its own numbers. The solver receives 256 noisy samples of the exact solution (1% RMS Gaussian noise),",
             "the domain, black-box forcing / initial / boundary data, and the 10-class library in `library.json`.",
             "It must identify the equation class and coefficients and return the solution over the whole domain.",
             "", "This release contains the training and validation cases only. The sealed test cases (the tex",
             "reserved profile and harder literature variants) are withheld so they remain a valid held-out test.", "",
             "## Files", "",
             "| Path | Contents |", "| --- | --- |",
             "| `data/{train,validation}.parquet` | one row per case: agent inputs (`data_*`, `domain`, `noise_std`, ...), hidden truth (`equation_class`, `true_parameters`, `hidden_spec`, `problem_tex`), and evaluation settings |",
             "| `records/*.jsonl` | the same rows as JSON lines |",
             "| `reference/<case_id>.npz` | exact solution on the evaluation grid: `x`, `t`, `u` (`u[t, x]`; `u[t, y, x]` with `y` for the 2-D case) |",
             "| `problems/<case_id>.tex` | the true equation, domain, coefficients, and exact solution |",
             "| `library.json` | candidate equation classes shown to solvers |",
             "| `equations_bench.py` | NumPy/SciPy forcing, initial, boundary, and exact-solution functions of a case's `hidden_spec` |",
             "| `pde_benchmarks.tex` | the benchmark specification |", "",
             "Fields `data_x`, `data_y`, `data_t` are null where a case has no such coordinate (steady cases have no",
             "`t`; the fractional ODE has no `x`). Forcing and boundary data are not tabulated: evaluate them with",
             "`equations_bench.forcing(spec, x, t, y)`, `initial(spec, x, y)`, and `boundary(spec, t)` using",
             "`spec = json.loads(row['hidden_spec'])` (solvers under test should only see these as black boxes).", "",
             "## Scoring", "",
             f"Interpolate the solution linearly onto the evaluation grid ({GRID['space']} x-nodes x {GRID['time']} t-nodes in 1-D,",
             f"{GRID['wave_space']}x{GRID['wave_space']}x{GRID['wave_time']} for the 2-D wave, {GRID['space']} nodes for steady cases, "
             f"{GRID['time']} for the ODE) and compute",
             "the trapezoidal space-time and final-time relative L2 errors against `reference/<case_id>.npz`. A case",
             "passes when both are at most `relative_l2_tolerance` (2e-3), initial and Dirichlet values hold within",
             "1e-10, and the solve fits `max_seconds` (90 s) and `memory_mb`.", "",
             "## Cases", "", "| Case | Split | tex case | Description | Class | True coefficients |",
             "| --- | --- | --- | --- | --- | --- |"]
    for case in sorted(cases, key=lambda c: (c.split != "evolution", ORDER.index(c.truth.problem))):
        params = ", ".join(f"{k}={v:.4g}" for k, v in true_parameters(case.truth).items())
        lines.append(f"| `{case.case_id}` | {SPLITS[case.split]} | {case.truth.problem} | {TITLES[case.truth.problem]} | "
                     f"{case.family} | {params} |")
    lines += ["", f"Protocol `{PROTOCOL_VERSION}`, evaluator `{EVALUATOR_VERSION}`, noise fraction {NOISE_FRACTION}."]
    return "\n".join(lines)+"\n"


def build():
    import pyarrow as pa
    import pyarrow.parquet as pq
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    root = PACKAGE/"pdebench"
    for sub in ("data", "records", "reference", "problems"):
        (root/sub).mkdir(parents=True)
    tex = exporter()
    cases = [case for split in SPLITS for case in make_cases(split)]
    if any(case.split == "test" or case.case_id.startswith("case-r") for case in cases):
        raise RuntimeError("Sealed test cases must not be packaged")
    for split, name in SPLITS.items():
        rows = [row(c, tex) for c in sorted((c for c in cases if c.split == split), key=lambda c: c.case_id)]
        (root/f"records/{name}.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in rows))
        pq.write_table(pa.Table.from_pylist(rows), root/f"data/{name}.parquet")
    for case in cases:
        x, t = evaluation_nodes(case.truth)
        extra = {"y": x} if case.truth.dimension == 2 else {}
        np.savez_compressed(root/f"reference/{case.case_id}.npz", x=x, t=t, u=reference(case.truth), **extra)
        (root/f"problems/{case.case_id}.tex").write_text(tex.problem_tex(case))
    (root/"library.json").write_text(json.dumps(LIBRARY, indent=2)+"\n")
    shutil.copyfile(PROJECT_ROOT/"pde_benchmarks.tex", root/"pde_benchmarks.tex")
    shutil.copyfile(PROJECT_ROOT/"src/pde_rsi/equations_bench.py", root/"equations_bench.py")
    (root/"README.md").write_text(card(cases))
    files = {str(p.relative_to(PACKAGE)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(PACKAGE.rglob("*")) if p.is_file()}
    manifest = {"protocol_version": PROTOCOL_VERSION, "evaluator_version": EVALUATOR_VERSION,
                "splits": {name: sum(c.split == split for c in cases) for split, name in SPLITS.items()},
                "sealed_test_cases_included": False, "files": files,
                "created_utc": datetime.now(timezone.utc).isoformat()}
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return cases


def validate(cases):
    """Load the parquet through datasets, compare with the JSONL mirror, and re-check the exact data."""
    from datasets import load_dataset
    root = PACKAGE/"pdebench"
    loaded = load_dataset("parquet", data_files={n: str(root/f"data/{n}.parquet") for n in SPLITS.values()},
                          cache_dir=str(PROJECT_ROOT/".runtime/cache/huggingface/validation-pdebench"))
    by_id = {c.case_id: c for c in cases}
    for name in SPLITS.values():
        mirror = [json.loads(line) for line in (root/f"records/{name}.jsonl").read_text().splitlines()]
        if list(loaded[name]) != mirror:
            raise ValueError(f"Parquet rows differ from records/{name}.jsonl")
        for item in mirror:
            case = by_id[item["case_id"]]
            if item["split"] != name or not np.array_equal(np.asarray(item["data_u"]), case.data()["u"]):
                raise ValueError(f"Data mismatch for {item['case_id']}")
            with np.load(PACKAGE/item["reference_file"]) as archive:
                if not np.array_equal(archive["u"], reference(case.truth)):
                    raise ValueError(f"Reference mismatch for {item['case_id']}")
    leaked = [p for p in PACKAGE.rglob("*") if "case-r" in p.name]
    if leaked:
        raise ValueError(f"Sealed case files present: {leaked}")
    return {name: len(loaded[name]) for name in SPLITS.values()}


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
    if "config_name: pdebench" not in header:
        header = header.rstrip("\n")+"\n"+CONFIG
    section = ("\n## pde_benchmarks.tex suite (`pdebench`)\n\n32 student-style identify-then-solve cases (16 train, 16"
               " validation) from `pde_benchmarks.tex`: Poisson, heat, convection-diffusion, Burgers, 2-D wave,"
               " fractional ODE, Caputo, multi-term, and variable-order problems with noisy data and exact solutions."
               " Sealed test cases are withheld. See [`pdebench/README.md`](pdebench/README.md).\n")
    if "## pde_benchmarks.tex suite" not in body:
        body = body.rstrip("\n")+"\n"+section
    staged = PACKAGE/".root-README.md"
    staged.write_text("---"+header.rstrip("\n")+"\n---"+body)
    operations = [CommitOperationAdd(path_in_repo=str(p.relative_to(PACKAGE)), path_or_fileobj=p)
                  for p in sorted(PACKAGE.rglob("*")) if p.is_file() and p != staged]
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=staged))
    print(f"Uploading {len(operations)} files to {REPO} (private={info.private})", flush=True)
    commit = api.create_commit(
        repo_id=REPO, repo_type="dataset", operations=operations, parent_commit=info.sha, num_threads=4,
        commit_message="Add pde_benchmarks.tex suite (pdebench config: train and validation cases)",
        commit_description=("32 identify-then-solve cases from pde_benchmarks.tex with noisy data, exact solutions on "
                            "the evaluation grid, and per-case LaTeX. Sealed test cases are withheld."))
    remote = api.repo_info(REPO, repo_type="dataset", revision=commit.oid, files_metadata=True)
    siblings = {item.rfilename: item for item in remote.siblings}
    verified = 0
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
        verified += 1
    staged.unlink()
    report = {"repository": REPO, "url": f"https://huggingface.co/datasets/{REPO}/tree/main/pdebench",
              "commit": commit.oid, "commit_url": commit.commit_url, "private": info.private,
              "files_uploaded": len(operations), "verified_files": verified, "sealed_test_cases_uploaded": False,
              "time_utc": datetime.now(timezone.utc).isoformat()}
    (PROJECT_ROOT/"runs/huggingface-upload-pdebench.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    project_path(PACKAGE)
    cases = build()
    counts = validate(cases)
    size = sum(p.stat().st_size for p in PACKAGE.rglob("*") if p.is_file())
    print(json.dumps({"package": str(PACKAGE), "rows": counts, "files": sum(1 for p in PACKAGE.rglob('*') if p.is_file()),
                      "megabytes": round(size/1e6, 1)}, indent=2), flush=True)
    if args.push:
        push()


if __name__ == "__main__":
    main()
