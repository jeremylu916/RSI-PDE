"""Build (and optionally push) the 1001-case expanded release (src/pde_rsi/expanded.py) to Hugging Face.

Every benchmark problem of the project with new coefficients, initial data and boundary data, under expanded/ in
JeremyLu1017/PDE-RSI. Each suite has train / validation / test splits. The release's test split is public; it is
not one of the project's sealed test splits, which stay private (expanded.sealed_overlap() checks no case repeats one).
    expanded/index.parquet            one row per case: suite, template, variant, split, class, true parameters
    expanded/pdebench/                pde_benchmarks.tex Problems 1-11, 20 templates x 20 variants
    expanded/pdebench_ext/            pde_benchmarks.tex Problems 12-16, 10 templates x 20 variants
    expanded/inventory/               PDE Data Inventory, 12 families x 20 variants
    expanded/manufactured/            code-protocol 1-D, 2-D wave and group FPDE families (6 x 15, 15, 4 x 14)
References are float64, as in the earlier releases: they satisfy the initial and boundary data to 1e-10.
Build the references first (scripts/build_expanded_references.py), then, after sourcing scripts/env.sh:
    python scripts/build_expanded_hf.py            # build and validate datasets/huggingface/expanded-1001
    python scripts/build_expanded_hf.py --push     # also commit it to JeremyLu1017/PDE-RSI
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np

from pde_rsi import expanded as X
from pde_rsi import inventory as inv
from pde_rsi import pdebench as pb
from pde_rsi import pdebench_ext as px
from pde_rsi.reference import output_nodes, output_nodes2d, output_nodes_group, reference, reference2d, reference_group
from pde_rsi.storage import PROJECT_ROOT
from pde_rsi.tasks import GROUP_FAMILIES

REPO = "JeremyLu1017/PDE-RSI"
PACKAGE = PROJECT_ROOT/"datasets/huggingface/expanded-1001"
ROOT = PACKAGE/"expanded"
NAMES = ("train", "validation", "test")
SUITES = ("pdebench", "pdebench_ext", "inventory", "manufactured")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT/f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ALL = load_script("build_all_benchmarks_hf")
TEX_HF = load_script("build_pdebench_hf")
TEX = TEX_HF.exporter()
TEX_HF.SPLITS = {**TEX_HF.SPLITS, **{n: n for n in NAMES}}  # expanded cases carry release split names
TEX.SPLIT_NAMES.update(train="training", validation="validation", test="test")
_EXACT_TEX = TEX.exact_tex


def exact_tex(s):
    """pdebench exact solution with the expanded release's amplitude U (3b) and linear Dirichlet lift."""
    text = _EXACT_TEX(s)
    if s["problem"] == "3b" and s.get("U", 1.0) != 1.0:
        text = text.replace("u(x) = ", rf"u(x) = {TEX.g(s['U'])}\,", 1)
    if s.get("bc_left") or s.get("bc_right"):
        slope = (s["bc_right"]-s["bc_left"])/(s["b"]-s["a"])
        text += rf" + {TEX.g(s['bc_left'])} + ({TEX.g(slope)})\,(x-{TEX.g(s['a'])})"
    return text


TEX.exact_tex = exact_tex


def variant(case_id):
    return int(case_id.rsplit("-", 1)[1])


def write_split_tables(root, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    (root/"data").mkdir(parents=True, exist_ok=True)
    (root/"records").mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        items = [r for r in rows if r["split"] == name]
        (root/f"records/{name}.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in items))
        pq.write_table(pa.Table.from_pylist(items), root/f"data/{name}.parquet")


def build_pdebench(cases):
    root = ROOT/"pdebench"
    for sub in ("reference", "problems"):
        (root/sub).mkdir(parents=True)
    rows = []
    for case in cases:
        row = TEX_HF.row(case, TEX)
        row["reference_file"] = f"expanded/pdebench/reference/{case.case_id}.npz"
        rows.append({**row, "template": case.truth.problem, "variant": variant(case.case_id)})
        x, t = pb.evaluation_nodes(case.truth)
        extra = {"y": x} if case.truth.dimension == 2 else {}
        np.savez_compressed(root/f"reference/{case.case_id}.npz", x=x, t=t, u=pb.reference(case.truth), **extra)
        (root/f"problems/{case.case_id}.tex").write_text(TEX.problem_tex(case))
    write_split_tables(root, rows)
    (root/"library.json").write_text(json.dumps(pb.LIBRARY, indent=2)+"\n")
    shutil.copyfile(PROJECT_ROOT/"src/pde_rsi/equations_bench.py", root/"equations_bench.py")
    return rows


def build_ext(cases):
    root = ROOT/"pdebench_ext"
    (root/"reference").mkdir(parents=True)
    rows = []
    for case in cases:
        row = ALL.ext_row(case, case.split, folder="expanded/pdebench_ext")
        rows.append({**row, "template": case.truth.spec["problem"], "variant": variant(case.case_id)})
        np.savez_compressed(root/f"reference/{case.case_id}.npz", u=px.reference(case.truth),
                            **px.evaluation_axes(case.truth))
    write_split_tables(root, rows)
    (root/"library.json").write_text(json.dumps(px.LIBRARY, indent=2)+"\n")
    shutil.copyfile(PROJECT_ROOT/"src/pde_rsi/equations_ext.py", root/"equations_ext.py")
    return rows


def build_inventory(cases):
    root = ROOT/"inventory"
    for sub in ("reference", "inputs"):
        (root/sub).mkdir(parents=True)
    rows = []
    for case in cases:
        row = ALL.inventory_row(case, case.split, folder="expanded/inventory")
        rows.append({**row, "template": case.truth.spec["family_key"], "variant": variant(case.case_id)})
        fields = case.truth.spec.get("fields") or {}
        if fields:
            np.savez_compressed(root/f"inputs/{case.case_id}.npz", **fields)
        shutil.copyfile(inv.reference_path(case.truth), root/f"reference/{case.case_id}.npz")
    write_split_tables(root, rows)
    (root/"library.json").write_text(json.dumps(inv.library(), indent=2)+"\n")
    shutil.copyfile(PROJECT_ROOT/"src/pde_rsi/equations_inventory.py", root/"equations_inventory.py")
    return rows


def manufactured_reference(task):
    if task.family == "wave2d":
        x, y, t = output_nodes2d(task)
        return {"x": x, "y": y, "t": t, "u": reference2d(task, x, y, t)}
    if task.family in GROUP_FAMILIES:
        x, t = output_nodes_group(task)
        return {"x": x, "t": t, "u": reference_group(task, x, t)}
    x, t = output_nodes(task)
    return {"x": x, "t": t, "u": reference(task, x, t)}


def build_manufactured(tasks):
    root = ROOT/"manufactured"
    (root/"reference").mkdir(parents=True)
    rows = []
    for task in tasks:
        record = (task.wave_public_record() if task.family == "wave2d" else
                  task.group_public_record() if task.family in GROUP_FAMILIES else task.public_record())
        rows.append({"case_id": task.task_id, "split": X.release_split(task), "template": task.family,
                     "variant": variant(task.task_id), "equation_class": task.family, "equation": record["equation"],
                     "true_parameters": json.dumps({k: record[k] for k in ("diffusivity", "velocity", "orders", "weights")}),
                     "task": json.dumps(record), "reference_file": f"expanded/manufactured/reference/{task.task_id}.npz"})
        np.savez_compressed(root/f"reference/{task.task_id}.npz", **manufactured_reference(task))
    write_split_tables(root, rows)
    return rows


def card(rows):
    counts = Counter((r["suite"], r["split"]) for r in rows)
    templates = Counter(r["suite"] for r in rows if r["variant"] == 0)
    lines = ["# Expanded release: 1001 PDE cases", "",
             "Every benchmark problem of PDE-RSI with new coefficients, initial data and boundary data. Each of the 53",
             "problem templates keeps its equation class and geometry type and gets 14 to 20 variants drawn from a seeded",
             f"generator (`{X.VERSION}`). Per template, the variants are split into train, validation and test.", "",
             "| Suite | Templates | Variants per template | Train | Validation | Test | Total |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    per = {"pdebench": "20", "pdebench_ext": "20", "inventory": "20", "manufactured": "15 (1-D, 2-D wave), 14 (group FPDE)"}
    for suite in SUITES:
        total = sum(counts[(suite, n)] for n in NAMES)
        lines.append(f"| `{suite}` | {templates[suite]} | {per[suite]} | "+" | ".join(str(counts[(suite, n)]) for n in NAMES)
                     + f" | {total} |")
    lines.append("| **all** | "+f"{sum(templates.values())} | | "+" | ".join(str(sum(counts[(s, n)] for s in SUITES)) for n in NAMES)
                 + f" | **{len(rows)}** |")
    lines += ["", "## What varies", "",
              "- `pdebench` (Problems 1-11):",
              "  - Manufactured cases redraw the domain length, amplitude, D, v, beta, the fractional orders and the",
              "    spatial profile (sine, polynomial or mixed sine).",
              "  - In 3 of 4 variants of 1a, 2a, 3a, 4a, 7 and 8, the solution gets a linear lift, so the Dirichlet data",
              "    u(a) and u(b) are nonzero; the forcing is adjusted to match.",
              "  - Literature cases redraw their physical numbers: the Peclet number, the boundary-layer amplitude U (3b),",
              "    the front sharpness and location, the Gaussian pulse, the Burgers viscosity and shock states, and the",
              "    time-power p and order alpha of the fractional cases.",
              "- `pdebench_ext` (Problems 12-16):",
              "  - KS: the domain, the amplitude and the horizon. For 12b, also the amplitude and shift of the",
              "    Kassam-Trefethen initial data.",
              "  - Taylor-Green: the wavenumber and nu. Kovasznay: the Reynolds number.",
              "  - Cahn-Hilliard: M, epsilon and the domain. For 14b, five new random initial Fourier modes.",
              "  - Maxwell: the cavity size, the mode (m, n) and the wave speed.",
              "  - Curl-curl: the mode numbers and k. For 16b, k lies between two Maxwell eigenvalues.",
              "- `inventory`:",
              "  - DiffusionPDE-based families (Burgers random, Darcy, Poisson, Helmholtz, Navier-Stokes) use DiffusionPDE",
              "    test instances 4-23, which no other split of the project uses.",
              "  - Other families redraw their coefficients, Gaussian initial data, forcing or random coefficient fields.",
              "- `manufactured`: new length, horizon, coefficients, amplitude, mode, profile, time exponent and fractional",
              "  orders.", "",
              "No case repeats a case of the project, public or sealed. The release's `test` split is public. The",
              "project's own sealed test splits are not in this release.", "",
              "## Files", "", "| Path | Contents |", "| --- | --- |",
              "| `index.parquet` | one row per case: `suite`, `template`, `variant`, `split`, class, true parameters |",
              "| `<suite>/data/{train,validation,test}.parquet`, `<suite>/records/*.jsonl` | agent inputs (noisy samples, domain, conditions), hidden truth (`hidden_spec`, `true_parameters`) and evaluation settings; the columns match the unexpanded `pdebench`, `pdebench_ext` and `inventory` configurations |",
              "| `<suite>/reference/<case_id>.npz` | reference solution on the evaluation grid (float64) |",
              "| `pdebench/problems/<case_id>.tex` | the true equation, coefficients and exact solution |",
              "| `inventory/inputs/<case_id>.npz` | gridded inputs (DiffusionPDE fields, coefficient fields) |",
              "| `manufactured/data/*.parquet` | one task per row; `task` is the full code-protocol task JSON (forcing, initial and boundary expressions, limits) |",
              "| `<suite>/library.json`, `equations_*.py` | candidate classes and the forcing / initial / boundary / exact functions of `hidden_spec` |",
              "",
              "## References and scoring", "",
              "- **Closed form:** cases that have one use it.",
              "- **Computed references:**",
              "  - 12b and 14b: Fourier-spectral ETDRK4.",
              "  - Inventory families: the suite's reference solvers.",
              "  - Each is checked against a half-resolution solve; the measured differences are in",
              "    `reference_report.json`.",
              "- **Scoring** is the same as for the unexpanded suites:",
              "  - relative L2 over space-time and at the final time, at most 2e-3 (inventory 4-D elliptic: 5e-2);",
              "  - initial and boundary conditions to 1e-10;",
              "  - within `max_seconds`."]
    return "\n".join(lines)+"\n"


def build():
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    ROOT.mkdir(parents=True)
    clashes = X.sealed_overlap()
    if clashes:
        raise RuntimeError(f"Expanded cases repeat existing cases: {clashes[:5]}")
    cases = X.all_cases()
    missing = [c.case_id for c in cases["inventory"] if not inv.reference_path(c.truth).exists()]
    missing += [c.case_id for c in cases["pdebench_ext"] if not px.reference_path(c.truth).exists()
                and c.truth.spec["problem"] in ("12b", "14b")]
    if missing:
        raise SystemExit(f"{len(missing)} references missing (run scripts/build_expanded_references.py): {missing[:5]}")
    rows = {"pdebench": build_pdebench(cases["pdebench"]), "pdebench_ext": build_ext(cases["pdebench_ext"]),
            "inventory": build_inventory(cases["inventory"]), "manufactured": build_manufactured(cases["manufactured"])}
    index = [{"case_id": r["case_id"], "suite": suite, "template": r["template"], "variant": r["variant"],
              "split": r["split"], "equation_class": r["equation_class"], "true_parameters": r["true_parameters"],
              "reference_file": r["reference_file"], "records_file": f"expanded/{suite}/records/{r['split']}.jsonl"}
             for suite in SUITES for r in rows[suite]]
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(index), ROOT/"index.parquet")
    report = {}
    for path in sorted(p for p in (PROJECT_ROOT/"data/expanded").glob("reference_report-*.json")
                       if not p.name.startswith("reference_report-bench-")):
        # rows of cases another node built carry no error ("skipped"; older builders dropped the flag)
        report.update({k: v for k, v in json.loads(path.read_text()).items()
                       if not v.get("skipped") and v.get("coarse_vs_fine_relative_l2") is not None})
    (ROOT/"reference_report.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    shutil.copyfile(PROJECT_ROOT/"pde_benchmarks.tex", ROOT/"pde_benchmarks.tex")
    shutil.copyfile(PROJECT_ROOT/"src/pde_rsi/expanded.py", ROOT/"expanded.py")
    (ROOT/"README.md").write_text(card(index))
    files = {str(p.relative_to(PACKAGE)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(PACKAGE.rglob("*")) if p.is_file()}
    counts = {suite: dict(Counter(r["split"] for r in rows[suite])) for suite in SUITES}
    (PACKAGE/"release_manifest.json").write_text(json.dumps(
        {"version": X.VERSION, "cases": len(index), "counts": counts, "project_sealed_cases_included": False,
         "files": files, "created_utc": datetime.now(timezone.utc).isoformat()}, indent=2)+"\n")
    return index


def validate(index):
    """Counts, splits, unique ids, parquet == JSONL mirror, every file present, and the stored references pass."""
    from datasets import load_dataset
    if len(index) != 1001 or len({r["case_id"] for r in index}) != 1001:
        raise ValueError("Expected 1001 distinct cases")
    if Counter(r["split"] for r in index) != Counter(train=705, validation=148, test=148):
        raise ValueError(f"Unexpected splits {Counter(r['split'] for r in index)}")
    if any(not r["case_id"].startswith("x-") for r in index):
        raise ValueError("Every expanded case id starts with x-")
    for suite in SUITES:
        root = ROOT/suite
        loaded = load_dataset("parquet", data_files={n: str(root/f"data/{n}.parquet") for n in NAMES},
                              cache_dir=str(PROJECT_ROOT/f".runtime/cache/huggingface/validation-expanded-{suite}"))
        for name in NAMES:
            mirror = [json.loads(line) for line in (root/f"records/{name}.jsonl").read_text().splitlines()]
            if list(loaded[name]) != mirror:
                raise ValueError(f"{suite}: parquet rows differ from records/{name}.jsonl")
            for item in mirror:
                for key in ("reference_file", "inputs_file"):
                    if item.get(key) and not (PACKAGE/item[key]).exists():
                        raise ValueError(f"Missing {item[key]}")
    # the stored references pass the evaluators as exact submissions (first variant of every template)
    cases = {c.case_id: c for c in X.tex_cases()+X.ext_cases()}
    failures = []
    for r in index:
        if r["variant"] or r["suite"] not in ("pdebench", "pdebench_ext"):
            continue
        case = cases[r["case_id"]]
        with np.load(PACKAGE/r["reference_file"]) as archive:
            u = archive["u"].astype(float)
            if r["suite"] == "pdebench":
                x, t = archive["x"], archive["t"]
                y = archive["y"] if "y" in archive.files else None
                result = pb.evaluate(case.truth, _submission(x=x, t=t, y=y, u=u))
            else:
                axes = {k: archive[k] for k in archive.files if k != "u"}
                steady = case.truth.layout == "2d_steady"
                result = px.evaluate(case.truth, _submission(x=axes["x"], y=axes.get("y"), t=axes.get("t", np.zeros(1)),
                                                             u=u[0] if steady else u))
        if not result["passed"]:
            failures.append((r["case_id"], result.get("relative_l2"), result.get("boundary_error")))
    if failures:
        raise ValueError(f"Stored references fail their evaluator: {failures}")


def _submission(**fields):
    import types
    return types.SimpleNamespace(method="reference", **fields)


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
    header = header.rstrip("\n")+"\n"
    for suite in SUITES:
        if f"config_name: expanded_{suite}\n" not in header:
            header += f"- config_name: expanded_{suite}\n  data_files:\n"+"".join(
                f"  - split: {n}\n    path: expanded/{suite}/data/{n}.parquet\n" for n in NAMES)
    if "config_name: expanded_index\n" not in header:
        header += "- config_name: expanded_index\n  data_files:\n  - split: all\n    path: expanded/index.parquet\n"
    marker = "## Expanded release: 1001 cases"
    if marker not in body:
        body = body.rstrip("\n")+(f"\n\n{marker} (`expanded_*`)\n\nAll benchmark problems with new coefficients, initial data and"
                                  " boundary data: 1001 cases over 53 templates (705 train, 148 validation, 148 test), in"
                                  " the configurations `expanded_pdebench`, `expanded_pdebench_ext`, `expanded_inventory`,"
                                  " `expanded_manufactured` and the index `expanded_index`. The test split of this release"
                                  " is public; the project's sealed test splits are not included. See"
                                  " [`expanded/README.md`](expanded/README.md).\n")
    staged = PACKAGE/".root-README.md"
    staged.write_text("---"+header+"---"+body)
    sources = [p for p in sorted(PACKAGE.rglob("*")) if p.is_file() and p != staged and p.name != "release_manifest.json"]
    operations = [CommitOperationAdd(path_in_repo=str(p.relative_to(PACKAGE)), path_or_fileobj=p) for p in sources]
    operations.append(CommitOperationAdd(path_in_repo="expanded/release_manifest.json",
                                         path_or_fileobj=PACKAGE/"release_manifest.json"))
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=staged))
    print(f"Uploading {len(operations)} files to {REPO} (private={info.private})", flush=True)
    commit = api.create_commit(
        repo_id=REPO, repo_type="dataset", operations=operations, parent_commit=info.sha, num_threads=8,
        commit_message="Add the 1001-case expanded release (new coefficients, initial and boundary data)",
        commit_description="53 templates; train / validation / test = 705 / 148 / 148. The project's sealed test splits "
                           "are not included.")
    verified = ALL.verify_commit(api, commit.oid, operations)
    staged.unlink()
    report = {"repository": REPO, "url": f"https://huggingface.co/datasets/{REPO}/tree/main/expanded",
              "commit": commit.oid, "commit_url": commit.commit_url, "private": info.private,
              "files_uploaded": len(operations), "verified_files": verified, "cases": 1001,
              "project_sealed_cases_uploaded": False, "time_utc": datetime.now(timezone.utc).isoformat()}
    (PROJECT_ROOT/"runs/huggingface-upload-expanded-1001.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    index = build()
    validate(index)
    size = sum(p.stat().st_size for p in PACKAGE.rglob("*") if p.is_file())
    print(json.dumps({"package": str(PACKAGE), "cases": len(index),
                      "splits": dict(Counter(r["split"] for r in index)), "megabytes": round(size/1e6, 1)}, indent=2),
          flush=True)
    if args.push:
        push()


if __name__ == "__main__":
    main()
