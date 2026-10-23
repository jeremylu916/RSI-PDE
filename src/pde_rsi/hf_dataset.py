"""Typed, public-only Hugging Face dataset export; no RSI or Hub writes."""

import hashlib
import inspect
import json
from pathlib import Path
import shutil

from .experiment import selected_tasks
from .policy import SolverSettings
from .storage import PROJECT_ROOT, project_path
from .tasks import FAMILIES, GROUP_FAMILIES, WAVE_FAMILIES, Task, wave_frequency

DATASET_VERSION = "pde-problems-public-v1"
SPLIT_MAP = {"evolution": "train", "development": "validation"}
METHODS = {
    "poisson": "centered finite difference Poisson",
    "heat": "backward Euler",
    "convection_diffusion": "backward Euler with centered transport",
    "burgers": "IMEX Burgers",
    "caputo": "nonuniform Caputo L1",
    "multi_term": "weighted nonuniform Caputo L1",
}


def features():
    """Explicit scalar/list/struct types, including empty lists and nullable fields."""
    from datasets import Features, Value
    string, real, integer, boolean = (Value(t) for t in ("string", "float64", "int64", "bool"))
    coefficient = {"type": string, "value": real, "expr": string}
    expression = {"type": string, "expr": string}
    boundary = {"location": string, "type": string, "coordinate": real, "value": real}
    return Features({
        "id": string,
        "pde_classification": {
            "equation_type": string, "math_type": string, "linearity": string,
            "spatial_dimension": integer, "time_dependent": boolean,
            "fractional": boolean, "initial_time_regularity": string,
        },
        "oracle_config": {
            "pde": {"type": string, "equation": string,
                    "coefficients": {"kappa": coefficient, "velocity": coefficient},
                    "forcing": expression,
                    "time_operator": {"type": string, "orders": [real], "weights": [real],
                                      "history_start": real, "definition": string},
                    "space_operator": {"type": string, "definition": string, "exterior_conditions": string}},
            "domain": {"geometry": string,
                       "axes": [{"name": string, "lower": real, "upper": real}],
                       "time": {"start": real, "end": real}},
            "initial_condition": expression,
            "boundary_conditions": [boundary],
            "reference": {"type": string, "access": string, "solution_provided": boolean},
        },
        "evaluation_config": {
            "metrics": [string], "relative_l2_tolerance": real,
            "boundary_tolerance": real, "initial_tolerance": real,
            "sampling": {"space_points": integer, "time_points": integer, "interpolation": string},
            "compute_limits": {"wall_seconds": real, "memory_mb": integer,
                               "cpu_threads": integer, "gpus": integer},
            "solution_format": string, "reference_access": string,
        },
        "problem_statement": string,
        "solver_config": {"method": string, "implementation": string, "entrypoint": string,
                          "nx": integer, "nt": integer, "grading": real, "advection": string},
        "provenance": {"source_id": string, "source_kind": string, "source_split": string,
                       "template_group": string, "source_file": string, "source_line": integer,
                       "source_sha256": string, "reuse_permission": string},
        "task_record": {"task_id": string, "family": string, "template_group": string, "split": string,
                        "length": real, "final_time": real, "diffusivity": real, "velocity": real,
                        "amplitude": real, "mode": integer, "profile": string, "beta": real,
                        "orders": [real], "weights": [real], "relative_l2_tolerance": real,
                        "boundary_tolerance": real, "initial_tolerance": real, "max_seconds": real,
                        "memory_mb": integer, "output_space_points": integer, "output_time_points": integer,
                        "source_id": string, "reuse_permission": string},
    })


def task_row(task, source_line, source_sha256):
    """Map a public task into the screenshot's classification/oracle/evaluation columns."""
    if task.split not in SPLIT_MAP:
        raise ValueError("Only public evolution/development tasks may be exported")
    public = task.public_record()
    fractional = bool(task.orders)
    temporal = task.family != "poisson"
    classification = {
        "equation_type": task.family,
        "math_type": "elliptic" if not temporal else "time_fractional_diffusion" if fractional else "parabolic",
        "linearity": "nonlinear" if task.family == "burgers" else "linear",
        "spatial_dimension": 1, "time_dependent": temporal, "fractional": fractional,
        "initial_time_regularity": "not_applicable" if not temporal else
                                   "weak_initial_time_regularity" if task.beta < 1 else "smooth",
    }
    oracle = {
        "pde": {
            "type": task.family, "equation": public["equation"],
            "coefficients": {
                "kappa": {"type": "constant", "value": task.diffusivity, "expr": None},
                "velocity": {"type": "constant", "value": task.velocity, "expr": None}
                            if task.family == "convection_diffusion" else None,
            },
            "forcing": {"type": "expression", "expr": public["forcing_expression"]},
            "time_operator": {
                "type": "caputo" if fractional else "first_derivative" if temporal else "none",
                "orders": list(task.orders) if fractional else [1.0] if temporal else [],
                "weights": list(task.weights) if fractional else [1.0] if temporal else [],
                "history_start": 0.0 if fractional else None,
                "definition": "D_C^alpha u(t) = integral_0^t (t-s)^(-alpha) u'(s) ds / Gamma(1-alpha)"
                              if fractional else "du/dt" if temporal else "not applicable",
            },
            "space_operator": {"type": "local_laplacian", "definition": "d^2u/dx^2",
                               "exterior_conditions": "not applicable to local spatial Laplacian"},
        },
        "domain": {"geometry": "interval", "axes": [{"name": "x", "lower": 0.0, "upper": task.length}],
                   "time": {"start": 0.0, "end": task.final_time} if temporal else None},
        "initial_condition": {"type": "expression", "expr": public["initial_expression"]}
                             if temporal else {"type": "not_applicable", "expr": None},
        "boundary_conditions": [
            {"location": "left", "type": "Dirichlet", "coordinate": 0.0, "value": 0.0},
            {"location": "right", "type": "Dirichlet", "coordinate": task.length, "value": 0.0},
        ],
        "reference": {"type": "manufactured_analytic", "access": "evaluator_only", "solution_provided": False},
    }
    evaluation = {
        "metrics": ["space_time_relative_l2", "final_time_relative_l2", "absolute_linf",
                    "initial_error", "boundary_error"],
        "relative_l2_tolerance": task.relative_l2_tolerance,
        "boundary_tolerance": task.boundary_tolerance, "initial_tolerance": task.initial_tolerance,
        "sampling": {"space_points": task.output_space_points,
                     "time_points": task.output_time_points if temporal else 1,
                     "interpolation": "linear_to_common_evaluator_nodes"},
        "compute_limits": {"wall_seconds": task.max_seconds, "memory_mb": task.memory_mb,
                           "cpu_threads": 1, "gpus": 0},
        "solution_format": "npz: x, t, u, scalar method", "reference_access": "evaluator_only",
    }
    statement = (f"Solve {public['equation']} on x in [0,{task.length:g}]"
                 + (f", t in [0,{task.final_time:g}]" if temporal else "")
                 + f" with diffusivity={task.diffusivity:g}"
                 + (f", velocity={task.velocity:g}" if task.family == "convection_diffusion" else "")
                 + f". Forcing f={public['forcing_expression']}. Boundary conditions: u(0)=u({task.length:g})=0."
                 + (f" Initial condition: u(x,0)={public['initial_expression']}." if temporal else "")
                 + (f" Caputo orders={list(task.orders)}, weights={list(task.weights)}, history starts at 0." if fractional else "")
                 + f" Return x,t,u,method; relative L2 tolerance {task.relative_l2_tolerance:g}.")
    return {
        "id": task.task_id, "pde_classification": classification, "oracle_config": oracle,
        "evaluation_config": evaluation, "problem_statement": statement,
        "solver_config": {"method": METHODS[task.family],
                          "implementation": "solver_bundle/pde_baseline/solvers.py", "entrypoint": "solve",
                          **vars(SolverSettings())},
        "provenance": {"source_id": task.source_id, "source_kind": "self_generated_manufactured",
                       "source_split": task.split, "template_group": task.template_group,
                       "source_file": f"benchmarks/{task.split}.jsonl", "source_line": source_line,
                       "source_sha256": source_sha256, "reuse_permission": task.reuse_permission},
        "task_record": task.to_dict(),
    }


def write_json(path, data):
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def bundle_solvers(root, output):
    """Export public numerical modules, omitting task generators and all RSI/reference code."""
    bundle = output/"solver_bundle/pde_baseline"
    bundle.mkdir(parents=True)
    (bundle/"__init__.py").write_text('"""Portable classical solvers for the public PDE problems."""\n'
                                   'import sys\nsys.dont_write_bytecode = True\n')
    task_preamble = ('from dataclasses import asdict, dataclass\nimport math\n\n'
                    f'FAMILIES = {FAMILIES!r}\nWAVE_FAMILIES = {WAVE_FAMILIES!r}\nGROUP_FAMILIES = {GROUP_FAMILIES!r}\n'
                    'ALL_FAMILIES = FAMILIES + WAVE_FAMILIES + GROUP_FAMILIES\nSPLITS = ("evolution", "development")\n\n')
    (bundle/"tasks.py").write_text(task_preamble+inspect.getsource(Task)+"\n\n"+inspect.getsource(wave_frequency))
    (bundle/"policy.py").write_text('from dataclasses import dataclass\n\n'+inspect.getsource(SolverSettings)+"\n")
    for name in ("equations.py", "equations2d.py", "equations_group.py", "solvers.py"):
        shutil.copyfile(root/"src/pde_rsi"/name, bundle/name)
    (bundle/"storage.py").write_text('"""Portable path normalization; output location is chosen by the caller."""\n'
                                   'from pathlib import Path\n\ndef project_path(path):\n    return Path(path).resolve()\n')
    template_root = root/"templates/huggingface"
    for name in ("run_case.py", "SOLVERS.md", "requirements.txt"):
        shutil.copyfile(template_root/name, output/"solver_bundle"/name)


def export_package(root, output):
    from datasets import Dataset  # imported lazily; optional packaging dependency
    import yaml
    root, output = project_path(root), project_path(output)
    if root != PROJECT_ROOT:
        raise ValueError("Export the canonical depot project only")
    if output.exists():
        raise FileExistsError("Choose a new package directory; existing releases are never overwritten")
    schema = features()
    by_family = {family: {split: [] for split in SPLIT_MAP.values()} for family in FAMILIES}
    source_hashes = {}
    for source_split, target_split in SPLIT_MAP.items():
        tasks = selected_tasks(root, source_split)
        path = root/f"benchmarks/{source_split}.jsonl"
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        source_hashes[str(path.relative_to(root))] = sha
        for source_line, task in enumerate(tasks, 1):
            by_family[task.family][target_split].append(task_row(task, source_line, sha))
    output.mkdir(parents=True)
    config_entries, rows = [], []
    for family, splits in by_family.items():
        config = {"config_name": family, "data_files": []}
        if family == "poisson":
            config["default"] = True
        for split, records in splits.items():
            relative = f"data/{family}/{split}.parquet"
            path = output/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Dataset.from_list(records, features=schema).to_parquet(str(path))
            json_path = output/f"records/{family}/{split}.jsonl"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n" for row in records))
            config["data_files"].append({"split": split, "path": relative})
            rows.extend(records)
        config_entries.append(config)
    write_json(output/"features.json", schema.to_dict())
    bundle_solvers(root, output)
    metadata = {"pretty_name": "PDE and FPDE Problems — Public Manufactured Pilot",
                "language": ["en"], "tags": ["pde", "fpde", "scientific-computing", "tabular", "datasets"],
                "size_categories": ["n<1K"], "configs": config_entries}
    template = root/"templates/huggingface"
    (output/"README.md").write_text("---\n"+yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True)+"---\n\n"
                                  +(template/"DATASET_CARD.md").read_text())
    shutil.copyfile(template/"SCHEMA.md", output/"SCHEMA.md")
    from .hf_preview import write_preview
    write_preview(output/"preview.html", rows)
    manifest = {"dataset_version": DATASET_VERSION, "format": "typed_parquet_with_jsonl_mirrors",
                "total_public_cases": len(rows), "split_mapping": SPLIT_MAP,
                "configs": {family: {split: len(items) for split, items in splits.items()}
                            for family, splits in by_family.items()},
                "source_sha256": source_hashes,
                "references_included": False, "reserved_final_cases_included": False,
                "published": False, "files": {}}
    manifest["files"] = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output/"manifest.json", manifest)
    return {"output": str(output), **manifest}


def validate_package(directory):
    """Read all six configs through Hugging Face, checking hashes and private-data exclusion."""
    from datasets import Features, load_dataset
    import yaml
    directory = project_path(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    actual_files = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if actual_files != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Package has missing or unexpected files")
    for relative, expected in manifest["files"].items():
        path = project_path(directory/relative)
        if not path.is_relative_to(directory) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Package checksum mismatch or path escape: {relative}")
    card = yaml.safe_load((directory/"README.md").read_text().split("---", 2)[1])
    if {c["config_name"] for c in card["configs"]} != set(FAMILIES):
        raise ValueError("The card must declare the six implemented PDE configurations")
    expected_schema = Features.from_dict(json.loads((directory/"features.json").read_text()))
    seen_ids, groups, counts = set(), {}, {}
    for family in FAMILIES:
        loaded = load_dataset(str(directory), family, cache_dir=str(PROJECT_ROOT/".runtime/cache/huggingface/validation"))
        if set(loaded) != {"train", "validation"}:
            raise ValueError("Only train and validation splits are permitted in a public release")
        counts[family] = {}
        for split, data in loaded.items():
            if data.features != expected_schema or len(data) != manifest["configs"][family][split]:
                raise ValueError(f"Feature schema or row count mismatch: {family}/{split}")
            counts[family][split] = len(data)
            mirror = [json.loads(line) for line in (directory/f"records/{family}/{split}.jsonl").read_text().splitlines()]
            # JSON distinguishes null structs; Arrow preserves them through the typed schema.
            if list(data) != json.loads(json.dumps(mirror)):
                raise ValueError("Parquet rows differ from their reviewable JSONL mirrors")
            for row in data:
                native = Task.from_dict(row["task_record"])
                if row["id"] != native.task_id or row["id"] in seen_ids or native.family != family:
                    raise ValueError("Duplicated ID or invalid family/task mapping")
                if native.split not in SPLIT_MAP or SPLIT_MAP[native.split] != split:
                    raise ValueError("Reserved or incorrectly mapped source task in release")
                group = row["provenance"]["template_group"]
                if group in groups and groups[group] != split:
                    raise ValueError("A related template crosses train/validation")
                groups[group] = split
                seen_ids.add(row["id"])
                if row["oracle_config"]["reference"]["solution_provided"]:
                    raise ValueError("An evaluator-only solution was exported")
                if row["oracle_config"]["pde"]["equation"] != native.public_record()["equation"]:
                    raise ValueError("Exported equation does not match the public native task")
                expected = task_row(native, row["provenance"]["source_line"], row["provenance"]["source_sha256"])
                if row != json.loads(json.dumps(expected)):
                    raise ValueError("Problem/evaluation/solver fields differ from the native task contract")
    if len(seen_ids) != manifest["total_public_cases"]:
        raise ValueError("Total dataset count differs from the manifest")
    return {"validated": True, "output": str(directory), "total_cases": len(seen_ids), "configs": counts,
            "private_cases_included": False, "exact_solutions_included": False,
            "local_huggingface_loading": "passed", "published": False}
