import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset
import pyarrow.parquet as pq

from pde_rsi.evaluator import evaluate
from pde_rsi.experiment import selected_tasks
from pde_rsi.hf_dataset import export_package, features, task_row, validate_package
from pde_rsi.solvers import Solution
from pde_rsi.storage import PROJECT_ROOT
from pde_rsi.tasks import FAMILIES, Task


class HuggingFaceDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="hf-dataset-test-")
        cls.work = Path(cls.temporary.name)
        cls.package = cls.work/"package"
        with patch("pde_rsi.hf_dataset.selected_tasks", wraps=selected_tasks) as selector:
            cls.manifest = export_package(PROJECT_ROOT, cls.package)
            cls.source_splits_read = [call.args[1] for call in selector.call_args_list]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_six_hf_configs_load_with_typed_columns_and_grouped_splits(self):
        report = validate_package(self.package)
        self.assertEqual(report["total_cases"], 48)
        self.assertEqual(set(report["configs"]), set(FAMILIES))
        self.assertTrue(all(counts == {"train": 4, "validation": 4} for counts in report["configs"].values()))
        schema = pq.read_schema(self.package/"data/poisson/train.parquet")
        self.assertEqual(str(schema.field("id").type), "string")
        self.assertIn("struct", str(schema.field("oracle_config").type))
        self.assertIn("double", str(schema.field("oracle_config").type))

    def test_export_never_selects_private_cases_or_ships_generator_reference_rsi(self):
        self.assertEqual(self.source_splits_read, ["evolution", "development"])
        self.assertFalse(self.manifest["reserved_final_cases_included"])
        for path in (self.package/"solver_bundle").rglob("*.py"):
            text = path.read_text()
            self.assertNotIn("make_pilot", text)
            self.assertNotIn("ImprovementPolicy", text)
            self.assertNotIn("evaluator_private", text)
            self.assertNotIn("from .reference", text)
        with self.assertRaisesRegex(ValueError, "Only public"):
            task = selected_tasks(PROJECT_ROOT, "evolution")[0]
            from dataclasses import replace
            task_row(replace(task, split="sealed"), 1, "invalid")

    def test_native_contract_matches_domain_fractional_operator_and_inapplicable_fields(self):
        poisson = pq.read_table(self.package/"data/poisson/train.parquet").to_pylist()[0]
        self.assertIsNone(poisson["oracle_config"]["domain"]["time"])
        self.assertIsNone(poisson["oracle_config"]["pde"]["coefficients"]["velocity"])
        self.assertEqual(poisson["evaluation_config"]["sampling"]["time_points"], 1)
        fractional = pq.read_table(self.package/"data/multi_term/validation.parquet").to_pylist()[0]
        operator = fractional["oracle_config"]["pde"]["time_operator"]
        self.assertEqual(operator["type"], "caputo")
        self.assertEqual(operator["orders"], fractional["task_record"]["orders"])
        self.assertEqual(operator["weights"], fractional["task_record"]["weights"])
        self.assertEqual(operator["history_start"], 0.0)
        self.assertEqual(fractional["provenance"]["source_split"], "development")

    def test_detects_problem_mismatch_even_when_file_checksums_are_updated(self):
        modified = self.work/"mismatched-package"
        shutil.copytree(self.package, modified)
        parquet = modified/"data/poisson/train.parquet"
        records = pq.read_table(parquet).to_pylist()
        records[0]["oracle_config"]["pde"]["coefficients"]["kappa"]["value"] *= 2
        Dataset.from_list(records, features=features()).to_parquet(parquet)
        mirror = modified/"records/poisson/train.jsonl"
        mirror.write_text("".join(json.dumps(row)+"\n" for row in records))
        manifest = json.loads((modified/"manifest.json").read_text())
        for path in (parquet, mirror):
            manifest["files"][str(path.relative_to(modified))] = hashlib.sha256(path.read_bytes()).hexdigest()
        (modified/"manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "native task contract"):
            validate_package(modified)

    def test_standalone_solver_bundle_runs_all_families_and_passes_independent_accuracy(self):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for family in FAMILIES:
            with self.subTest(family=family):
                data = self.package/f"data/{family}/validation.parquet"
                row = pq.read_table(data).to_pylist()[0]
                output = self.work/f"solution-{family}"
                result = subprocess.run(
                    [sys.executable, str(self.package/"solver_bundle/run_case.py"), "--data", str(data),
                     "--id", row["id"], "--nx", "64", "--nt", "96", "--grading", "2",
                     "--output", str(output)], env=env, cwd=self.work,
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                metrics = evaluate(Task.from_dict(row["task_record"]), Solution.load_npz(output/"solution.npz"))
                self.assertTrue(metrics["accuracy_pass"], metrics)
                self.assertTrue(metrics["constraints_pass"], metrics)
                record = json.loads((output/"run.json").read_text())
                self.assertFalse(record["resources_certified"])

    def test_existing_release_is_preserved_and_extra_files_are_rejected(self):
        with self.assertRaises(FileExistsError):
            export_package(PROJECT_ROOT, self.package)
        extra = self.work/"extra-file-package"
        shutil.copytree(self.package, extra)
        (extra/"unexpected.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "unexpected files"):
            validate_package(extra)


if __name__ == "__main__":
    unittest.main()
