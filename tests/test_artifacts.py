import os
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from pde_rsi.evaluator import evaluate
from pde_rsi.policy import SolverSettings
from pde_rsi.solvers import Solution, solve
from pde_rsi.storage import ENV_PREFIX, PROJECT_ROOT, doctor
from pde_rsi.tasks import make_pilot


class ArtifactTests(unittest.TestCase):
    def test_depot_interpreter_and_temporary_files(self):
        report = doctor()
        self.assertEqual(Path(report["python_prefix"]), ENV_PREFIX)
        self.assertTrue(Path(tempfile.gettempdir()).is_relative_to(PROJECT_ROOT))
        self.assertFalse(report["user_site_enabled"])
        self.assertFalse(report["bytecode_writes"])
        self.assertEqual(os.environ["CONDA_REGISTER_ENVS"], "false")

    def test_solution_round_trip_preserves_independent_metrics(self):
        for family in ("poisson", "caputo"):
            task = next(t for t in make_pilot() if t.family == family and t.split == "development")
            solution = solve(task, SolverSettings(nx=64, nt=96, grading=2))
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary)/"solution.npz"
                solution.save_npz(path)
                restored = Solution.load_npz(path)
                np.testing.assert_array_equal(solution.u, restored.u)
                self.assertEqual(evaluate(task, solution), evaluate(task, restored))
                with self.assertRaises(FileExistsError):
                    solution.save_npz(path)

    def test_npz_rejects_object_arrays_and_malformed_outputs(self):
        task = next(t for t in make_pilot() if t.family == "heat")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"object.npz"
            np.savez(path, x=np.array([object()], dtype=object), t=[0], u=[[0]], method="invalid")
            with self.assertRaises(ValueError):
                Solution.load_npz(path)
            path = Path(temporary)/"bad-shape.npz"
            np.savez(path, x=[0, 0.5, 1], t=[0, 0.5, 1], u=[[0]], method="invalid")
            with self.assertRaises(ValueError):
                evaluate(task, Solution.load_npz(path))

    def test_cli_rejects_outside_output_before_creating_it(self):
        path = Path.home()/"PDE-RSI-rejected-output-test"
        self.assertFalse(path.exists())
        result = subprocess.run([str(PROJECT_ROOT/"scripts/run.sh"), "baseline", "--output", str(path)],
                                capture_output=True, text=True, cwd=PROJECT_ROOT)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("Path must remain inside", result.stderr)
        self.assertFalse(path.exists())

    def test_conda_transaction_leaves_home_catalog_untouched(self):
        catalog = Path.home()/".conda/environments.txt"
        before = (catalog.stat().st_mtime_ns, hashlib.sha256(catalog.read_bytes()).hexdigest())
        specification = PROJECT_ROOT/"configs/conda-cluster-explicit.txt"
        package = next(line for line in specification.read_text().splitlines() if "/_libgcc_mutex-" in line)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/"envs", prefix="storage-check-") as prefix:
            with tempfile.TemporaryDirectory() as temporary:
                spec = Path(temporary)/"minimal-explicit.txt"
                spec.write_text("@EXPLICIT\n"+package+"\n")
                result = subprocess.run(["bash", str(PROJECT_ROOT/"scripts/conda.sh"), "create",
                                         "--prefix", prefix, "--offline", "--file", str(spec), "--yes"],
                                        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue((Path(prefix)/"conda-meta/history").is_file())
        after = (catalog.stat().st_mtime_ns, hashlib.sha256(catalog.read_bytes()).hexdigest())
        self.assertEqual(before, after)
        self.assertTrue((PROJECT_ROOT/".runtime/conda/environments.txt").is_file())


if __name__ == "__main__":
    unittest.main()
