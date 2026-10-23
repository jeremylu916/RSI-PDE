"""Independent L2 checks and candidate-validation guards using public cases only."""

import copy
from dataclasses import replace
import math
from pathlib import Path
import unittest

from pde_rsi.evaluator import EVALUATOR_VERSION, acceptance_gate, evaluate
from pde_rsi.reference import output_nodes, reference
from pde_rsi.solvers import Solution
from pde_rsi.tasks import load_tasks


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class L2MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.public_tasks = [
            task
            for split in ("evolution", "development")
            for task in load_tasks(PROJECT_ROOT / "benchmarks" / f"{split}.jsonl")
        ]

    def task(self, family):
        return next(
            task for task in self.public_tasks
            if task.family == family and task.split == "evolution"
        )

    def test_exact_references_have_zero_absolute_and_relative_l2(self):
        self.assertEqual(len(self.public_tasks), 48)
        for task in self.public_tasks:
            with self.subTest(task_id=task.task_id):
                x, t = output_nodes(task)
                metrics = evaluate(
                    task, Solution(x, t, reference(task, x, t), "test exact reference")
                )
                for field in (
                    "absolute_l2", "final_absolute_l2",
                    "relative_l2", "final_relative_l2",
                ):
                    self.assertAlmostEqual(metrics[field], 0.0, places=13)
                self.assertEqual(metrics["evaluator_version"], EVALUATOR_VERSION)

    def test_scaled_error_matches_independent_analytic_norms(self):
        scale = 0.01
        for family in ("poisson", "heat"):
            # Dense time sampling makes the trapezoidal heat norm agree with its
            # analytic integral to 1e-8. Expected norms do not reuse evaluator code.
            task = replace(self.task(family), output_time_points=1025)
            self.assertEqual(task.profile, "sine")
            self.assertEqual(task.mode, 1)
            if family == "heat":
                self.assertEqual(task.beta, 1.0)
            x, t = output_nodes(task, "legacy")  # the record's 1025 times; the fine grid fixes 500
            solution = Solution(
                x, t, (1 + scale) * reference(task, x, t),
                "test scaled reference",
            )
            metrics = evaluate(task, solution, grid="legacy")
            spatial_norm = task.amplitude * math.sqrt(task.length / 2)
            if family == "poisson":
                integrated_norm = final_norm = spatial_norm
            else:
                horizon = task.final_time
                integrated_norm = spatial_norm * math.sqrt(
                    horizon + horizon**2 + horizon**3 / 3
                )
                final_norm = spatial_norm * (1 + horizon)

            self.assertAlmostEqual(
                metrics["absolute_l2"], scale * integrated_norm,
                delta=scale * integrated_norm * 1e-8,
            )
            self.assertAlmostEqual(
                metrics["final_absolute_l2"], scale * final_norm, places=13,
            )
            self.assertAlmostEqual(metrics["relative_l2"], scale, places=13)
            self.assertAlmostEqual(metrics["final_relative_l2"], scale, places=13)
            if family == "poisson":
                self.assertEqual(metrics["absolute_l2"], metrics["final_absolute_l2"])

    def results(self):
        tasks = [task for task in self.public_tasks if task.family == "heat"][:2]
        return [
            {
                "task_id": task.task_id, "family": task.family, "split": task.split,
                "status": "ok", "relative_l2": 0.001,
                "final_relative_l2": 0.001, "absolute_l2": 0.0001,
                "final_absolute_l2": 0.0001, "passed": True,
                "accuracy_pass": True, "constraints_pass": True,
                "resources_pass": True, "runtime_seconds": 0.001,
                "evaluator_version": EVALUATOR_VERSION,
            }
            for task in tasks
        ]

    def test_final_time_regression_rejected_despite_aggregate_gain(self):
        baseline = self.results()
        candidate = copy.deepcopy(baseline)
        for result in candidate:
            result["relative_l2"] *= 0.5
            result["final_relative_l2"] *= 0.25
        # Both cases still pass the tolerance; even the family-average final
        # error improves. A paired check must detect this one 40% regression.
        candidate[0]["final_relative_l2"] = 0.0014
        gate = acceptance_gate(baseline, candidate)
        self.assertGreater(gate["relative_gain"], 0.4)
        self.assertFalse(gate["accepted"])
        self.assertTrue(
            any("final" in reason.lower() for reason in gate["reasons"]),
            gate["reasons"],
        )

    def test_duplicate_task_ids_rejected_in_either_comparison_arm(self):
        baseline = self.results()
        candidate = copy.deepcopy(baseline)
        for arm in ("baseline", "candidate"):
            before, after = copy.deepcopy(baseline), copy.deepcopy(candidate)
            selected = before if arm == "baseline" else after
            selected.append(copy.deepcopy(selected[0]))
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                acceptance_gate(before, after)

    def test_mixed_evaluator_versions_rejected_within_and_across_arms(self):
        baseline = self.results()
        for mismatch in ("within_candidate", "across_arms"):
            candidate = copy.deepcopy(baseline)
            changed = candidate[:1] if mismatch == "within_candidate" else candidate
            for result in changed:
                result["evaluator_version"] = EVALUATOR_VERSION + "-different"
            with self.subTest(mismatch=mismatch), self.assertRaises(ValueError):
                acceptance_gate(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
