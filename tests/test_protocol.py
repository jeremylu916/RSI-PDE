from dataclasses import replace
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pde_rsi.evaluator import acceptance_gate, summarize
from pde_rsi.experiment import compare, continuation_probe, finalize
from pde_rsi.policy import AgentVersion, ImprovementPolicy, propose
from pde_rsi.runner import Budget, BudgetExceeded, run_task
from pde_rsi.tasks import FAMILIES, make_pilot, validate_groups, write_pilot
from pde_rsi.storage import project_path


def fake_results(tasks, version, budget):
    budget.reserve(len(tasks))
    budget.solver_calls += len(tasks)
    return [{"task_id": t.task_id, "family": t.family, "split": t.split, "status": "ok",
             "relative_l2": 1/version.solver[t.family].nx, "final_relative_l2": 1/version.solver[t.family].nx,
             "passed": True, "accuracy_pass": True, "constraints_pass": True, "resources_pass": True,
             "runtime_seconds": 0.001} for t in tasks]


class ProtocolTests(unittest.TestCase):
    def test_counts_deduplication_and_template_grouping(self):
        tasks = make_pilot()
        self.assertEqual(len(tasks), 72)
        for split in ("evolution", "development", "sealed"):
            self.assertEqual(sum(t.split == split for t in tasks), 24)
        groups = {t.template_group for t in tasks}
        self.assertEqual(len(groups), 18)
        validate_groups(tasks)
        with self.assertRaises(ValueError):
            validate_groups(tasks+[replace(tasks[0], task_id="leaked", split="sealed")])
        with self.assertRaises(ValueError):
            validate_groups(tasks+[tasks[0]])

    def test_policy_round_trip_and_lineage(self):
        initial = AgentVersion()
        clone = AgentVersion.from_dict(initial.to_dict())
        self.assertEqual(initial.version_id, clone.version_id)
        feedback = summarize(fake_results(make_pilot()[:12], initial, Budget(12, 100)))
        # Full family feedback is required for the meta-policy ranking.
        feedback["families"] = {f: {"mean_relative_l2": (i+1)*0.01} for i, f in enumerate(FAMILIES)}
        proposed = propose(initial, feedback, recursive=True)
        self.assertEqual(proposed.parent_id, initial.version_id)
        self.assertEqual(proposed.improvement.procedure, "feedback_refinement")
        self.assertEqual(initial.improvement.procedure, "uniform_refinement")
        self.assertEqual(AgentVersion.from_dict(proposed.to_dict()).version_id, proposed.version_id)

    def test_regression_gate_rejects_hidden_family_regression(self):
        tasks = [t for t in make_pilot() if t.split == "development"]
        before = fake_results(tasks, AgentVersion(), Budget(24, 100))
        after = copy.deepcopy(before)
        for result in after:
            result["relative_l2"] *= 0.5
        after[0]["relative_l2"] = before[0]["relative_l2"]*2
        gate = acceptance_gate(before, after)
        self.assertFalse(gate["accepted"])
        self.assertTrue(any("instance regression" in reason for reason in gate["reasons"]))

    def test_resource_failure_rejected_even_with_accuracy_gain(self):
        tasks = [t for t in make_pilot() if t.split == "development"]
        before = fake_results(tasks, AgentVersion(), Budget(24, 100))
        after = copy.deepcopy(before)
        for result in after:
            result["relative_l2"] *= 0.5
        after[0]["resources_pass"] = False
        self.assertFalse(acceptance_gate(before, after)["accepted"])

    def test_call_and_time_budgets(self):
        budget = Budget(1, 100)
        budget.reserve(1)
        budget.solver_calls = 1
        with self.assertRaises(BudgetExceeded):
            budget.reserve(1)

    def test_cpu_budget_prevents_another_solve(self):
        budget = Budget(2, 100, max_cpu_seconds=0.1)
        parent, child = budget.initial_cpu
        budget.initial_cpu = (parent-1, child)
        with self.assertRaises(BudgetExceeded):
            budget.reserve(1)
        self.assertFalse(budget.report()["within_resource_ceilings"])
        budget = Budget(1, 0.01)
        budget.started -= 1
        with self.assertRaises(BudgetExceeded):
            budget.reserve(1)

    def test_resource_limited_worker_and_failure_accounting(self):
        task = make_pilot()[0]
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary)/"calls.jsonl"
            budget = Budget(2, 30, ledger_path=ledger, phase="test")
            good = run_task(task, AgentVersion().solver[task.family], budget)
            self.assertEqual(good["status"], "ok", good)
            self.assertGreater(good["peak_memory_mb"], 0)
            self.assertGreater(good["cpu_seconds"], 0)
            self.assertEqual(budget.solver_calls, 1)
            tiny_timeout = replace(task, max_seconds=0.001)
            failed = run_task(tiny_timeout, AgentVersion().solver[task.family], budget)
            self.assertEqual(failed["status"], "timeout")
            self.assertEqual(budget.solver_calls, 2)
            calls = [json.loads(line) for line in ledger.read_text().splitlines()]
            self.assertEqual([call["call"] for call in calls], [1, 2])
            self.assertEqual([call["status"] for call in calls], ["ok", "timeout"])
            self.assertTrue(all(call["phase"] == "test" for call in calls))

    @patch("pde_rsi.experiment.run_suite", side_effect=fake_results)
    def test_compare_reserves_final_budget_and_does_not_load_sealed(self, _mock):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_pilot(root)
            # The evolution comparison must work without readable sealed task data.
            (root/"evaluator_private/sealed.jsonl").unlink()
            result = compare(root, root/"run", generations=2, max_calls=36, wall_seconds=100, limit_per_family=1)
            self.assertFalse(result["final_evaluated"])
            self.assertEqual(result["arms"]["single"]["budget"]["solver_calls"], 30)
            self.assertEqual(result["arms"]["fixed"]["budget"]["solver_calls"], 30)
            self.assertEqual(result["arms"]["rsi"]["budget"]["solver_calls"], 30)
            self.assertTrue(result["matched_actual_solve_calls"])
            self.assertTrue(result["complete_requested_searches"])
            self.assertEqual(result["arms"]["single"]["accepted_candidates"], 2)
            single = json.loads((root/"run/single/frozen_agent.json").read_text())["agent"]
            fixed = json.loads((root/"run/fixed/frozen_agent.json").read_text())["agent"]
            self.assertEqual(single["generation"], 0)
            self.assertEqual(single["memory"], [])
            self.assertEqual(single["improvement"], AgentVersion().to_dict()["improvement"])
            self.assertEqual(single["solver"], fixed["solver"])
            self.assertTrue(fixed["memory"])

    @patch("pde_rsi.experiment.run_suite", side_effect=fake_results)
    def test_static_control_is_explicit_and_incomplete_call_budget_rejected(self, _mock):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_pilot(root)
            result = compare(root, root/"run", generations=1, max_calls=24, wall_seconds=100,
                             limit_per_family=1, single_mode="static")
            self.assertEqual(result["arms"]["single"]["budget"]["solver_calls"], 6)
            self.assertFalse(result["matched_actual_solve_calls"])
            with self.assertRaisesRegex(ValueError, "all requested updates"):
                compare(root, root/"short", generations=2, max_calls=24, limit_per_family=1)
            self.assertFalse((root/"short").exists())

    def test_storage_rejects_home_traversal_and_symlink_escape(self):
        with self.assertRaises(ValueError):
            compare(Path.home(), Path.home()/"pde-rsi-test-output")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            escape = root/"escape"
            escape.symlink_to(Path.home(), target_is_directory=True)
            with self.assertRaises(ValueError):
                project_path(escape/"output.json")
            with self.assertRaises(ValueError):
                write_pilot(escape)

    @patch("pde_rsi.experiment.run_suite", side_effect=fake_results)
    def test_continuation_keeps_same_starting_solver_memory_and_equal_calls(self, _mock):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_pilot(root)
            start = AgentVersion()
            start.improvement = ImprovementPolicy(procedure="feedback_refinement", fractional_grading=2)
            start.memory.append({"example": "same for both branches"})
            snapshot = root/"start.json"
            snapshot.write_text(json.dumps({"version_id": start.version_id, "agent": start.to_dict()}))
            result = continuation_probe(root, snapshot, root/"probe", generations=2, max_calls=30,
                                        wall_seconds=100, limit_per_family=1)
            self.assertTrue(result["matched_actual_solve_calls"])
            self.assertTrue(result["complete_paired_continuations"])
            initial = json.loads((root/"probe/initial_policy/generation-000.json").read_text())["agent"]
            evolved = json.loads((root/"probe/evolved_policy/generation-000.json").read_text())["agent"]
            self.assertEqual(initial["solver"], evolved["solver"])
            self.assertEqual(initial["memory"], evolved["memory"])
            self.assertNotEqual(initial["improvement"], evolved["improvement"])

    @patch("pde_rsi.experiment.run_suite", side_effect=fake_results)
    def test_final_evaluation_once_and_frozen_hash_guard(self, _mock):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_pilot(root)
            compare(root, root/"run", generations=1, max_calls=24, wall_seconds=100, limit_per_family=1)
            result = finalize(root, root/"run")
            self.assertTrue(result["final_evaluated"])
            with self.assertRaises(FileExistsError):
                finalize(root, root/"run")
            compare(root, root/"tampered", generations=0, max_calls=12, wall_seconds=100, limit_per_family=1)
            path = root/"tampered/rsi/frozen_agent.json"
            record = json.loads(path.read_text())
            record["agent"]["solver"]["heat"]["nx"] = 64
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "snapshot changed"):
                finalize(root, root/"tampered")


if __name__ == "__main__":
    unittest.main()
