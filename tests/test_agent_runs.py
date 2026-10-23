"""Agent-state, experiment-budget, and failure tests without model/solver calls."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pde_rsi.agent_runs import (
    INITIAL_WORKFLOW, LoopVersion, apply_proposal, evaluate_suite,
    run_arm, run_comparison,
)
from pde_rsi.evaluator import EVALUATOR_VERSION
from pde_rsi.model_backend import ModelCallError, ModelConfig
from pde_rsi.runner import Budget, BudgetExceeded
from pde_rsi.storage import PROJECT_ROOT
from pde_rsi.tasks import FAMILIES, load_tasks


class FakeBackend:
    def __init__(self, config, reject_first=False, fail=False):
        self.config = config
        self.contexts = []
        self.reject_first = reject_first
        self.fail = fail

    def complete(self, messages, schema, directory, timeout):
        context = next(
            json.loads(message["content"])
            for message in reversed(messages)
            if message["role"] == "user"
            and "active_agent" in json.loads(message["content"])
        )
        self.contexts.append(copy.deepcopy(context))
        if self.fail:
            raise ModelCallError("Mock provider failure", call_attempted=True, wall_seconds=0.001)
        recursive = "You may also propose" in messages[0]["content"]
        record = {
            "hypothesis": "Refine the tested spatial grid within the declared allowance.",
            "solver": [],
            "next_improvement_policy": f"Accepted workflow {len(self.contexts)}" if recursive else None,
            "memory_update": "Use the independently checked numerical feedback.",
        }
        for family in FAMILIES:
            settings = copy.deepcopy(context["active_agent"]["agent"]["solver"][family])
            settings["nx"] = (
                max(8, settings["nx"] // 2)
                if self.reject_first and len(self.contexts) == 1
                else min(128, settings["nx"] + 16)
            )
            record["solver"].append({"family": family, **settings})
        return {
            "proposal": record,
            "usage": {"prompt_tokens": 10, "completion_tokens": 10,
                      "total_tokens": 20, "available": True},
            "resolved_model": self.config.model,
            "wall_seconds": 0.001,
        }


def fake_evaluation(tasks, version, budget, solution_directory=None):
    budget.reserve(len(tasks))
    budget.solver_calls += len(tasks)
    return [
        {
            "task_id": task.task_id, "family": task.family, "split": task.split,
            "status": "ok", "relative_l2": 0.02 / version.solver[task.family].nx,
            "final_relative_l2": 0.02 / version.solver[task.family].nx,
            "absolute_l2": 0.002 / version.solver[task.family].nx,
            "final_absolute_l2": 0.002 / version.solver[task.family].nx,
            "passed": True, "accuracy_pass": True, "constraints_pass": True,
            "resources_pass": True, "runtime_seconds": 0.001,
            "wall_seconds": 0.001, "cpu_seconds": 0.001, "peak_memory_mb": 1.0,
            "evaluator_version": EVALUATOR_VERSION,
        }
        for task in tasks
    ]


class AgentRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = {
            split: load_tasks(PROJECT_ROOT / "benchmarks" / f"{split}.jsonl")
            for split in ("evolution", "development")
        }
        cls.config = ModelConfig("http://localhost:9999/v1", "mock-frozen-model")

    def proposal(self, policy=None):
        return {
            "hypothesis": "Refine based on supplied diagnostics.",
            "solver": [
                {"family": family, "nx": 48, "nt": 64, "grading": 1.0,
                 "advection": "central"}
                for family in FAMILIES
            ],
            "next_improvement_policy": policy,
            "memory_update": "A bounded numerical lesson.",
        }

    def test_fixed_controller_forbids_policy_updates_and_preserves_memory(self):
        initial = LoopVersion("mock", improvement_workflow="Fixed instructions", memory=[{"lesson": "existing"}])
        with self.assertRaises(ValueError):
            apply_proposal(initial, self.proposal("New instructions"), recursive=False)
        candidate = apply_proposal(initial, self.proposal(), recursive=False)
        self.assertEqual(candidate.improvement_workflow, initial.improvement_workflow)
        self.assertEqual(candidate.memory, initial.memory)
        self.assertEqual(candidate.generation, initial.generation)
        self.assertEqual(candidate.parent_id, initial.parent_id)
        self.assertEqual(initial.solver["heat"].nx, 32)
        self.assertEqual(candidate.solver["heat"].nx, 48)

    def test_proposals_require_six_unique_bounded_solver_settings(self):
        malformed = []
        missing = self.proposal()
        missing["solver"].pop()
        malformed.append(missing)
        duplicate = self.proposal()
        duplicate["solver"][-1]["family"] = duplicate["solver"][0]["family"]
        malformed.append(duplicate)
        for field, value in (("nx", 129), ("nt", 385), ("nx", True), ("grading", 4.1), ("advection", "invalid")):
            record = self.proposal()
            record["solver"][0][field] = value
            malformed.append(record)
        for index, record in enumerate(malformed):
            with self.subTest(index=index), self.assertRaises(ValueError):
                apply_proposal(LoopVersion("mock"), record, recursive=True)

    def test_recursive_candidates_inherit_workflow_memory_and_lineage(self):
        initial = LoopVersion("mock")
        first = apply_proposal(initial, self.proposal("Refine the weakest family first."), recursive=True)
        second = apply_proposal(first, self.proposal(), recursive=True)
        self.assertEqual(first.parent_id, initial.version_id)
        self.assertEqual(second.parent_id, first.version_id)
        self.assertEqual(second.generation, 2)
        self.assertEqual(second.improvement_workflow, first.improvement_workflow)
        self.assertEqual(len(first.memory), 1)
        self.assertEqual(len(second.memory), 2)
        self.assertEqual(second.memory[0], first.memory[0])
        self.assertEqual(initial.memory, [])
        self.assertEqual(initial.improvement_workflow, INITIAL_WORKFLOW)

    def test_comparison_matches_allowances_covers_all_cases_and_inherits_policy(self):
        backends = []

        def factory(config):
            backend = FakeBackend(config)
            backends.append(backend)
            return backend

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".runtime/tmp") as temporary:
            root = Path(temporary)
            package = root / "package"
            package.mkdir()
            (package / "manifest.json").write_text("{}\n")
            output = root / "run"
            with patch("pde_rsi.agent_runs.public_tasks", return_value=self.tasks), patch(
                "pde_rsi.agent_runs.evaluate_suite", side_effect=fake_evaluation
            ), patch("pde_rsi.agent_runs.run_task", side_effect=AssertionError("Unexpected real solve")):
                report = run_comparison(
                    root, package, output, self.config, rounds=2, probe_rounds=1,
                    backend_factory=factory, wall_seconds=100, cpu_seconds=100,
                )
            self.assertTrue(report["matched_actual_solver_calls"])
            self.assertTrue(report["matched_actual_model_calls"])
            self.assertTrue(report["all_requested_rounds_complete"])
            self.assertFalse(report["sealed_cases_used"])
            for arm in ("single", "rsi"):
                arm_report = report["arms"][arm]
                self.assertEqual(arm_report["budget"]["solver_calls"], 72 + 48 * 2)
                self.assertEqual(arm_report["budget"]["model_calls"], 2)
                frozen = json.loads((output / arm / "frozen_results.json").read_text())
                self.assertEqual(len(frozen), 48)
                self.assertEqual({row["task_id"] for row in frozen}, {
                    task.task_id for tasks in self.tasks.values() for task in tasks
                })
            initial_single = json.loads((output / "single/initial_agent.json").read_text())
            initial_rsi = json.loads((output / "rsi/initial_agent.json").read_text())
            self.assertEqual(initial_single, initial_rsi)
            single = json.loads((output / "single/frozen_agent.json").read_text())["agent"]
            self.assertEqual(single["improvement_workflow"], INITIAL_WORKFLOW)
            self.assertEqual(single["memory"], [])
            self.assertEqual(single["generation"], 0)
            self.assertEqual(backends[1].contexts[1]["active_improvement_policy"], "Accepted workflow 1")
            self.assertEqual(len(backends[1].contexts[1]["active_agent"]["agent"]["memory"]), 1)
            initial_probe = json.loads((output / "continuation/initial_policy/initial_agent.json").read_text())["agent"]
            evolved_probe = json.loads((output / "continuation/evolved_policy/initial_agent.json").read_text())["agent"]
            self.assertNotEqual(initial_probe["improvement_workflow"], evolved_probe["improvement_workflow"])
            initial_probe.pop("improvement_workflow")
            evolved_probe.pop("improvement_workflow")
            self.assertEqual(initial_probe, evolved_probe)
            self.assertTrue(report["continuation_matched_actual_calls"])
            for arm in report["continuation"].values():
                self.assertEqual(arm["budget"]["solver_calls"], 72 + 48)
                self.assertEqual(arm["budget"]["model_calls"], 1)

    def test_rejected_numeric_proposal_does_not_update_active_workflow(self):
        backend = FakeBackend(self.config, reject_first=True)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".runtime/tmp") as temporary, patch(
            "pde_rsi.agent_runs.evaluate_suite", side_effect=fake_evaluation
        ):
            output = Path(temporary) / "arm"
            report, version, _ = run_arm(
                output, LoopVersion(self.config.model), self.tasks["evolution"],
                self.tasks["development"], 2, True, Budget(168, 100), backend, 2,
            )
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertFalse(events[0]["gate"]["accepted"])
            self.assertEqual(backend.contexts[1]["active_improvement_policy"], INITIAL_WORKFLOW)
            self.assertEqual(backend.contexts[1]["active_agent"]["agent"]["memory"], [])
            self.assertEqual(report["accepted_candidates"], 1)
            self.assertEqual(version.improvement_workflow, "Accepted workflow 2")
            self.assertEqual(version.generation, 1)

    def test_failed_model_call_consumes_attempted_slot_without_success(self):
        backend = FakeBackend(self.config, fail=True)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".runtime/tmp") as temporary, patch(
            "pde_rsi.agent_runs.evaluate_suite", side_effect=fake_evaluation
        ):
            starting = LoopVersion(self.config.model)
            output = Path(temporary) / "arm"
            report, version, _ = run_arm(
                output, starting, self.tasks["evolution"], self.tasks["development"],
                2, True, Budget(168, 100), backend, 1,
            )
            self.assertEqual(report["budget"]["model_calls"], 1)
            self.assertEqual(len(backend.contexts), 1)
            self.assertEqual(report["completed_rounds"], 0)
            self.assertEqual(report["accepted_candidates"], 0)
            self.assertFalse(report["budget"]["model_usage"][0]["available"])
            self.assertEqual(version.snapshot(), starting.snapshot())
            self.assertFalse(list(output.glob("proposal-*.json")))
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertEqual(events[0]["event"], "model_failure")
            self.assertTrue(events[0]["request_attempted"])
            self.assertEqual(events[1]["event"], "model_call_budget_exhausted")

    def test_exhausted_solver_budget_records_unexecuted_cases_as_failures(self):
        tasks = self.tasks["evolution"][:2]
        budget = Budget(1, 100)
        with patch("pde_rsi.agent_runs.run_task", side_effect=BudgetExceeded("No allowance")) as solve:
            results = evaluate_suite(tasks, LoopVersion("mock"), budget)
        self.assertEqual(solve.call_count, 1)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(row["status"] == "budget_exhausted" for row in results))
        self.assertTrue(all(not row["passed"] for row in results))
        self.assertTrue(all(row["absolute_l2"] is None for row in results))


if __name__ == "__main__":
    unittest.main()
