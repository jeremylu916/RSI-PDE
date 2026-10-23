"""Code-writing agent tests: static policy, sandbox, proposals, and protocol without model calls."""

import copy
import dataclasses
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

from pde_rsi.code_agents import (
    history_window,
    INITIAL_WORKFLOW, CodeVersion, apply_proposal, build_messages, finish_one_shot, request_one_shot, run_arm,
    run_comparison,
)
from pde_rsi.codegen import (
    ProgramRejected, agent_problem_record, check_program, run_program_task, sandbox_self_test, classical_program,
)
from pde_rsi.evaluator import EVALUATOR_VERSION
from pde_rsi.model_backend import ModelCallError, ModelConfig
from pde_rsi.runner import Budget
from pde_rsi.storage import PROJECT_ROOT
from pde_rsi.tasks import FAMILIES, load_tasks

HAS_BWRAP = shutil.which("bwrap") is not None


def refined(code, nx):
    return re.sub(r"(?m)^NX = \d+", f"NX = {nx}", code)


def grid_size(code):
    return int(re.search(r"(?m)^NX = (\d+)", code).group(1))


class FakeBackend:
    def __init__(self, config, reject_first=False, fail=False, invalid_code=False, broken_attempts=0):
        self.config = config
        self.contexts = []
        self.messages = []
        self.reject_first = reject_first
        self.fail = fail
        self.invalid_code = invalid_code
        self.broken_attempts = broken_attempts
        self.calls_this_round = {}

    def complete(self, messages, schema, directory, timeout):
        self.messages.append(copy.deepcopy(messages))
        context = next(json.loads(m["content"]) for m in reversed(messages)
                       if m["role"] == "user" and m["content"].startswith("{")
                       and "active_program" in json.loads(m["content"]))
        self.contexts.append(context)
        if self.fail:
            raise ModelCallError("Mock provider failure", call_attempted=True, wall_seconds=0.001)
        recursive = "You may also propose" in messages[0]["content"]
        self.current_iteration = context["iteration"]
        program = context["active_program"]
        coarser = self.reject_first and len(self.contexts) == 1
        updates = []
        for family in ("heat", "caputo"):
            nx = grid_size(program[family])
            code = refined(program[family], max(8, nx // 2) if coarser else nx + 16)
            if self.invalid_code:
                code = "import os\n" + code
            iteration = context["iteration"]
            if self.calls_this_round.get(iteration, 0) < self.broken_attempts:
                code += "\n# BROKEN\n"
            updates.append({"family": family, "change_summary": "Change the spatial grid.", "code": code})
        record = {"hypothesis": "Spatial truncation dominates.", "program_updates": updates,
                  "next_improvement_policy": f"Accepted workflow {len(self.contexts)}" if recursive else None,
                  "memory_update": "Spatial refinement reduced the error."}
        self.calls_this_round[context["iteration"]] = self.calls_this_round.get(context["iteration"], 0) + 1
        return {"proposal": record, "usage": {"prompt_tokens": 10, "completion_tokens": 10,
                                              "total_tokens": 20, "available": True},
                "resolved_model": self.config.model, "wall_seconds": 0.001}


def fake_evaluation(tasks, version, budget, solution_directory=None):
    budget.reserve(len(tasks))
    budget.solver_calls += len(tasks)
    rows = []
    for task in tasks:
        error = 0.02 / grid_size(version.program[task.family])
        if "# BROKEN" in version.program[task.family]:
            rows.append({"task_id": task.task_id, "family": task.family, "split": task.split,
                         "status": "execution_failure", "message": "Traceback: broken", "relative_l2": 1e6,
                         "final_relative_l2": 1e6, "absolute_l2": None, "final_absolute_l2": None,
                         "passed": False, "accuracy_pass": False, "constraints_pass": False,
                         "resources_pass": False, "runtime_seconds": 0.001, "wall_seconds": 0.001,
                         "cpu_seconds": 0.001, "peak_memory_mb": 0.0, "evaluator_version": EVALUATOR_VERSION})
            continue
        rows.append({"task_id": task.task_id, "family": task.family, "split": task.split, "status": "ok",
                     "relative_l2": error, "final_relative_l2": error, "absolute_l2": error / 10,
                     "final_absolute_l2": error / 10, "passed": True, "accuracy_pass": True,
                     "constraints_pass": True, "resources_pass": True, "runtime_seconds": 0.001,
                     "wall_seconds": 0.001, "cpu_seconds": 0.001, "peak_memory_mb": 1.0,
                     "nx": 32, "nt": 48, "evaluator_version": EVALUATOR_VERSION})
    return rows


class SlowFakeBackend(FakeBackend):
    def complete(self, messages, schema, directory, timeout):
        time.sleep(0.4)
        return super().complete(messages, schema, directory, timeout)


class TimeBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = {split: load_tasks(PROJECT_ROOT / "benchmarks" / f"{split}.jsonl")
                     for split in ("evolution", "development")}
        cls.config = ModelConfig("http://localhost:9999/v1", "mock-frozen-model")

    def test_history_window_keeps_task_messages_and_last_rounds(self):
        transcript = [("system", "s"), ("user", "task")] + [("user", f"r{i}") for i in range(5)]
        starts = [2, 3, 4, 5, 6]
        self.assertEqual(history_window(transcript, starts, None), transcript)
        self.assertEqual(history_window(transcript, starts, 2), transcript[:2] + transcript[5:])

    def test_concurrent_arms_share_a_deadline_on_distinct_cores(self):
        cores = sorted(os.sched_getaffinity(0))
        if len(cores) < 2:
            self.skipTest("needs two CPU cores")
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".runtime/tmp") as temporary:
            root = Path(temporary)
            package = root / "package"
            package.mkdir()
            (package / "manifest.json").write_text("{}\n")
            output = root / "run"
            with patch("pde_rsi.code_agents.public_tasks", return_value=self.tasks), patch(
                    "pde_rsi.code_agents.evaluate_suite", side_effect=fake_evaluation), patch(
                    "pde_rsi.code_agents.sandbox_self_test", return_value={"isolated": True, "sandbox_version": "mock"}), patch(
                    "pde_rsi.code_agents.run_program_task", side_effect=AssertionError("Unexpected real solve")):
                report = run_comparison(root, package, output, self.config, probe_rounds=0,
                                        backend_factory=SlowFakeBackend, wall_seconds=1900, cpu_seconds=1900,
                                        attempts_per_round=1, time_budget_seconds=2.0, concurrent_arms=True,
                                        cores=cores[:2], history_rounds=2, max_rounds=40)
            self.assertEqual(report["hardware"]["arm_cores"], {"single": cores[0], "rsi": cores[1]})
            self.assertEqual(report["configuration"]["time_budget_seconds_per_arm"], 2.0)
            events = [json.loads(line) for line in (output / "rsi" / "events.jsonl").open()]
            self.assertEqual(events[-1]["event"], "time_budget_reached")
            rounds = report["arms"]["rsi"]["completed_rounds"]
            self.assertTrue(2 <= rounds < 40, rounds)
            self.assertEqual(report["arms"]["single"]["budget"]["model_calls"], 1)
            for arm in ("single", "rsi"):
                self.assertTrue((output / arm / "frozen_results.json").is_file())
            with self.assertRaises(ValueError):
                run_comparison(root, package, root / "bad", self.config, probe_rounds=1, time_budget_seconds=10,
                               wall_seconds=5000, cpu_seconds=5000)


class PricedFakeBackend(FakeBackend):
    """Writes a provider response so the spending cap can price it: 1M output tokens per call."""
    def complete(self, messages, schema, directory, timeout):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "response.json").write_text(json.dumps({"usage": {"inputTokens": 0, "outputTokens": 1_000_000}}))
        return super().complete(messages, schema, directory, timeout)


class BudgetCapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = {split: load_tasks(PROJECT_ROOT / "benchmarks" / f"{split}.jsonl")
                     for split in ("evolution", "development")}

    def run_with(self, **caps):
        config = ModelConfig("http://localhost:9999/v1", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".runtime/tmp") as temporary:
            root = Path(temporary)
            package = root / "package"
            package.mkdir()
            (package / "manifest.json").write_text("{}\n")
            with patch("pde_rsi.code_agents.public_tasks", return_value=self.tasks), patch(
                    "pde_rsi.code_agents.evaluate_suite", side_effect=fake_evaluation), patch(
                    "pde_rsi.code_agents.sandbox_self_test", return_value={"isolated": True, "sandbox_version": "mock"}), patch(
                    "pde_rsi.code_agents.run_program_task", side_effect=AssertionError("Unexpected real solve")):
                report = run_comparison(root, package, root / "run", config, rounds=6, probe_rounds=0,
                                        backend_factory=PricedFakeBackend, wall_seconds=100, cpu_seconds=100,
                                        attempts_per_round=1, **caps)
            events = [json.loads(l) for l in (root / "run" / "rsi" / "events.jsonl").open()]
            return report, events

    def test_validation_submission_cap_stops_rounds(self):
        report, events = self.run_with(max_validation_submissions=2)
        self.assertEqual(report["arms"]["rsi"]["completed_rounds"], 2)
        self.assertEqual(events[-1]["event"], "validation_budget_reached")

    def test_spending_cap_stops_model_requests(self):
        # Sonnet 4.5 output is $15 per 1M tokens, so each fake call costs $15; a $40 cap allows three calls.
        report, events = self.run_with(spend_cap_usd=40.0)
        budget = report["arms"]["rsi"]["budget"]
        self.assertEqual(budget["model_calls"], 3)
        self.assertAlmostEqual(budget["model_cost_usd"], 45.0)
        self.assertEqual(events[-1]["event"], "spending_cap_reached")
        self.assertAlmostEqual(report["arms"]["single"]["budget"]["model_cost_usd"], 15.0)

    def test_single_run_is_one_request_frozen_as_returned(self):
        report, _ = self.run_with()
        single = report["arms"]["single"]
        self.assertEqual((single["one_shot"], single["one_shot_status"], single["completed_rounds"]), (True, "ok", 1))
        self.assertEqual(single["budget"]["model_calls"], 1)
        self.assertEqual(single["changed_from_start"], ["caputo", "heat"])
        self.assertEqual(report["arms"]["rsi"]["completed_rounds"], 6)

    def test_single_run_does_not_retry_an_invalid_answer(self):
        config = ModelConfig("http://localhost:9999/v1", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        backend = FakeBackend(config, invalid_code=True)
        starting = CodeVersion(config.model)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".runtime/tmp") as temporary, patch(
                "pde_rsi.code_agents.evaluate_suite", side_effect=fake_evaluation):
            output = Path(temporary) / "single"
            request_one_shot(output, starting, self.tasks["evolution"], self.tasks["development"], backend,
                             budget=Budget(200, 100))
            report, version, _ = finish_one_shot(output, starting, self.tasks["evolution"],
                                                 self.tasks["development"], Budget(48, 100))
        self.assertEqual(len(backend.contexts), 1)
        self.assertNotIn("diagnostics", backend.messages[0][0]["content"].split("single run")[1])
        self.assertTrue(backend.contexts[0]["single_run"])
        self.assertEqual(report["one_shot_status"], "invalid_proposal")
        self.assertEqual(version.program, starting.program)

    def test_single_run_resends_only_provider_failures(self):
        config = ModelConfig("http://localhost:9999/v1", "mock-frozen-model")

        class Flaky(FakeBackend):
            def complete(self, messages, schema, directory, timeout):
                if not self.contexts:
                    self.contexts.append(None)
                    raise ModelCallError("Model call failed: http_error (HTTP 503)", call_attempted=True)
                return super().complete(messages, schema, directory, timeout)
        backend = Flaky(config)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".runtime/tmp") as temporary, patch(
                "pde_rsi.code_agents.evaluate_suite", side_effect=fake_evaluation):
            output = Path(temporary) / "single"
            request_one_shot(output, CodeVersion(config.model), self.tasks["evolution"], self.tasks["development"],
                             backend, budget=Budget(200, 100))
            record = json.loads((output / "one_shot.json").read_text())
        self.assertEqual((record["status"], len(record["requests"])), ("ok", 2))


class StaticPolicyTests(unittest.TestCase):
    def test_classical_programs_pass_static_policy(self):
        program = classical_program()
        self.assertEqual(set(program), set(FAMILIES))
        for family, code in program.items():
            self.assertTrue(check_program(code, family))

    def test_policy_rejects_escape_routes_and_contract_violations(self):
        body = "def solve(problem):\n    return None\n"
        rejected = {
            "os import": "import os\n" + body,
            "subprocess import": "from subprocess import run\n" + body,
            "relative import": "from . import x\n" + body,
            "open": body + "open('/etc/passwd')\n",
            "eval": body + "eval('1')\n",
            "dunder name": body + "__import__('os')\n",
            "dunder attribute": body + "x = (1).__class__\n",
            "private attribute": "import numpy as np\n" + body + "np._core\n",
            "frame attribute": body + "def g():\n    yield 1\nh = g().gi_frame\n",
            "numpy file input": "import numpy as np\n" + body + "np.load('x')\n",
            "star import": "from numpy import *\n" + body,
            "missing solve": "def other(problem):\n    return None\n",
            "syntax": "def solve(problem:\n",
            "getattr": body + "getattr(1, 'real')\n",
            "too long": body + "#" + "x" * 30001 + "\n",
        }
        for name, code in rejected.items():
            with self.subTest(name), self.assertRaises(ProgramRejected):
                check_program(code)

    def test_model_view_hides_manufactured_solution_fields(self):
        task = load_tasks(PROJECT_ROOT / "benchmarks/development.jsonl")[0]
        record = agent_problem_record(task)
        for hidden in ("profile", "amplitude", "mode", "beta", "forcing_expression",
                       "initial_expression", "template_group"):
            self.assertNotIn(hidden, record)
        self.assertEqual(record["task_id"], task.task_id)
        self.assertIn("equation", record)


@unittest.skipUnless(HAS_BWRAP, "bubblewrap is required for sandbox tests")
class SandboxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = load_tasks(PROJECT_ROOT / "benchmarks/development.jsonl")

    def task(self, family):
        return next(task for task in self.tasks if task.family == family)

    def test_sandbox_hides_project_files_and_network(self):
        report = sandbox_self_test()
        self.assertTrue(report["isolated"])
        self.assertFalse(report["network"])

    def test_classical_programs_execute_and_match_trusted_solvers(self):
        from pde_rsi.policy import SolverSettings
        from pde_rsi.runner import run_task
        program = classical_program()
        for family in FAMILIES:
            task = self.task(family)
            with self.subTest(family):
                generated = run_program_task(task, program[family], Budget(1, 60))
                trusted = run_task(task, SolverSettings(), Budget(1, 60))
                self.assertEqual(generated["status"], "ok")
                self.assertAlmostEqual(generated["relative_l2"], trusted["relative_l2"], delta=1e-12)
                self.assertTrue(generated["constraints_pass"])

    def test_runtime_rejects_imports_files_and_reports_tracebacks(self):
        task = self.task("heat")
        cases = {
            "import os": "def solve(problem):\n    import os\n    return os.listdir('/')\n",
            "open": "def solve(problem):\n    return open('/etc/hostname').read()\n",
            "error line": "def solve(problem):\n    x = 1\n    raise ValueError('boom at solver')\n",
            "hidden fields": "def solve(problem):\n    return problem.beta\n",
        }
        for name, code in cases.items():
            with self.subTest(name):
                result = run_program_task(task, code, Budget(1, 60))
                self.assertEqual(result["status"], "execution_failure")
                self.assertFalse(result["passed"])
        result = run_program_task(task, cases["error line"], Budget(1, 60))
        self.assertIn("line 3", result["message"])
        self.assertIn("boom at solver", result["message"])

    def test_sandbox_cpu_is_metered_into_the_budget(self):
        task = self.task("poisson")
        busy = ("import numpy as np\ndef solve(problem):\n    total = 0.0\n"
                "    for i in range(3000000):\n        total += i\n"
                "    x = np.linspace(0, problem.length, 65)\n    u = np.zeros((1, 65))\n"
                "    return x, np.array([0.0]), u\n")
        budget = Budget(1, 60)
        result = run_program_task(task, busy, budget)
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["sandbox_cpu_seconds"], 0.1)
        self.assertGreaterEqual(budget.report()["sandbox_cpu_seconds"], result["sandbox_cpu_seconds"])
        self.assertGreater(budget.report()["total_cpu_seconds"], 0.1)

    def test_invalid_output_and_timeout_are_charged_failures(self):
        task = self.task("heat")
        bad_shape = ("import numpy as np\ndef solve(problem):\n"
                     "    x = np.linspace(0, problem.length, 5)\n    t = np.linspace(0, problem.final_time, 4)\n"
                     "    return x, t, np.zeros((3, 5))\n")
        budget = Budget(2, 60)
        self.assertEqual(run_program_task(task, bad_shape, budget)["status"], "invalid_output")
        overflow = ("import numpy as np\ndef solve(problem):\n"
                    "    x = np.linspace(0, problem.length, 9)\n    t = np.linspace(0, problem.final_time, 5)\n"
                    "    u = np.full((5, 9), 1e300)\n    u[0] = problem.initial(x)\n    u[:, [0, -1]] = 0\n"
                    "    return x, t, u\n")
        result = run_program_task(task, overflow, Budget(1, 60))
        self.assertEqual(result["status"], "invalid_output")
        self.assertIn("overflow", result["message"])
        slow = "def solve(problem):\n    while True:\n        pass\n"
        result = run_program_task(dataclasses.replace(task, max_seconds=2.0), slow, budget)
        self.assertIn(result["status"], {"timeout", "execution_failure"})
        self.assertEqual(budget.solver_calls, 2)


class ProposalTests(unittest.TestCase):
    def proposal(self, policy=None, families=("heat",), nx=48):
        program = classical_program()
        return {"hypothesis": "Refine.", "next_improvement_policy": policy, "memory_update": "Lesson.",
                "program_updates": [{"family": f, "change_summary": "Refine.", "code": refined(program[f], nx)}
                                    for f in families]}

    def test_partial_update_keeps_other_programs_and_fixed_controller(self):
        initial = CodeVersion("mock", memory=[{"lesson": "existing"}])
        with self.assertRaises(ValueError):
            apply_proposal(initial, self.proposal("New workflow"), recursive=False)
        candidate, changed = apply_proposal(initial, self.proposal(), recursive=False)
        self.assertEqual(changed, ["heat"])
        self.assertEqual(grid_size(candidate.program["heat"]), 48)
        for family in FAMILIES:
            if family != "heat":
                self.assertEqual(candidate.program[family], initial.program[family])
        self.assertEqual((candidate.memory, candidate.generation, candidate.improvement_workflow),
                         (initial.memory, initial.generation, initial.improvement_workflow))

    def test_recursive_candidate_inherits_workflow_memory_and_lineage(self):
        initial = CodeVersion("mock")
        first, _ = apply_proposal(initial, self.proposal("Refine the weakest family."), recursive=True)
        second, _ = apply_proposal(first, self.proposal(nx=64), recursive=True)
        self.assertEqual((first.parent_id, second.parent_id), (initial.version_id, first.version_id))
        self.assertEqual(second.generation, 2)
        self.assertEqual(second.improvement_workflow, "Refine the weakest family.")
        self.assertEqual(len(second.memory), 2)
        self.assertEqual(second.memory[0]["changed_families"], ["heat"])
        self.assertEqual(initial.improvement_workflow, INITIAL_WORKFLOW)

    def test_malformed_and_noop_proposals_are_rejected(self):
        initial = CodeVersion("mock")
        malformed = []
        noop = self.proposal()
        noop["program_updates"][0]["code"] = initial.program["heat"]
        malformed.append(noop)
        duplicate = self.proposal(families=("heat", "heat"))
        malformed.append(duplicate)
        unsafe = self.proposal()
        unsafe["program_updates"][0]["code"] = "import os\n" + unsafe["program_updates"][0]["code"]
        malformed.append(unsafe)
        extra = self.proposal()
        extra["solver"] = []
        malformed.append(extra)
        empty = self.proposal(families=())
        malformed.append(empty)
        for index, record in enumerate(malformed):
            with self.subTest(index=index), self.assertRaises(ValueError):
                apply_proposal(initial, record, recursive=True)

    def test_transcript_compacts_all_but_latest_programs(self):
        program = classical_program()
        proposal = self.proposal()
        rows = [{"task_id": "heat-g1-v0", "status": "execution_failure", "relative_l2": 1e6, "passed": False,
                 "message": "x" * 900, "runtime_seconds": 1.0}]
        context = {"active_program": program, "evolution_diagnostics": rows, "validation_diagnostics": rows}
        transcript = [("system", "s"), ("user", {"public_problems": []}),
                      ("user", context), ("assistant", proposal),
                      ("user", "Proposal rejected before execution: x"),
                      ("user", context), ("assistant", proposal)]
        messages = build_messages(transcript)
        first, latest = json.loads(messages[2]["content"]), json.loads(messages[5]["content"])
        self.assertIn("omitted", first["active_program"])
        self.assertEqual(len(first["evolution_diagnostics"][0]["message"]), 300)
        self.assertNotIn("runtime_seconds", first["validation_diagnostics"][0])
        self.assertEqual(latest["evolution_diagnostics"], rows)
        self.assertEqual(latest["active_program"], program)
        self.assertIn("omitted", json.loads(messages[3]["content"])["program_updates"][0]["code"])
        self.assertEqual(json.loads(messages[6]["content"]), proposal)
        self.assertEqual(messages[4]["content"], "Proposal rejected before execution: x")


class CodeProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = {split: load_tasks(PROJECT_ROOT / "benchmarks" / f"{split}.jsonl")
                     for split in ("evolution", "development")}
        cls.config = ModelConfig("http://localhost:9999/v1", "mock-frozen-model")

    def test_comparison_matches_allowances_and_inherits_policy(self):
        backends = []

        def factory(config):
            backends.append(FakeBackend(config))
            return backends[-1]

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".runtime/tmp") as temporary:
            root = Path(temporary)
            package = root / "package"
            package.mkdir()
            (package / "manifest.json").write_text("{}\n")
            output = root / "run"
            with patch("pde_rsi.code_agents.public_tasks", return_value=self.tasks), patch(
                    "pde_rsi.code_agents.evaluate_suite", side_effect=fake_evaluation), patch(
                    "pde_rsi.code_agents.sandbox_self_test", return_value={"isolated": True, "sandbox_version": "mock"}), patch(
                    "pde_rsi.code_agents.run_program_task", side_effect=AssertionError("Unexpected real solve")):
                report = run_comparison(root, package, output, self.config, rounds=2, probe_rounds=1,
                                        backend_factory=factory, wall_seconds=100, cpu_seconds=100,
                                        attempts_per_round=2)
            self.assertTrue(report["all_requested_rounds_complete"])
            self.assertFalse(report["sealed_cases_used"])
            self.assertEqual(report["single_arm"], "one_shot")
            # Allowance 24 + 2*(24*3 + 24) + 48; one self-test per round is used.
            self.assertEqual(report["arms"]["rsi"]["budget"]["max_solver_calls"], 264)
            self.assertEqual(report["arms"]["rsi"]["budget"]["solver_calls"], 72 + 72 * 2)
            self.assertEqual(report["arms"]["rsi"]["budget"]["max_model_calls"], 4)
            self.assertEqual(report["arms"]["rsi"]["accepted_candidates"], 2)
            # The single run: starting evolution + validation evaluations, one request, then the 48 public cases.
            self.assertEqual(report["arms"]["single"]["budget"]["solver_calls"], 24 + 24 + 48)
            self.assertEqual(report["arms"]["single"]["budget"]["model_calls"], 1)
            for arm in ("single", "rsi"):
                self.assertEqual(report["arms"][arm]["changed_from_start"], ["caputo", "heat"])
                self.assertTrue((output / arm / "programs/frozen/heat.py").is_file())
            self.assertEqual(json.loads((output / "single/initial_agent.json").read_text()),
                             json.loads((output / "rsi/initial_agent.json").read_text()))
            single = json.loads((output / "single/frozen_agent.json").read_text())["agent"]
            self.assertEqual((single["improvement_workflow"], single["memory"], single["generation"]),
                             (INITIAL_WORKFLOW, [], 0))
            self.assertIn("single run", backends[0].messages[0][0]["content"])
            self.assertEqual(backends[1].contexts[1]["active_improvement_policy"], "Accepted workflow 1")
            self.assertEqual(len(backends[1].contexts[1]["active_agent"]["agent"]["memory"]), 1)
            self.assertEqual(grid_size(backends[1].contexts[1]["active_program"]["heat"]), 48)
            second_request = backends[1].messages[1]
            contexts = [json.loads(m["content"]) for m in second_request
                        if m["role"] == "user" and "active_program" in m["content"]]
            self.assertIn("omitted", contexts[0]["active_program"])
            self.assertIsInstance(contexts[-1]["active_program"], dict)
            for problem in json.loads(second_request[1]["content"])["public_problems"]:
                self.assertNotIn("beta", problem)
            evolved = json.loads((output / "continuation/evolved_policy/initial_agent.json").read_text())["agent"]
            initial = json.loads((output / "continuation/initial_policy/initial_agent.json").read_text())["agent"]
            self.assertEqual(initial["improvement_workflow"], INITIAL_WORKFLOW)
            self.assertNotEqual(evolved["improvement_workflow"], INITIAL_WORKFLOW)
            evolved.pop("improvement_workflow")
            initial.pop("improvement_workflow")
            self.assertEqual(initial, evolved)
            self.assertTrue(report["continuation_matched_actual_calls"])
            self.assertTrue((output / "summary.md").is_file())

    def test_rejected_candidate_keeps_active_program_and_workflow(self):
        backend = FakeBackend(self.config, reject_first=True)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".runtime/tmp") as temporary, patch(
                "pde_rsi.code_agents.evaluate_suite", side_effect=fake_evaluation):
            output = Path(temporary) / "arm"
            report, version, _ = run_arm(output, CodeVersion(self.config.model), self.tasks["evolution"],
                                         self.tasks["development"], 2, True, Budget(264, 100), backend, 2,
                                         attempts_per_round=1)
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            events = [event for event in events if "gate" in event]
            self.assertFalse(events[0]["gate"]["accepted"])
            self.assertEqual(backend.contexts[1]["active_improvement_policy"], INITIAL_WORKFLOW)
            self.assertEqual(grid_size(backend.contexts[1]["active_program"]["heat"]), 32)
            self.assertEqual(report["accepted_candidates"], 1)
            self.assertEqual(version.generation, 1)
            self.assertTrue((output / "programs/candidate-001-1/heat.py").is_file())

    def test_statically_rejected_code_is_not_executed(self):
        backend = FakeBackend(self.config, invalid_code=True)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".runtime/tmp") as temporary, patch(
                "pde_rsi.code_agents.evaluate_suite", side_effect=fake_evaluation) as evaluation:
            output = Path(temporary) / "arm"
            report, version, _ = run_arm(output, CodeVersion(self.config.model), self.tasks["evolution"],
                                         self.tasks["development"], 1, True, Budget(168, 100), backend, 1,
                                         attempts_per_round=1)
            phases = [call.args[0][0].split for call in evaluation.call_args_list]
            self.assertEqual(len(phases), 3)  # initial validation, evolution, frozen evaluation
            self.assertEqual(report["completed_rounds"], 0)
            self.assertEqual(version.program, classical_program())
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertEqual(events[0]["event"], "invalid_proposal")
            self.assertIn("import 'os'", events[0]["message"])

    def test_failed_model_call_is_charged(self):
        backend = FakeBackend(self.config, fail=True)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".runtime/tmp") as temporary, patch(
                "pde_rsi.code_agents.evaluate_suite", side_effect=fake_evaluation):
            report, version, _ = run_arm(Path(temporary) / "arm", CodeVersion(self.config.model),
                                         self.tasks["evolution"], self.tasks["development"], 2, True,
                                         Budget(312, 100), backend, 1)
            self.assertEqual(report["budget"]["model_calls"], 1)
            self.assertEqual(report["completed_rounds"], 0)
            self.assertEqual(version.program, classical_program())

    def test_broken_self_test_is_revised_within_the_round(self):
        backend = FakeBackend(self.config, broken_attempts=1)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".runtime/tmp") as temporary, patch(
                "pde_rsi.code_agents.evaluate_suite", side_effect=fake_evaluation):
            output = Path(temporary) / "arm"
            report, version, _ = run_arm(output, CodeVersion(self.config.model), self.tasks["evolution"],
                                         self.tasks["development"], 1, True, Budget(192, 100), backend, 3,
                                         attempts_per_round=3)
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertEqual([e.get("event", "gate") for e in events], ["self_test", "self_test", "gate"])
            self.assertFalse(events[0]["passed"])
            self.assertEqual(events[2]["attempt"], 2)
            self.assertTrue(events[2]["gate"]["accepted"])
            self.assertEqual(report["budget"]["model_calls"], 2)
            self.assertNotIn("# BROKEN", version.program["heat"])
            revision_request = backend.messages[1][-1]
            self.assertIn("self_test", revision_request["content"])
            self.assertFalse((output / "candidate-validation-001.json").read_text().count("execution_failure"))

    def test_candidate_broken_in_every_attempt_is_never_validated(self):
        backend = FakeBackend(self.config, broken_attempts=3)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".runtime/tmp") as temporary, patch(
                "pde_rsi.code_agents.evaluate_suite", side_effect=fake_evaluation) as evaluation:
            output = Path(temporary) / "arm"
            report, version, _ = run_arm(output, CodeVersion(self.config.model), self.tasks["evolution"],
                                         self.tasks["development"], 1, True, Budget(192, 100), backend, 3,
                                         attempts_per_round=3)
            splits = [call.args[0][0].split for call in evaluation.call_args_list]
            # initial validation, active evolution, three self-tests, frozen evaluation
            self.assertEqual(splits, ["development", "evolution", "evolution", "evolution", "evolution", "evolution"])
            self.assertEqual(report["completed_rounds"], 0)
            self.assertEqual(report["budget"]["model_calls"], 3)
            self.assertEqual(version.program, classical_program())
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertEqual(events[-1]["event"], "no_submission")


if __name__ == "__main__":
    unittest.main()
