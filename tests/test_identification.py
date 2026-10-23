"""Identification protocol: hidden truth, data, labels only on training feedback, sandbox, and full loop."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pde_rsi.code_agents import CodeVersion, run_comparison
from pde_rsi.codegen import check_program
from pde_rsi.evaluator import EVALUATOR_VERSION
from pde_rsi.identification import (
    LIBRARY, N_POINTS, PROGRAM_KEY, TRUE_CLASS, IdentificationProtocol, make_cases, score_model, true_parameters,
)
from pde_rsi.model_backend import ModelConfig
from pde_rsi.runner import Budget
from pde_rsi.storage import PROJECT_ROOT

HAS_BWRAP = shutil.which("bwrap") is not None
HIDDEN = ("family", "diffusivity", "velocity", "orders", "weights", "profile", "amplitude", "beta", "mode",
          "task_id", "template_group", "forcing_expression")


class CaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {split: make_cases(PROJECT_ROOT, split) for split in ("evolution", "development")}

    def test_two_cases_per_true_family_and_library_covers_truth(self):
        for split, cases in self.cases.items():
            self.assertEqual(len(cases), 16)
            families = [case.family for case in cases]
            self.assertTrue(all(families.count(f) == 2 for f in set(families)))
            self.assertTrue(all(TRUE_CLASS[case.family] in LIBRARY for case in cases))
            self.assertEqual([c.case_id for c in cases], [f"case-{split[0] if split == 'evolution' else 'v'}"
                                                          f"{i:02d}".replace("case-e", "case-t") for i in range(16)])

    def test_public_view_hides_family_and_coefficients(self):
        for case in self.cases["development"]:
            view = case.public_view()
            for name in HIDDEN:
                self.assertNotIn(name, view)
            self.assertNotIn(case.family, json.dumps(view))
            self.assertGreater(view["noise_std"], 0)
            self.assertEqual(view["output_space_points"], 512)

    def test_data_are_deterministic_noisy_samples(self):
        case = self.cases["evolution"][0]
        first, second = case.data(), case.data()
        self.assertTrue(np.array_equal(first["u"], second["u"]))
        self.assertEqual(len(first["x"]), N_POINTS)
        self.assertTrue(np.all((first["t"] > 0) & (first["t"] < case.truth.final_time)))
        self.assertGreater(first["noise_std"], 0)
        self.assertEqual(case.truth.max_seconds, 30.0)

    def test_model_scoring(self):
        case = next(c for c in self.cases["evolution"] if c.family == "convection_diffusion")
        truth = true_parameters(case.truth)
        good = score_model(case, {"family": "convection_diffusion", "parameters": {k: 1.1*v for k, v in truth.items()}})
        self.assertTrue(good["family_correct"])
        self.assertAlmostEqual(good["parameter_relative_errors"]["D"], 0.1, places=12)
        bad = score_model(case, {"family": "heat", "parameters": {"D": 0.1}})
        self.assertFalse(bad["family_correct"])
        self.assertEqual(bad["parameter_relative_errors"], {})
        self.assertFalse(score_model(case, None)["family_correct"])


class ProtocolFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.protocol = IdentificationProtocol()
        self.tasks = self.protocol.load_tasks(PROJECT_ROOT, None, ())

    def fake_row(self, case):
        return {"task_id": case.task_id, "family": case.family, "split": case.split, "case_id": case.case_id,
                "status": "ok", "relative_l2": 1e-3, "passed": True, "true_family": TRUE_CLASS[case.family],
                "family_correct": True, "identified_family": "heat", "identified_parameters": {"D": 0.1}}

    def test_labels_only_on_training_feedback(self):
        train = self.protocol.diagnostics([self.fake_row(self.tasks["evolution"][0])])[0]
        validation = self.protocol.diagnostics([self.fake_row(self.tasks["development"][0])])[0]
        self.assertIn("true_family", train)
        self.assertIn("true_parameters", train)
        for key in ("true_family", "true_parameters", "family_correct", "task_id", "family"):
            self.assertNotIn(key, validation)

    def test_gate_reasons_hide_true_families_and_ids(self):
        case = self.tasks["development"][3]
        reasons = self.protocol.reasons([f"family regression: {case.family}", f"instance regression: {case.task_id}",
                                         "insufficient aggregate accuracy gain"], self.tasks["development"])
        self.assertEqual(reasons, ["family regression: a hidden equation class",
                                   f"instance regression: {case.case_id}", "insufficient aggregate accuracy gain"])
        self.assertIn("variable_order_xt", self.protocol.contract(()))

    def test_starting_programs(self):
        classical = self.protocol.starting_program("classical", ())
        self.assertEqual(list(classical), [PROGRAM_KEY])
        self.assertTrue(check_program(classical[PROGRAM_KEY], PROGRAM_KEY))
        self.assertIn("NotImplementedError", self.protocol.starting_program("scratch", ())[PROGRAM_KEY])


@unittest.skipUnless(HAS_BWRAP, "bubblewrap is required for sandbox tests")
class IdentificationSandboxTests(unittest.TestCase):
    def setUp(self):
        self.protocol = IdentificationProtocol()
        self.tasks = self.protocol.load_tasks(PROJECT_ROOT, None, ())

    def test_classical_baseline_identifies_a_heat_case(self):
        case = next(c for c in self.tasks["development"] if c.family == "heat")
        version = CodeVersion("mock", program=self.protocol.starting_program("classical", ()))
        result = self.protocol.evaluate([case], version, Budget(1, 120, max_cpu_seconds=120))[0]
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["family_correct"])
        self.assertLess(result["parameter_relative_errors"]["D"], 0.05)
        self.assertEqual(result["evaluator_version"], EVALUATOR_VERSION)

    def test_program_cannot_read_the_truth(self):
        case = self.tasks["development"][0]
        leak = ("import numpy as np\ndef solve(problem):\n    D = problem.diffusivity\n"
                "    return {'x': np.linspace(0, problem.length, 9), 't': np.linspace(0, problem.final_time, 5),"
                " 'u': np.zeros((5, 9))}\n")
        result = self.protocol.evaluate([case], CodeVersion("m", program={PROGRAM_KEY: leak}), Budget(1, 60))[0]
        self.assertEqual(result["status"], "execution_failure")
        self.assertIn("diffusivity", result["message"])
        self.assertFalse(result["family_correct"])


class FakeIdentificationBackend:
    def __init__(self, config):
        self.config = config
        self.messages = []

    def complete(self, messages, schema, directory, timeout):
        self.messages.append(messages)
        context = next(json.loads(m["content"]) for m in reversed(messages)
                       if m["role"] == "user" and m["content"].startswith("{") and "active_program" in m["content"])
        code = context["active_program"][PROGRAM_KEY]+"\n# revision\n"
        recursive = "You may also propose" in messages[0]["content"]
        return {"proposal": {"hypothesis": "Fit better.", "memory_update": "lesson",
                             "next_improvement_policy": "Refit." if recursive else None,
                             "program_updates": [{"family": PROGRAM_KEY, "change_summary": "Revise.", "code": code}]},
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "available": True},
                "resolved_model": self.config.model, "wall_seconds": 0.001}


def fake_evaluate(self, tasks, version, budget, solution_directory=None):
    budget.reserve(len(tasks))
    budget.solver_calls += len(tasks)
    error = 1e-3/(1+version.program[PROGRAM_KEY].count("# revision"))
    return [{"task_id": c.task_id, "family": c.family, "split": c.split, "case_id": c.case_id, "status": "ok",
             "relative_l2": error, "final_relative_l2": error, "absolute_l2": error, "final_absolute_l2": error,
             "passed": True, "accuracy_pass": True, "constraints_pass": True, "resources_pass": True,
             "runtime_seconds": 0.01, "wall_seconds": 0.01, "cpu_seconds": 0.01, "peak_memory_mb": 1.0,
             "evaluator_version": EVALUATOR_VERSION, "true_family": TRUE_CLASS[c.family], "family_correct": True,
             "identified_family": TRUE_CLASS[c.family], "identified_parameters": {}, "parameter_relative_errors": {}}
            for c in tasks]


class IdentificationLoopTests(unittest.TestCase):
    def test_full_comparison_with_fake_model(self):
        backends = []

        def factory(config):
            backends.append(FakeIdentificationBackend(config))
            return backends[-1]

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/".runtime/tmp") as temporary, patch.object(
                IdentificationProtocol, "evaluate", fake_evaluate), patch(
                "pde_rsi.code_agents.sandbox_self_test", return_value={"isolated": True, "sandbox_version": "mock"}):
            output = Path(temporary)/"run"
            report = run_comparison(PROJECT_ROOT, None, output, ModelConfig("http://localhost:9999/v1", "mock"),
                                    rounds=2, probe_rounds=1, backend_factory=factory, wall_seconds=100,
                                    cpu_seconds=100, attempts_per_round=1, families=(PROGRAM_KEY,),
                                    protocol=IdentificationProtocol())
            self.assertEqual(report["task_protocol"], "identification")
            self.assertEqual(report["arms"]["rsi"]["accepted_candidates"], 2)
            first_request = backends[0].messages[0]
            problems = json.loads(first_request[1]["content"])["public_problems"]
            self.assertEqual(len(problems), 32)
            text = first_request[1]["content"]
            for case in IdentificationProtocol().load_tasks(PROJECT_ROOT, None, ())["development"]:
                self.assertNotIn(case.task_id, text)
            self.assertIn("variable_order_t", text)


if __name__ == "__main__":
    unittest.main()
