"""Opt-in RSI options (best-of-K, pass-first gate, experiment tool) without model calls or real solves."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pde_rsi import modular
from pde_rsi.code_agents import CodeVersion, proposal_schema, rsi_options, run_arm, run_comparison, system_prompt
from pde_rsi.codegen import classical_program
from pde_rsi.evaluator import EVALUATOR_VERSION, acceptance_gate
from pde_rsi.model_backend import ModelConfig
from pde_rsi.runner import Budget
from pde_rsi.storage import PROJECT_ROOT
from pde_rsi.tasks import load_tasks

from test_code_agents import FakeBackend, fake_evaluation, grid_size, refined
from test_modular import ROUTER, part, row


def gate_row(task_id, error, family="heat", status="ok"):
    return {**row(task_id, family, family, error, status), "evaluator_version": EVALUATOR_VERSION}


class PassFirstGateTests(unittest.TestCase):
    def test_new_pass_is_accepted_despite_a_failing_family_regression(self):
        # a now passes; c fails in both versions and its family mean rises 20%
        old = [gate_row("a", 2.1e-3), gate_row("b", 1e-4), gate_row("c", 5e-1, "caputo")]
        new = [gate_row("a", 1.9e-3), gate_row("b", 1.1e-4), gate_row("c", 6e-1, "caputo")]
        standard = acceptance_gate(old, new)
        self.assertIn("family regression: caputo", standard["reasons"])
        decision = acceptance_gate(old, new, pass_first=True)
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["pass_first"], {"gained": ["a"], "waived": standard["reasons"]})
        self.assertNotIn("pass_first", standard)

    def test_new_pass_is_accepted_despite_small_gain(self):
        old = [gate_row("a", 2.005e-3), gate_row("b", 1e-4)]
        new = [gate_row("a", 1.999e-3), gate_row("b", 1e-4)]
        self.assertEqual(acceptance_gate(old, new)["reasons"], ["insufficient aggregate accuracy gain"])
        self.assertTrue(acceptance_gate(old, new, pass_first=True)["accepted"])

    def test_lost_pass_or_margin_regression_still_rejects(self):
        old = [gate_row("a", 2.1e-3), gate_row("b", 1e-4)]
        lost = [gate_row("a", 1.9e-3), gate_row("b", 2.5e-3)]
        self.assertFalse(acceptance_gate(old, lost, pass_first=True)["accepted"])
        margin = [gate_row("a", 1.9e-3), gate_row("b", 1.5e-4)]  # b passes in both but rises 50%
        decision = acceptance_gate(old, margin, pass_first=True)
        self.assertFalse(decision["accepted"])
        self.assertIn("instance regression: b", decision["reasons"])

    def test_execution_failure_is_never_waived(self):
        old = [gate_row("a", 2.1e-3), gate_row("b", 5e-1)]
        new = [gate_row("a", 1.9e-3), gate_row("b", 1e6, status="execution_failure")]
        self.assertFalse(acceptance_gate(old, new, pass_first=True)["accepted"])

    def test_no_gain_in_passes_keeps_standard_decision(self):
        old = [gate_row("a", 1e-3), gate_row("b", 1e-4)]
        new = [gate_row("a", 0.999e-3), gate_row("b", 1e-4)]
        self.assertEqual(acceptance_gate(old, new, pass_first=True), acceptance_gate(old, new))

    def test_part_gate_inherits_pass_first(self):
        active = {"identify": ROUTER, "heat": part("heat")}
        candidate = {**active, "heat": part("heat", "better")}
        old = [gate_row("h1", 2.005e-3), gate_row("h2", 1e-4)]
        new = [gate_row("h1", 1.999e-3), gate_row("h2", 1e-4)]
        self.assertFalse(modular.part_gate(active, candidate, ["heat"], old, new)[0]["accepted"])
        decision, merged, _ = modular.part_gate(active, candidate, ["heat"], old, new, pass_first=True)
        self.assertTrue(decision["accepted"])
        self.assertEqual(merged["heat"], candidate["heat"])


class OptionTests(unittest.TestCase):
    def test_defaults_add_nothing(self):
        self.assertEqual(rsi_options(), {})
        self.assertEqual(system_prompt(True, False, None), system_prompt(True))
        self.assertNotIn("experiment", proposal_schema(("heat",))["properties"])

    def test_invalid_combinations(self):
        for bad in ({"best_of": 0}, {"gate": "loose"}, {"experiments_per_round": -1},
                    {"best_of": 2, "experiments_per_round": 1}):
            with self.assertRaises(ValueError):
                rsi_options(**bad)

    def test_prompt_and_schema_describe_enabled_options(self):
        prompt = system_prompt(True, False, {"best_of": 3, "gate": "pass-first"})
        self.assertIn("Best-of-3", prompt)
        self.assertIn("pass-first", prompt)
        schema = proposal_schema(("heat",), None, experiment_cases=8)
        self.assertEqual(schema["properties"]["experiment"]["properties"]["cases"]["maxItems"], 8)
        self.assertNotIn("experiment", schema["required"])


class BestOfBackend(FakeBackend):
    """Sample k refines heat and caputo by 16*k points, so the last sample has the lowest training error.
    With broken_samples, those sample numbers return programs that fail to execute."""
    def __init__(self, config, broken_samples=()):
        super().__init__(config)
        self.broken_samples = set(broken_samples)

    def complete(self, messages, schema, directory, timeout):
        note = json.loads(messages[-1]["content"]) if messages[-1]["content"].startswith("{") else {}
        sample = note.get("best_of_sample")
        if sample is None:  # a single-run request
            return super().complete(messages, schema, directory, timeout)
        response = super().complete(messages[:-1], schema, directory, timeout)
        program = self.contexts[-1]["active_program"]
        for update in response["proposal"]["program_updates"]:
            code = refined(program[update["family"]], grid_size(program[update["family"]])+16*sample)
            update["code"] = code+("\n# BROKEN\n" if sample in self.broken_samples else "")
        return response


class ExperimentBackend(FakeBackend):
    """Call 1 requests an experiment on the given cases; later calls submit the same refinement."""
    def __init__(self, config, cases):
        super().__init__(config)
        self.cases = cases

    def complete(self, messages, schema, directory, timeout):
        response = super().complete(messages, schema, directory, timeout)
        if len(self.contexts) == 1:
            response["proposal"]["experiment"] = {"cases": self.cases, "question": "Does refinement help?"}
        return response


class RunArmOptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = {split: load_tasks(PROJECT_ROOT/"benchmarks"/f"{split}.jsonl") for split in ("evolution", "development")}
        cls.config = ModelConfig("http://localhost:9999/v1", "mock-frozen-model")

    def run_arm(self, backend, rounds=1, **options):
        temporary = tempfile.TemporaryDirectory(dir=PROJECT_ROOT/".runtime/tmp")
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)/"arm"
        with patch("pde_rsi.code_agents.evaluate_suite", side_effect=fake_evaluation):
            start = CodeVersion(self.config.model, program=classical_program())
            report, version, _ = run_arm(directory, start, self.tasks["evolution"], self.tasks["development"], rounds,
                                         True, Budget(10000, 100), backend, 30, attempts_per_round=2, **options)
        events = [json.loads(line) for line in (directory/"events.jsonl").open()]
        return report, version, events, directory

    def test_best_of_submits_the_lowest_training_error(self):
        backend = BestOfBackend(self.config)
        start_nx = grid_size(classical_program()["heat"])
        report, version, events, directory = self.run_arm(backend, best_of=3)
        best_of = [e for e in events if e.get("event") == "best_of"]
        self.assertEqual(best_of[0]["selected_sample"], 3)
        self.assertEqual(report["budget"]["model_calls"], 3)
        self.assertEqual(report["accepted_candidates"], 1)
        self.assertEqual(report["rsi_options"], {"best_of": 3})
        self.assertEqual(grid_size(version.program["heat"]), start_nx+48)
        self.assertTrue((directory/"proposal-001-1-s2.json").exists())

    def test_best_of_feeds_back_the_least_broken_sample(self):
        backend = BestOfBackend(self.config, broken_samples=(1, 2))
        report, _, events, _ = self.run_arm(backend, best_of=2)
        self.assertEqual(report["budget"]["model_calls"], 4)  # two attempts of two samples
        selections = [e["selected_sample"] for e in events if e.get("event") == "best_of"]
        self.assertEqual(selections, [None, None])
        later = [json.loads(m["content"]) for m in backend.messages[-1] if m["content"].startswith('{"self_test"')]
        self.assertEqual(len(later[-1]["self_test"]["best_of_samples"]), 2)

    def test_experiment_runs_training_cases_without_using_an_attempt(self):
        case = next(t.task_id for t in self.tasks["evolution"] if t.family == "heat")  # a family the proposal refines
        backend = ExperimentBackend(self.config, [case])
        report, _, events, directory = self.run_arm(backend, experiments_per_round=1)
        experiments = [e for e in events if e.get("event") == "experiment"]
        self.assertEqual(len(experiments), 1)
        self.assertEqual(experiments[0]["cases"], [case])
        self.assertLess(experiments[0]["candidate_macro_relative_l2"], experiments[0]["active_macro_relative_l2"])
        self.assertEqual(report["budget"]["model_calls"], 2)
        self.assertEqual(report["completed_rounds"], 1)  # one validation submission
        gate = next(e for e in events if "gate" in e)
        self.assertEqual(gate["attempt"], 1)
        second = [json.loads(m["content"]) for m in backend.messages[1] if m["content"].startswith('{"experiment_result"')]
        self.assertEqual(second[0]["experiment_result"]["cases"], [case])
        self.assertTrue((directory/f"experiment-001-1x1.json").exists())

    def test_unknown_or_validation_cases_are_refused(self):
        backend = ExperimentBackend(self.config, [self.tasks["development"][0].task_id])
        _, _, events, _ = self.run_arm(backend, experiments_per_round=1)
        experiment = next(e for e in events if e.get("event") == "experiment")
        self.assertIn("Unknown training case ids", experiment["error"])

    def test_experiment_key_is_invalid_when_the_tool_is_off(self):
        backend = ExperimentBackend(self.config, [])
        _, _, events, _ = self.run_arm(backend)
        self.assertEqual(events[0]["event"], "invalid_proposal")

    def test_options_reach_only_the_rsi_arm(self):
        tasks = self.tasks
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/".runtime/tmp") as temporary:
            root = Path(temporary)
            package = root/"package"
            package.mkdir()
            (package/"manifest.json").write_text("{}\n")
            with patch("pde_rsi.code_agents.public_tasks", return_value=tasks), patch(
                    "pde_rsi.code_agents.evaluate_suite", side_effect=fake_evaluation), patch(
                    "pde_rsi.code_agents.sandbox_self_test", return_value={"isolated": True, "sandbox_version": "mock"}):
                metadata = run_comparison(root, package, root/"run", self.config, rounds=1, probe_rounds=0,
                                          backend_factory=BestOfBackend, wall_seconds=100, cpu_seconds=100,
                                          attempts_per_round=1, best_of=2, gate="pass-first")
        self.assertEqual(metadata["configuration"]["rsi_options"], {"best_of": 2, "gate": "pass-first"})
        self.assertEqual(metadata["arms"]["single"]["budget"]["model_calls"], 1)
        self.assertNotIn("rsi_options", metadata["arms"]["single"])
        self.assertEqual(metadata["arms"]["rsi"]["budget"]["model_calls"], 2)


if __name__ == "__main__":
    unittest.main()
