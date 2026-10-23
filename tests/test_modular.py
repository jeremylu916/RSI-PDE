"""Harness v2: modular parts, patch edits, per-part gate, and the sandbox clock."""

import json
from pathlib import Path
import shutil
import tempfile
import types
import unittest

from pde_rsi import modular
from pde_rsi.code_agents import CodeProtocol, CodeVersion, apply_proposal, proposal_schema, run_arm, system_prompt
from pde_rsi.codegen import MAX_CODE_CHARS, ProgramRejected
from pde_rsi.evaluator import EVALUATOR_VERSION
from pde_rsi.model_backend import ModelConfig
from pde_rsi.runner import Budget
from pde_rsi.storage import PROJECT_ROOT
from pde_rsi.tasks import load_tasks

CLASSES = ["heat", "caputo"]
ROUTER = '''"""Router."""


def solve(problem):
    model = {"family": problem.family, "parameters": {}, "method": "numerical"}
    return {**SOLVERS[problem.family](problem, model), "model": model}
'''


def part(name, marker="", helper="value"):
    return (f"def {name}_{helper}():\n    return 1.0  # {marker}\n\n\n"
            f"def solve_{name}(problem, model):\n    return {{\"u\": {name}_{helper}()}}\n\n\n"
            f"SOLVERS[\"{name}\"] = solve_{name}\n")


class PartTests(unittest.TestCase):
    def test_create_patch_and_assemble(self):
        program, changed = modular.apply_updates(modular.starting_program(), [
            {"family": "identify", "change_summary": "router", "code": ROUTER},
            {"family": "heat", "change_summary": "heat", "code": part("heat")}], CLASSES)
        self.assertEqual(changed, ["identify", "heat"])
        program, changed = modular.apply_updates(program, [
            {"family": "heat", "change_summary": "patch", "edits": [{"old": "return 1.0", "new": "return 2.0"}]}],
            CLASSES)
        self.assertIn("return 2.0", program["heat"])
        namespace = {}
        exec(modular.assemble(program), namespace)
        self.assertEqual(namespace["solve"](types.SimpleNamespace(family="heat"))["u"], 2.0)
        self.assertLess(modular.assemble(program).index("part: heat"), modular.assemble(program).index("part: identify"))

    def test_rules(self):
        start = {"identify": ROUTER, "heat": part("heat")}
        cases = [
            ([{"family": "heat", "change_summary": "x", "edits": [{"old": "missing text", "new": ""}]}], "occurs 0 times"),
            ([{"family": "caputo", "change_summary": "x", "code": part("caputo").replace('SOLVERS["caputo"] = solve_caputo', "")}],
             "must register"),
            ([{"family": "caputo", "change_summary": "x", "code": part("caputo", helper="value").replace("caputo_value", "heat_value")}],
             "defined in both"),
            ([{"family": "caputo", "change_summary": "x", "code": part("caputo")+"\n\ndef solve(problem):\n    return 0\n"}],
             "must not define solve"),
            ([{"family": "caputo", "change_summary": "x", "code": "import os\n"+part("caputo")}], "not permitted"),
            ([{"family": "caputo", "change_summary": "x", "code": part("caputo")+"#"*MAX_CODE_CHARS}], "exceeds"),
            ([{"family": "wave", "change_summary": "x", "code": part("wave")}], "unique members"),
            ([{"family": "caputo", "change_summary": "x", "edits": [{"old": "a", "new": "b"}]}], "does not exist yet"),
            ([{"family": "heat", "change_summary": "x", "code": part("heat"), "edits": [{"old": "a", "new": "b"}]}],
             "exactly one of code or edits"),
        ]
        for updates, message in cases:
            with self.assertRaises((ValueError, ProgramRejected)) as context:
                modular.apply_updates(start, updates, CLASSES)
            self.assertIn(message, str(context.exception))

    def test_schema_prompt_and_proposal(self):
        schema = proposal_schema(["identify"], CLASSES)
        item = schema["properties"]["program_updates"]["items"]
        self.assertEqual(item["properties"]["family"]["enum"], ["identify", "common", *CLASSES])
        self.assertEqual(item["required"], ["family", "change_summary"])
        self.assertIn("edits", system_prompt(True, modular=True))
        version = CodeVersion("m", program=modular.starting_program())
        candidate, changed = apply_proposal(version, {
            "hypothesis": "h", "next_improvement_policy": None, "memory_update": "",
            "program_updates": [{"family": "identify", "change_summary": "r", "code": ROUTER}]}, False, CLASSES)
        self.assertEqual(candidate.program["identify"], ROUTER)


def row(task_id, family, routed, error, status="ok"):
    return {"task_id": task_id, "family": family, "split": "development", "status": status,
            "relative_l2": error, "final_relative_l2": error, "passed": error <= 2e-3, "accuracy_pass": error <= 2e-3,
            "constraints_pass": status == "ok", "resources_pass": True, "identified_family": routed,
            "absolute_l2": error/10, "final_absolute_l2": error/10, "runtime_seconds": 0.001, "wall_seconds": 0.001,
            "cpu_seconds": 0.001, "peak_memory_mb": 1.0, "evaluator_version": EVALUATOR_VERSION}


class GateTests(unittest.TestCase):
    def test_accepts_only_the_improving_part(self):
        active = {"identify": ROUTER, "heat": part("heat"), "caputo": part("caputo")}
        candidate = {**active, "heat": part("heat", "better"), "caputo": part("caputo", "worse")}
        old = [row("h1", "heat", "heat", 1e-2), row("c1", "caputo", "caputo", 1e-3), row("p1", "poisson", None, 5e-1)]
        new = [row("h1", "heat", "heat", 1e-4), row("c1", "caputo", "caputo", 5e-3), row("p1", "poisson", None, 5e-1)]
        decision, merged, rows = modular.part_gate(active, candidate, ["heat", "caputo"], old, new)
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["accepted_parts"], ["heat"])
        self.assertEqual(merged["heat"], candidate["heat"])
        self.assertEqual(merged["caputo"], active["caputo"])
        self.assertEqual([r["relative_l2"] for r in rows], [1e-4, 1e-3, 5e-1])
        # an unrelated, still failing case no longer vetoes a class-part improvement
        old[2]["status"] = new[2]["status"] = "execution_failure"
        self.assertTrue(modular.part_gate(active, candidate, ["heat", "caputo"], old, new)[0]["accepted"])

    def test_router_change_is_gated_whole(self):
        active = {"identify": ROUTER, "heat": part("heat")}
        candidate = {**active, "identify": ROUTER+"\n# changed\n"}
        old = [row("h1", "heat", "heat", 1e-2)]
        new = [row("h1", "heat", "heat", 1e-4)]
        decision, merged, _ = modular.part_gate(active, candidate, ["identify"], old, new)
        self.assertEqual((decision["mode"], decision["accepted"], merged), ("whole", True, candidate))


WORKFLOW = modular.INITIAL_WORKFLOW


class FakeModular(CodeProtocol):
    """Errors follow markers in each class part, so the per-part merge is observable without a sandbox."""
    modular = True
    initial_workflow = WORKFLOW

    def classes(self):
        return CLASSES

    def evaluate(self, tasks, version, budget, solution_directory=None):
        rows = []
        for task in tasks:
            code = version.program.get(task.family, "")
            error = 1e-4 if "GOOD" in code else 5e-2 if "BAD" in code else 1e-2
            routed = task.family if task.family in version.program else None
            rows.append({**row(task.task_id, task.family, routed, error), "split": task.split})
        return rows


class ModularBackend:
    """Round 1: router + a good heat part. Round 2: a patch that breaks heat and a new good caputo part."""
    def __init__(self, config):
        self.config, self.calls = config, 0

    def complete(self, messages, schema, directory, timeout):
        self.calls += 1
        updates = ([{"family": "identify", "change_summary": "router", "code": ROUTER},
                    {"family": "heat", "change_summary": "heat", "code": part("heat", "GOOD")}] if self.calls == 1 else
                   [{"family": "heat", "change_summary": "patch", "edits": [{"old": "GOOD", "new": "BAD"}]},
                    {"family": "caputo", "change_summary": "caputo", "code": part("caputo", "GOOD")}])
        return {"proposal": {"hypothesis": "h", "program_updates": updates, "next_improvement_policy": None,
                             "memory_update": ""}, "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "resolved_model": self.config.model, "wall_seconds": 0.0}


class RunArmTests(unittest.TestCase):
    def test_per_part_merge_inside_run_arm(self):
        tasks = {split: [t for t in load_tasks(PROJECT_ROOT/"benchmarks"/f"{split}.jsonl") if t.family in CLASSES]
                 for split in ("evolution", "development")}
        config = ModelConfig("http://localhost:9999/v1", "mock-frozen-model")
        starting = CodeVersion(config.model, program=modular.starting_program(), improvement_workflow=WORKFLOW)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/".runtime/tmp") as temporary:
            report, version, _ = run_arm(Path(temporary)/"arm", starting, tasks["evolution"], tasks["development"], 2,
                                         True, Budget(800, 100), ModularBackend(config), 2, attempts_per_round=1,
                                         protocol=FakeModular())
            events = [json.loads(line) for line in (Path(temporary)/"arm/events.jsonl").open()]
        gates = [e["gate"] for e in events if "gate" in e]
        # Round 1 changes the router, so it is gated whole. Round 2 changes only class parts, so it is gated per part.
        self.assertEqual([g["mode"] for g in gates], ["whole", "per_part"])
        self.assertEqual(gates[1]["accepted_parts"], ["caputo"])
        self.assertEqual(report["accepted_candidates"], 2)
        self.assertIn("GOOD", version.program["heat"])
        self.assertIn("GOOD", version.program["caputo"])


@unittest.skipUnless(shutil.which("bwrap"), "needs bubblewrap")
class ClockTests(unittest.TestCase):
    def test_elapsed_seconds_is_available(self):
        from pde_rsi import pdebench
        case = pdebench.make_cases("evolution")[0]
        code = ("import numpy as np\n\n\ndef solve(problem):\n    e = problem.elapsed_seconds()\n"
                "    assert 0 <= e < problem.max_seconds\n    raise ValueError('elapsed %.3f' % e)\n")
        result = pdebench.run_case(case, code, Budget(1, 100, max_cpu_seconds=100), gpu=False)
        self.assertIn("elapsed", result["message"])


if __name__ == "__main__":
    unittest.main()


class RepairTests(unittest.TestCase):
    def test_one_rule_repair_after_a_rejected_single_answer(self):
        from pde_rsi.code_agents import repair_one_shot, request_one_shot
        tasks = {split: [t for t in load_tasks(PROJECT_ROOT/"benchmarks"/f"{split}.jsonl") if t.family in CLASSES]
                 for split in ("evolution", "development")}
        config = ModelConfig("http://localhost:9999/v1", "mock-frozen-model")
        starting = CodeVersion(config.model, program=modular.starting_program(), improvement_workflow=WORKFLOW)

        class TwoAnswers:
            def __init__(self):
                self.config, self.messages = config, []

            def complete(self, messages, schema, directory, timeout):
                self.messages.append(messages)
                code = part("heat", "GOOD")
                if len(self.messages) == 1:  # first answer breaks the static policy
                    code = "import time\n"+code
                return {"proposal": {"hypothesis": "h", "next_improvement_policy": None, "memory_update": "",
                                     "program_updates": [{"family": "identify", "change_summary": "r", "code": ROUTER},
                                                         {"family": "heat", "change_summary": "h", "code": code}]},
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1}, "resolved_model": config.model,
                        "wall_seconds": 0.0}
        backend = TwoAnswers()
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/".runtime/tmp") as temporary:
            strict, repaired = Path(temporary)/"single", Path(temporary)/"single_repair"
            request_one_shot(strict, starting, tasks["evolution"], tasks["development"], backend, FakeModular(),
                             budget=Budget(200, 100))
            self.assertEqual(json.loads((strict/"one_shot.json").read_text())["status"], "invalid_proposal")
            version = repair_one_shot(strict, repaired, starting, tasks["evolution"], tasks["development"], backend,
                                      FakeModular())
            record = json.loads((repaired/"one_shot.json").read_text())
            self.assertEqual((record["status"], record["repair_of"]["status"]), ("ok", "invalid_proposal"))
            self.assertIn("GOOD", version.program["heat"])
            notice = backend.messages[1][-1]["content"]
            self.assertIn("import 'time' is not permitted", notice)
            self.assertIn("No solver has been run", notice)
            self.assertEqual(json.loads((strict/"one_shot.json").read_text())["status"], "invalid_proposal")
            with self.assertRaises(ValueError):  # a valid strict answer is never repaired
                repair_one_shot(repaired, Path(temporary)/"again", starting, tasks["evolution"],
                                tasks["development"], backend, FakeModular())
