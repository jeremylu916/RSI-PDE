"""2D wave family: benchmark files, manufactured solution, evaluator, solvers, sandbox, and agent protocol."""

import dataclasses
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pde_rsi.code_agents import CodeVersion, apply_proposal, load_public_tasks, run_comparison, solver_contract
from pde_rsi.codegen import agent_problem_record, classical_program, run_program_task
from pde_rsi.equations2d import forcing2d, initial2d
from pde_rsi.evaluator import EVALUATOR_VERSION, evaluate
from pde_rsi.model_backend import ModelConfig
from pde_rsi.policy import SolverSettings
from pde_rsi.reference import output_nodes2d, reference2d
from pde_rsi.runner import Budget, run_task
from pde_rsi.solvers import Solution, solve
from pde_rsi.storage import PROJECT_ROOT
from pde_rsi.tasks import ALL_FAMILIES, FAMILIES, load_wave_tasks, make_pilot, make_wave_pilot

HAS_BWRAP = shutil.which("bwrap") is not None


def exact_solution(task, nx, nt):
    x = np.linspace(0, task.length, nx+1)
    t = np.linspace(0, task.final_time, nt+1)
    return Solution(x, t, reference2d(task, x, x, t), "exact", y=x.copy())


class WaveBenchmarkTests(unittest.TestCase):
    def test_files_match_generator_and_1d_benchmark_is_unchanged(self):
        generated = {t.task_id: t for t in make_wave_pilot()}
        for split in ("evolution", "development"):
            tasks = load_wave_tasks(PROJECT_ROOT, split)
            self.assertEqual(len(tasks), 4)
            for task in tasks:
                self.assertEqual(task.to_dict(), generated[task.task_id].to_dict())
        manifest = json.loads((PROJECT_ROOT/"benchmarks/manifest.json").read_text())
        for split in ("evolution", "development"):
            records = "".join(json.dumps(t.public_record(), sort_keys=True)+"\n" for t in make_pilot() if t.split == split)
            self.assertEqual(hashlib.sha256(records.encode()).hexdigest(), manifest["files"][split]["sha256"])
        with self.assertRaises(ValueError):
            load_wave_tasks(PROJECT_ROOT, "sealed")

    def test_splits_use_disjoint_profiles_and_valid_regularity(self):
        tasks = make_wave_pilot()
        self.assertEqual({t.split: t.profile for t in tasks},
                         {"evolution": "sine", "development": "polynomial", "sealed": "mixed_sine"})
        self.assertTrue(all(t.beta > 1 for t in tasks))
        with self.assertRaises(ValueError):
            dataclasses.replace(tasks[0], beta=0.9).validate()

    def test_forcing_matches_manufactured_solution(self):
        for task in make_wave_pilot():
            with self.subTest(task.task_id):
                x, y, t, h = np.array([0.3*task.length]), np.array([0.6*task.length]), 0.7*task.final_time, 1e-4
                u = lambda xx, yy, tt: reference2d(task, xx, yy, [tt])[0, 0, 0]
                u_tt = (u(x, y, t+h)-2*u(x, y, t)+u(x, y, t-h))/h**2
                laplacian = (u(x+h, y, t)+u(x-h, y, t)+u(x, y+h, t)+u(x, y-h, t)-4*u(x, y, t))/h**2
                expected = u_tt-task.diffusivity*laplacian
                self.assertAlmostEqual(forcing2d(task, x, y, t)[0], expected, delta=1e-4*max(1, abs(expected)))
                self.assertAlmostEqual(initial2d(task, x, y)[0], u(x, y, 0.0), places=12)


class WaveEvaluatorTests(unittest.TestCase):
    task = make_wave_pilot()[4]

    def test_exact_solution_on_evaluator_nodes_has_zero_error(self):
        result = evaluate(self.task, exact_solution(self.task, 127, 99))
        self.assertLess(result["relative_l2"], 1e-14)
        self.assertTrue(result["passed"])
        self.assertEqual((result["nx"], result["ny"], result["nt"]), (127, 127, 99))
        self.assertEqual(result["evaluator_version"], EVALUATOR_VERSION)

    def test_invalid_shapes_coverage_and_boundaries_are_detected(self):
        good = exact_solution(self.task, 32, 16)
        with self.assertRaises(ValueError):
            evaluate(self.task, Solution(good.x, good.t, good.u, "no y"))
        with self.assertRaises(ValueError):
            evaluate(self.task, Solution(good.x, good.t, good.u[:, :, :-1], "bad shape", y=good.y))
        with self.assertRaises(ValueError):
            evaluate(self.task, Solution(good.x, good.t[:-1], good.u[:-1], "short time", y=good.y))
        shifted = good.u.copy()
        shifted[:, 0, :] += 1e-3
        result = evaluate(self.task, Solution(good.x, good.t, shifted, "bad bc", y=good.y))
        self.assertFalse(result["constraints_pass"])
        self.assertFalse(result["passed"])

    def test_fine_grid_node_counts(self):
        from pde_rsi.reference import output_nodes, output_nodes2d, output_nodes_group
        one_d = make_pilot()
        x, t = output_nodes(next(task for task in one_d if task.family == "heat"))
        self.assertEqual((len(x), len(t)), (512, 500))
        self.assertEqual(len(output_nodes(next(task for task in one_d if task.family == "poisson"))[1]), 1)
        x, y, t = output_nodes2d(self.task)
        self.assertEqual((len(x), len(y), len(t)), (128, 128, 100))
        from pde_rsi.tasks import make_group_pilot
        fode = next(task for task in make_group_pilot() if task.family == "fode")
        self.assertEqual([len(a) for a in output_nodes_group(fode)], [1, 500])
        self.assertEqual([len(a) for a in output_nodes_group(fode, "legacy")], [1, 65])
        self.assertEqual(agent_problem_record(self.task)["output_space_points"], 128)

    def test_1d_metrics_are_unchanged_by_the_2d_path(self):
        # Values recorded with manufactured-evaluator-v2-l2 for the classical starting programs.
        run = PROJECT_ROOT/"runs/code-agents-classical-claude-20261014/starting_reference_results.json"
        if not run.exists():
            self.skipTest("historical reference run not present")
        recorded = {row["task_id"]: row for row in json.loads(run.read_text())}
        for task in [t for t in make_pilot() if t.split == "development"][::4]:
            with self.subTest(task.task_id):
                result = evaluate(task, solve(task, SolverSettings()), grid="legacy")
                self.assertAlmostEqual(result["relative_l2"], recorded[task.task_id]["relative_l2"], delta=1e-15)

    def test_trusted_leapfrog_converges_on_smooth_cases(self):
        task = make_wave_pilot()[0]
        errors = [evaluate(task, solve(task, SolverSettings(nx=nx, nt=8)))["relative_l2"] for nx in (16, 32, 64)]
        self.assertLess(errors[1], errors[0]/2.5)
        self.assertLess(errors[2], errors[1])
        self.assertTrue(np.allclose(solve(task, SolverSettings()).u[:, [0, -1], :], 0))


@unittest.skipUnless(HAS_BWRAP, "bubblewrap is required for sandbox tests")
class WaveSandboxTests(unittest.TestCase):
    task = make_wave_pilot()[0]

    def test_classical_program_matches_trusted_solver(self):
        program = classical_program(("wave2d",))["wave2d"]
        generated = run_program_task(self.task, program, Budget(1, 60))
        trusted = run_task(self.task, SolverSettings(), Budget(1, 60))
        self.assertEqual(generated["status"], "ok")
        self.assertAlmostEqual(generated["relative_l2"], trusted["relative_l2"], delta=1e-13)
        self.assertEqual(generated["ny"], 32)

    def test_tuple_output_and_1d_output_errors(self):
        tuple_program = ("import numpy as np\ndef solve(problem):\n"
                         "    x = np.linspace(0, problem.length, 65)\n    t = np.linspace(0, problem.final_time, 17)\n"
                         "    xx, yy = np.meshgrid(x, x)\n    u = np.repeat(problem.initial(xx, yy)[None], 17, axis=0)\n"
                         "    u[:, [0, -1], :] = 0\n    u[:, :, [0, -1]] = 0\n    return x, x, t, u\n")
        result = run_program_task(self.task, tuple_program, Budget(1, 60))
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["constraints_pass"])
        flat = ("import numpy as np\ndef solve(problem):\n    x = np.linspace(0, problem.length, 9)\n"
                "    t = np.linspace(0, problem.final_time, 5)\n    return {'x': x, 't': t, 'u': np.zeros((5, 9))}\n")
        self.assertEqual(run_program_task(self.task, flat, Budget(1, 60))["status"], "execution_failure")

    def test_hidden_profile_parameters_stay_hidden(self):
        record = agent_problem_record(self.task)
        for hidden in ("profile", "amplitude", "mode", "beta", "forcing_expression", "template_group"):
            self.assertNotIn(hidden, record)
        self.assertEqual(record["spatial_dimension"], 2)
        program = "def solve(problem):\n    return problem.beta\n"
        self.assertEqual(run_program_task(self.task, program, Budget(1, 60))["status"], "execution_failure")


class FakeWaveBackend:
    def __init__(self, config):
        self.config = config
        self.contexts = []

    def complete(self, messages, schema, directory, timeout):
        context = next(json.loads(m["content"]) for m in reversed(messages)
                       if m["role"] == "user" and m["content"].startswith("{") and "active_program" in m["content"])
        self.contexts.append(context)
        self.schema = schema
        code = context["active_program"]["wave2d"]
        nx = int(re.search(r"(?m)^NX = (\d+)", code).group(1))
        recursive = "You may also propose" in messages[0]["content"]
        return {"proposal": {"hypothesis": "Refine the wave grid.", "memory_update": "Refinement helped.",
                             "next_improvement_policy": "Refine again." if recursive else None,
                             "program_updates": [{"family": "wave2d", "change_summary": "Refine.",
                                                  "code": re.sub(r"(?m)^NX = \d+", f"NX = {nx+16}", code)}]},
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "available": True},
                "resolved_model": self.config.model, "wall_seconds": 0.001}


def fake_evaluation(tasks, version, budget, solution_directory=None):
    budget.reserve(len(tasks))
    budget.solver_calls += len(tasks)
    rows = []
    for task in tasks:
        error = 0.02/int(re.search(r"(?m)^NX = (\d+)", version.program[task.family]).group(1))
        rows.append({"task_id": task.task_id, "family": task.family, "split": task.split, "status": "ok",
                     "relative_l2": error, "final_relative_l2": error, "absolute_l2": error, "final_absolute_l2": error,
                     "passed": True, "accuracy_pass": True, "constraints_pass": True, "resources_pass": True,
                     "runtime_seconds": 0.001, "wall_seconds": 0.001, "cpu_seconds": 0.001, "peak_memory_mb": 1.0,
                     "nx": 32, "ny": 32, "nt": 48, "evaluator_version": EVALUATOR_VERSION})
    return rows


class WaveAgentProtocolTests(unittest.TestCase):
    def test_public_task_loader_combines_benchmarks(self):
        tasks = load_public_tasks(PROJECT_ROOT, PROJECT_ROOT/"datasets/huggingface/pde-problems", ALL_FAMILIES)
        self.assertEqual(len(tasks["evolution"]), 24+4+8)
        self.assertEqual(len(tasks["development"]), 24+4+8)
        wave = load_public_tasks(PROJECT_ROOT, None, ("wave2d",))
        self.assertEqual({t.family for t in wave["development"]}, {"wave2d"})

    def test_contract_and_proposals_cover_only_requested_families(self):
        self.assertNotIn("Two-dimensional", solver_contract(FAMILIES))
        self.assertIn("u[n, j, i]", solver_contract(("wave2d",)))
        version = CodeVersion("mock", program=classical_program(("wave2d",)))
        code = version.program["wave2d"].replace("NX = 32", "NX = 48")
        update = {"family": "wave2d", "change_summary": "Refine.", "code": code}
        record = {"hypothesis": "h", "program_updates": [update], "next_improvement_policy": None, "memory_update": ""}
        candidate, changed = apply_proposal(version, record, recursive=False)
        self.assertEqual(changed, ["wave2d"])
        heat = dict(update, family="heat")
        with self.assertRaises(ValueError):
            apply_proposal(version, dict(record, program_updates=[heat]), recursive=False)

    def test_wave_only_comparison_runs_with_matched_budgets(self):
        backends = []

        def factory(config):
            backends.append(FakeWaveBackend(config))
            return backends[-1]

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/".runtime/tmp") as temporary:
            output = Path(temporary)/"run"
            with patch("pde_rsi.code_agents.evaluate_suite", side_effect=fake_evaluation), patch(
                    "pde_rsi.code_agents.sandbox_self_test", return_value={"isolated": True, "sandbox_version": "mock"}):
                report = run_comparison(PROJECT_ROOT, PROJECT_ROOT/"datasets/huggingface/pde-problems", output,
                                        ModelConfig("http://localhost:9999/v1", "mock"), rounds=2, probe_rounds=1,
                                        backend_factory=factory, wall_seconds=100, cpu_seconds=100,
                                        attempts_per_round=1, families=("wave2d",))
            self.assertEqual(report["families"], ["wave2d"])
            self.assertIsNone(report["dataset_manifest_sha256"])
            self.assertEqual(report["arms"]["rsi"]["accepted_candidates"], 2)
            self.assertEqual(report["arms"]["rsi"]["budget"]["solver_calls"], 4+2*(4*2+4)+8)
            # The single run is one request: starting evolution + validation evaluations, then the 8 public cases.
            self.assertEqual(report["arms"]["single"]["budget"]["model_calls"], 1)
            self.assertEqual(report["arms"]["single"]["budget"]["solver_calls"], 4+4+8)
            self.assertEqual(backends[0].schema["properties"]["program_updates"]["items"]["properties"]["family"]["enum"],
                             ["wave2d"])
            self.assertEqual(list(backends[0].contexts[0]["active_program"]), ["wave2d"])


if __name__ == "__main__":
    unittest.main()
