"""Group FPDE examples: forcing consistency, benchmark files, evaluator, solvers, sandbox, and agent view."""

import json
import math
import shutil
import unittest

import numpy as np
from scipy.integrate import quad

from pde_rsi.code_agents import load_public_tasks, solver_contract
from pde_rsi.codegen import agent_problem_record, classical_program, run_program_task
from pde_rsi.equations_group import forcing_group, order
from pde_rsi.evaluator import evaluate
from pde_rsi.policy import SolverSettings
from pde_rsi.reference import output_nodes_group, reference_group
from pde_rsi.runner import Budget, run_task
from pde_rsi.solvers import Solution, solve
from pde_rsi.storage import PROJECT_ROOT
from pde_rsi.tasks import GROUP_FAMILIES, load_group_tasks, make_group_pilot

HAS_BWRAP = shutil.which("bwrap") is not None


def caputo_of_power(power, alpha, t):
    """Caputo derivative of s**power at t by quadrature with the singular kernel; independent of the forcing code."""
    if alpha < 1:
        integrand, exponent, scale = (lambda s: power*s**(power-1)), -alpha, 1/math.gamma(1-alpha)
    else:
        integrand, exponent, scale = (lambda s: power*(power-1)*s**(power-2)), 1-alpha, 1/math.gamma(2-alpha)
    value, _ = quad(integrand, 0, t, weight="alg", wvar=(0, exponent), epsabs=1e-13, epsrel=1e-12)
    return scale*value


class GroupBenchmarkTests(unittest.TestCase):
    def test_forcing_is_consistent_with_the_exact_solution(self):
        for task in make_group_pilot():
            with self.subTest(task.task_id):
                for t in (0.2, 0.55, 1.0):
                    x = 0.7
                    alpha = float(order(task, t, np.array(x)))
                    temporal = caputo_of_power(task.beta, alpha, t)
                    if task.family == "fode":
                        expected = task.amplitude*temporal
                        self.assertAlmostEqual(float(forcing_group(task, 0.0, t)), expected, delta=1e-9)
                        continue
                    k = math.pi*task.mode/task.length
                    # -diffusivity*u_xx of A t^p sin(kx) is A*D*k^2*t^p*sin(kx).
                    expected = task.amplitude*(temporal+task.diffusivity*k*k*t**task.beta)*math.sin(k*x)
                    self.assertAlmostEqual(float(forcing_group(task, np.array([x]), t)[0]), expected, delta=1e-9)

    def test_paper_instances_are_in_the_validation_split(self):
        development = {t.family: t for t in load_group_tasks(PROJECT_ROOT, "development") if t.beta == 3.0}
        self.assertEqual(set(development), set(GROUP_FAMILIES))
        self.assertEqual(development["fode"].orders, (0.5,))
        self.assertEqual(development["fpde_variable_order_xt"].orders, (0.5,))
        self.assertAlmostEqual(float(order(development["fpde_variable_order_xt"], 0.8, np.array(1.1))),
                               0.5*math.sin(1.1)*0.8**2)
        self.assertAlmostEqual(development["fpde_caputo"].length, math.pi)
        self.assertAlmostEqual(order(development["fpde_variable_order"], 1.0), 0.5*math.sin(1.0)**2)
        with self.assertRaises(ValueError):
            load_group_tasks(PROJECT_ROOT, "sealed")

    def test_files_match_generator(self):
        generated = {t.task_id: t.to_dict() for t in make_group_pilot()}
        for split in ("evolution", "development"):
            tasks = load_group_tasks(PROJECT_ROOT, split)
            self.assertEqual(len(tasks), 8)
            for task in tasks:
                self.assertEqual(task.to_dict(), generated[task.task_id])


class GroupEvaluatorAndSolverTests(unittest.TestCase):
    tasks = {t.task_id: t for t in make_group_pilot()}

    def test_exact_solutions_score_zero_error(self):
        for task_id in ("fode-g1-v0", "fpde_caputo-g1-v0", "fpde_variable_order_xt-g1-v0", "fpde_variable_order-g1-v0"):
            task = self.tasks[task_id]
            with self.subTest(task_id):
                x, t = output_nodes_group(task)
                result = evaluate(task, Solution(x, t, reference_group(task, x, t), "exact"))
                self.assertLess(result["relative_l2"], 1e-14)
                self.assertTrue(result["passed"])

    def test_fode_requires_scalar_layout(self):
        task = self.tasks["fode-g1-v0"]
        t = np.linspace(0, 1, 9)
        with self.assertRaises(ValueError):
            evaluate(task, Solution(np.linspace(0, 1, 3), t, np.zeros((9, 3)), "wrong layout"))

    def test_trusted_solvers_converge(self):
        for task_id in ("fode-g1-v0", "fpde_caputo-g1-v0", "fpde_variable_order_xt-g1-v0", "fpde_variable_order-g1-v0"):
            task = self.tasks[task_id]
            with self.subTest(task_id):
                errors = [evaluate(task, solve(task, SolverSettings(nx=nx, nt=nt)))["relative_l2"]
                          for nx, nt in ((32, 48), (64, 96))]
                self.assertLess(errors[1], errors[0]/1.8)


@unittest.skipUnless(HAS_BWRAP, "bubblewrap is required for sandbox tests")
class GroupSandboxTests(unittest.TestCase):
    tasks = {t.task_id: t for t in make_group_pilot()}

    def test_classical_programs_match_trusted_solvers(self):
        programs = classical_program(GROUP_FAMILIES)
        for family in GROUP_FAMILIES:
            task = self.tasks[f"{family}-g1-v0"]
            with self.subTest(family):
                generated = run_program_task(task, programs[family], Budget(1, 60))
                trusted = run_task(task, SolverSettings(), Budget(1, 60))
                self.assertEqual(generated["status"], "ok")
                self.assertAlmostEqual(generated["relative_l2"], trusted["relative_l2"], delta=1e-12)

    def test_fode_tuple_output_and_hidden_power(self):
        task = self.tasks["fode-g1-v0"]
        program = ("import numpy as np\ndef solve(problem):\n    t = np.linspace(0, problem.final_time, 5)\n"
                   "    return t, np.zeros(5)\n")
        result = run_program_task(task, program, Budget(1, 60))
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["relative_l2"], 0.9)
        leak = "def solve(problem):\n    return problem.beta\n"
        self.assertEqual(run_program_task(task, leak, Budget(1, 60))["status"], "execution_failure")
        record = agent_problem_record(task)
        for hidden in ("beta", "amplitude", "profile", "forcing_expression"):
            self.assertNotIn(hidden, record)
        self.assertEqual(record["spatial_dimension"], 0)

    def test_agent_view_and_contract(self):
        record = agent_problem_record(self.tasks["fpde_variable_order-g1-v0"])
        self.assertEqual(record["order_function"], "0.5*sin(t)**2")
        xt = agent_problem_record(self.tasks["fpde_variable_order_xt-g1-v0"])
        self.assertEqual(xt["order_function"], "0.5*sin(x)*t**2")
        self.assertIn("problem.alpha(x, t)", solver_contract(GROUP_FAMILIES))
        self.assertNotIn("velocity", record)
        self.assertIn("problem.alpha(t)", solver_contract(GROUP_FAMILIES))
        tasks = load_public_tasks(PROJECT_ROOT, None, GROUP_FAMILIES)
        self.assertEqual((len(tasks["evolution"]), len(tasks["development"])), (8, 8))


if __name__ == "__main__":
    unittest.main()
