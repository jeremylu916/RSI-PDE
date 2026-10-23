from dataclasses import replace
import math
import unittest

import numpy as np
from scipy.integrate import quad

from pde_rsi.equations import forcing
from pde_rsi.evaluator import evaluate
from pde_rsi.policy import SolverSettings
from pde_rsi.reference import output_nodes, reference
from pde_rsi.solvers import Solution, l1_weights, solve
from pde_rsi.tasks import FAMILIES, make_pilot


class NumericalTests(unittest.TestCase):
    def task(self, family, split="evolution"):
        return next(t for t in make_pilot() if t.family == family and t.split == split)

    def test_local_forcing_matches_independently_differentiated_reference(self):
        for family in ("poisson", "heat", "convection_diffusion", "burgers"):
            for split in ("evolution", "development", "sealed"):
                with self.subTest(family=family, split=split):
                    task = self.task(family, split)
                    x, t, h = np.array([0.23*task.length, 0.68*task.length]), 0.17, 2e-5
                    u = reference(task, x, [t])[0]
                    ux = (reference(task, x+h, [t])[0]-reference(task, x-h, [t])[0])/(2*h)
                    uxx = (reference(task, x+h, [t])[0]-2*u+reference(task, x-h, [t])[0])/h**2
                    ut = (reference(task, x, [t+h])[0]-reference(task, x, [t-h])[0])/(2*h)
                    residual_forcing = -task.diffusivity*uxx
                    if family != "poisson":
                        residual_forcing += ut
                    if family == "convection_diffusion":
                        residual_forcing += task.velocity*ux
                    if family == "burgers":
                        residual_forcing += u*ux
                    np.testing.assert_allclose(forcing(task, x, t), residual_forcing, rtol=2e-6, atol=2e-7)

    def test_fractional_forcing_matches_integral_definition(self):
        # Quadrature of the defining Caputo integral checks the Gamma-form forcing
        # independently, including a weak initial derivative on the dev template.
        for family in ("caputo", "multi_term"):
            task = self.task(family, "development")
            x, t = np.array([0.27]), 0.21
            h = 1e-4
            spatial = reference(task, x, [0])[0]
            u = reference(task, x, [t])[0]
            uxx = (reference(task, x+h, [t])[0]-2*u+reference(task, x-h, [t])[0])/h**2
            temporal = 0.0
            for alpha, weight in zip(task.orders, task.weights):
                # qawse handles both endpoint singularities independently of L1.
                integral, _ = quad(lambda s: task.beta, 0, t, weight="alg",
                                   wvar=(task.beta-1, -alpha), epsabs=1e-10)
                temporal += weight*integral/math.gamma(1-alpha)
            expected = temporal*spatial-task.diffusivity*uxx
            np.testing.assert_allclose(forcing(task, x, t), expected, rtol=1e-6, atol=1e-7)

    def test_l1_is_exact_on_linear_function_on_graded_mesh(self):
        for alpha in (0.2, 0.6, 0.9):
            times = np.linspace(0, 0.5, 21)**2
            for n in (1, 2, 9, 20):
                numerical = l1_weights(times, n, alpha) @ np.diff(times[:n+1])
                expected = times[n]**(1-alpha)/math.gamma(2-alpha)
                self.assertAlmostEqual(numerical, expected, places=12)

    def test_refinement_reduces_error_in_all_families(self):
        for family in FAMILIES:
            task = self.task(family)
            with self.subTest(family=family):
                coarse = evaluate(task, solve(task, SolverSettings(nx=32, nt=48)))
                fine = evaluate(task, solve(task, SolverSettings(nx=64, nt=96)))
                self.assertLess(fine["relative_l2"], coarse["relative_l2"])
                self.assertTrue(fine["constraints_pass"])

    def test_fractional_graded_grid_improves_initial_singularity(self):
        task = replace(self.task("caputo", "development"), beta=0.4)
        uniform = evaluate(task, solve(task, SolverSettings(nx=64, nt=64, grading=1)))
        graded = evaluate(task, solve(task, SolverSettings(nx=64, nt=64, grading=2)))
        self.assertLess(graded["relative_l2"], uniform["relative_l2"])

    def test_invalid_solution_cannot_pass(self):
        task = self.task("heat")
        solution = solve(task, SolverSettings())
        solution.u[1, 2] = np.nan
        with self.assertRaises(ValueError):
            evaluate(task, solution)

    def test_wrong_boundary_and_initial_conditions_are_detected(self):
        task = self.task("heat")
        solution = solve(task, SolverSettings())
        solution.u[:, 0] = 0.1
        solution.u[0, 3] += 0.1
        metrics = evaluate(task, solution)
        self.assertFalse(metrics["constraints_pass"])
        self.assertFalse(metrics["passed"])

    def test_exact_reference_passes_with_small_residual(self):
        task = self.task("caputo")
        x, t = output_nodes(task)  # exact values on the evaluator's own nodes
        solution = Solution(x, t, reference(task, x, t), "reference validation only")
        metrics = evaluate(task, solution)
        self.assertLess(metrics["relative_l2"], 1e-12)
        self.assertLess(metrics["physical_residual_relative"], 2e-3)


if __name__ == "__main__":
    unittest.main()
