"""Inventory suite: continuous input definitions, hidden labels, evaluator, and the sandbox contract per layout."""

import math
import shutil
import unittest

import numpy as np

from pde_rsi import equations_inventory as eq
from pde_rsi.codegen import check_program
from pde_rsi.inventory import FAMILIES, SPLITS, InventoryProtocol, make_cases, reference_path, run_case
from pde_rsi.runner import Budget

# Contract-shaped and inaccurate: right shapes, exact IC/BC, zero elsewhere; calls every black box once.
SHAPE_PROGRAM = '''
import numpy as np

def solve(problem):
    model = {"family": "heat", "parameters": {}, "method": "numerical"}
    d, layout = problem.domain, problem.layout
    if layout == "4d_points":
        p = problem.points
        problem.coefficient(p[:5, 0], p[:5, 1], p[:5, 2], p[:5, 3])
        problem.forcing(p[:5, 0], p[:5, 1], p[:5, 2], p[:5, 3])
        return {"u": np.zeros(len(p)), "model": model}
    x = np.linspace(d["x"][0], d["x"][1], 33)
    if layout == "2d_steady":
        X, Y = np.meshgrid(x, x)
        problem.forcing(X, Y)
        if problem.has_coefficient_field:
            problem.coefficient(X, Y)
        return {"x": x, "y": x, "u": np.zeros((33, 33)), "model": model}
    t = np.linspace(0, d["t"][1], 5)
    if layout == "2d_periodic_time":
        X, Y = np.meshgrid(x, x)
        u = np.zeros((5, 33, 33))
        u[:] = problem.initial(X, Y)
        problem.forcing(X, Y, 0.5)
        if problem.has_coefficient_field:
            problem.coefficient(X, Y)
        return {"x": x, "y": x, "t": t, "u": u, "model": model}
    u = np.zeros((5, 33))
    u[:] = problem.initial(x)
    if layout == "1d_dirichlet":
        for n in range(5):
            u[n, 0], u[n, -1] = problem.boundary(t[n])
    else:
        u[:, -1] = u[:, 0]
    problem.forcing(x, 0.5)
    return {"x": x, "t": t, "u": u, "model": model}
'''


class InputDefinitionTests(unittest.TestCase):
    def test_periodic_interpolation_reproduces_grid_and_is_consistent(self):
        rng = np.random.default_rng(0)
        values = rng.standard_normal((16, 16))
        grid = np.arange(16)/16*2*math.pi
        X, Y = np.meshgrid(grid, grid)
        self.assertTrue(np.allclose(eq.periodic_2d(values, 2*math.pi, X, Y), values, atol=1e-12))
        px, py = rng.uniform(0, 2*math.pi, 7), rng.uniform(0, 2*math.pi, 7)
        sep = eq.periodic_2d(values, 2*math.pi, *np.meshgrid(px, py))
        point = eq.periodic_2d(values, 2*math.pi, *[a.copy() for a in np.meshgrid(px, py)][::-1][::-1])
        self.assertTrue(np.allclose(sep, point))
        one = rng.standard_normal(12)
        self.assertTrue(np.allclose(eq.periodic_1d(one, 1.0, np.arange(12)/12), one, atol=1e-12))

    def test_node_spline_and_cell_constant(self):
        rng = np.random.default_rng(1)
        nodes = rng.standard_normal((9, 9))
        g = np.linspace(0, 1, 9)
        self.assertTrue(np.allclose(eq.node_spline(nodes, *np.meshgrid(g, g)), nodes, atol=1e-12))
        cells = rng.standard_normal((4, 4))
        centres = (np.arange(4)+0.5)/4
        self.assertTrue(np.array_equal(eq.cell_constant(cells, *np.meshgrid(centres, centres)), cells))

    def test_random_fields_are_positive_and_bounded(self):
        rng = np.random.default_rng(2)
        x, y = np.meshgrid(np.linspace(0, 2*math.pi, 20), np.linspace(0, 2*math.pi, 20))
        self.assertTrue((eq.allen_cahn_kappa(rng.uniform(-1, 1, 57), x, y) > 0).all())
        self.assertTrue((eq.adr_kappa(np.clip(rng.standard_normal(60), -1, 1), 4, x, y) > 0).all())
        self.assertGreaterEqual(eq.burgers2d_kappa(np.ones(20), x, y).min(), 0.0035-1e-12)
        self.assertEqual(len(set(eq.wave_vectors(28))), 28)


class SuiteTests(unittest.TestCase):
    def test_every_family_in_every_split_and_views_hide_labels(self):
        for split in SPLITS:
            cases = make_cases(split)
            self.assertEqual({c.truth.spec["family_key"] for c in cases}, set(FAMILIES))
            for case in cases:
                if not reference_path(case.truth).exists():
                    continue
                view = case.public_view()
                text = repr(view)
                for hidden in ("family_key", "kind", "class", case.family, case.truth.spec["family_key"]):
                    self.assertNotIn(f"'{hidden}'", text) if hidden in ("family_key", "kind", "class") else None
                self.assertNotIn(case.family, text)

    def test_self_test_covers_each_layout(self):
        protocol = InventoryProtocol()
        chosen = protocol.self_test_tasks(make_cases("evolution"))
        self.assertEqual(len({c.truth.layout for c in chosen}), len(chosen))

    @unittest.skipUnless(shutil.which("bwrap"), "bubblewrap is required")
    def test_contract_program_runs_every_built_case_in_sandbox(self):
        check_program(SHAPE_PROGRAM, "identify")
        cases = [c for c in make_cases("evolution") if reference_path(c.truth).exists()]
        self.assertTrue(cases, "no references built")
        budget = Budget(len(cases), 3600, max_cpu_seconds=3600)
        for case in cases:
            result = run_case(case, SHAPE_PROGRAM, budget, gpu=False)
            self.assertEqual(result["status"], "ok", (case.task_id, result.get("message")))
            self.assertTrue(result["constraints_pass"], (case.task_id, result))
            self.assertFalse(result["passed"])  # zero solution is inaccurate
            self.assertEqual(result["identified_family"], "heat")


if __name__ == "__main__":
    unittest.main()
