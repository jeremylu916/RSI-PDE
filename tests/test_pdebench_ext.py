"""pde_benchmarks.tex Problems 12-16: exact solutions satisfy their PDEs, the evaluator accepts them, and the sandbox
runs a program on every layout."""

import math
import shutil
import types
import unittest

import numpy as np

from pde_rsi import equations_ext as eq
from pde_rsi import pdebench_ext as px
from pde_rsi.runner import Budget

H = 1e-2  # fourth-order central differences; small enough for truncation, large enough for round-off in u_xxxx


def d(g, var, n=1):
    def derivative(**kw):
        def at(shift):
            return g(**{**kw, var: kw[var]+shift})
        return (-at(2*H)+8*at(H)-8*at(-H)+at(-2*H))/(12*H)
    return derivative if n == 1 else d(derivative, var, n-1)


def submission(task):
    """The exact solution on the evaluation grid, in the solver output format."""
    axes, ref = px.evaluation_axes(task), px.reference(task)
    if task.layout == "2d_steady":
        u = ref[0]
    else:
        u = ref
    return types.SimpleNamespace(x=axes["x"], y=axes.get("y"), t=axes.get("t", np.zeros(1)), u=u, method="exact")


class EquationTests(unittest.TestCase):
    def test_closed_forms_satisfy_their_equations(self):
        rng = np.random.default_rng(1)
        for case in px.make_cases("evolution"):
            s = case.truth.spec
            if not eq.has_closed_form(s):
                continue
            y0, y1 = px.y_range(s)
            P = dict(x=s["a"]+(s["b"]-s["a"])*rng.uniform(0.2, 0.8, 20), y=y0+(y1-y0)*rng.uniform(0.2, 0.8, 20),
                     t=(s["T"] or 1.0)*rng.uniform(0.3, 0.9, 20))

            def F(j):
                return lambda x, y, t: eq.exact(s, x, t, y)[..., j]

            def f(j):
                return eq.forcing(s, P["x"], P["t"], P["y"])[..., j]

            def lap(g):
                return d(g, "x", 2)(**P)+d(g, "y", 2)(**P)
            klass, residuals = case.family, []
            if klass == "kuramoto_sivashinsky":
                u = F(0)
                residuals.append(d(u, "t")(**P)+u(**P)*d(u, "x")(**P)+d(u, "x", 2)(**P)+d(u, "x", 4)(**P)-f(0))
            elif klass == "navier_stokes_2d":
                u, v, p = F(0), F(1), F(2)
                ut = d(u, "t")(**P) if s["T"] else 0
                vt = d(v, "t")(**P) if s["T"] else 0
                residuals += [ut+u(**P)*d(u, "x")(**P)+v(**P)*d(u, "y")(**P)+d(p, "x")(**P)-s["nu"]*lap(u),
                              vt+u(**P)*d(v, "x")(**P)+v(**P)*d(v, "y")(**P)+d(p, "y")(**P)-s["nu"]*lap(v),
                              d(u, "x")(**P)+d(v, "y")(**P)]
            elif klass == "cahn_hilliard_2d":
                u = F(0)

                def mu(x, y, t):
                    value = u(x=x, y=y, t=t)
                    return value**3-value-s["eps"]**2*(d(u, "x", 2)(x=x, y=y, t=t)+d(u, "y", 2)(x=x, y=y, t=t))
                residuals.append(d(u, "t")(**P)-s["M"]*lap(mu)-f(0))
            elif klass == "maxwell_te":
                ex, ey, hz = F(0), F(1), F(2)
                c = s["c"]
                residuals += [d(ex, "t")(**P)-(c*d(hz, "y")(**P)-f(0)), d(ey, "t")(**P)-(-c*d(hz, "x")(**P)-f(1)),
                              d(hz, "t")(**P)+c*(d(ey, "x")(**P)-d(ex, "y")(**P))]
            else:
                ex, ey = F(0), F(1)

                def curl(x, y, t):
                    return d(ey, "x")(x=x, y=y, t=t)-d(ex, "y")(x=x, y=y, t=t)
                residuals += [d(curl, "y")(**P)-s["k"]**2*ex(**P)-f(0), -d(curl, "x")(**P)-s["k"]**2*ey(**P)-f(1)]
            scale = max(1.0, float(np.abs(eq.forcing(s, P["x"], P["t"], P["y"])).max()))
            for r in residuals:
                self.assertLess(float(np.abs(r).max())/scale, 1e-5, (case.task_id, klass))


class EvaluatorTests(unittest.TestCase):
    def test_exact_and_reference_solutions_pass_every_split(self):
        for split in px.SPLITS:
            for case in px.make_cases(split):
                if not eq.has_closed_form(case.truth.spec) and not px.reference_path(case.truth).exists():
                    self.skipTest("run scripts/build_pdebench_ext_references.py first")
                result = px.evaluate(case.truth, submission(case.truth))
                self.assertTrue(result["passed"], (case.task_id, result))
                self.assertLess(result["relative_l2"], 1e-12)

    def test_perturbation_fails_and_pressure_constant_is_free(self):
        task = next(c.truth for c in px.make_cases("development") if c.truth.spec["problem"] == "13b")
        good = submission(task)
        shifted = types.SimpleNamespace(**{**vars(good), "u": good.u+np.array([0.0, 0.0, 5.0])})
        self.assertTrue(px.evaluate(task, shifted)["passed"])
        wrong = types.SimpleNamespace(**{**vars(good), "u": good.u*(1+1e-2)})
        result = px.evaluate(task, wrong)
        self.assertFalse(result["accuracy_pass"])
        self.assertFalse(result["constraints_pass"])  # the Dirichlet velocity no longer holds

    def test_noisy_data_has_one_column_per_component(self):
        for case in px.make_cases("evolution"):
            if not eq.has_closed_form(case.truth.spec) and not px.reference_path(case.truth).exists():
                continue
            data = case.data()
            self.assertEqual(data["u"].shape, (px.N_POINTS, len(case.truth.components)))
            self.assertEqual(len(data["noise_std"]), len(case.truth.components))


PROGRAM = '''import numpy as np


def solve(problem):
    x = np.linspace(problem.domain["x"][0], problem.domain["x"][1], 33)
    model = {"family": "unknown", "parameters": {}, "method": "numerical"}
    k = len(problem.components)
    if problem.layout == "1d_periodic_time":
        t = np.linspace(0.0, problem.domain["t"][1], 5)
        f = problem.forcing(x[None, :], t[:, None])
        u = np.zeros((len(t), len(x))) + problem.initial(x)[None, :] + 0*f
        u[:, -1] = u[:, 0]
        return {"x": x, "t": t, "u": u, "model": model}
    y = np.linspace(problem.domain["y"][0], problem.domain["y"][1], 33)
    X, Y = np.meshgrid(x, y)
    if problem.layout == "2d_steady":
        f = problem.forcing(X, Y)
        u = np.zeros(X.shape + (k,))
        if problem.boundary_description.startswith("Dirichlet"):
            edge = np.zeros(X.shape, dtype=bool)
            edge[[0, -1], :] = True
            edge[:, [0, -1]] = True
            u[edge, :2] = problem.boundary(X[edge], Y[edge])
        return {"x": x, "y": y, "u": u, "model": model}
    t = np.linspace(0.0, problem.domain["t"][1], 5)
    f = problem.forcing(X[None], Y[None], t[:, None, None])
    first = problem.initial(X, Y)
    first = first if first.ndim == 3 else first[..., None]
    u = np.repeat(first[None], len(t), axis=0)
    return {"x": x, "y": y, "t": t, "u": u if k > 1 else u[..., 0], "model": model}
'''


@unittest.skipUnless(shutil.which("bwrap"), "needs bubblewrap")
class SandboxTests(unittest.TestCase):
    def test_program_runs_on_every_layout(self):
        seen = set()
        for case in px.make_cases("evolution"):
            if case.truth.layout in seen or not eq.has_closed_form(case.truth.spec):
                continue
            seen.add(case.truth.layout)
            result = px.run_case(case, PROGRAM, Budget(1, 120, max_cpu_seconds=120), gpu=False)
            self.assertEqual(result["status"], "ok", (case.task_id, result.get("message")))
            self.assertLess(result["boundary_error"], 1e-10, case.task_id)
            self.assertLess(result["initial_error"], 1e-10, case.task_id)
        self.assertEqual(seen, {"1d_periodic_time", "2d_periodic_time", "2d_time", "2d_steady"})


if __name__ == "__main__":
    unittest.main()
