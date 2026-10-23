"""The 1001-case expanded release: counts, splits, no overlap with existing cases, and lifted Dirichlet data."""

import collections
import unittest

import numpy as np

from pde_rsi import equations_bench as eq
from pde_rsi import expanded as X

H = 1e-4


def d(g, var, **P):
    a = {**P, var: P[var]+H}
    b = {**P, var: P[var]-H}
    return (g(**a)-g(**b))/(2*H)


def d2(g, var, **P):
    a = {**P, var: P[var]+H}
    b = {**P, var: P[var]-H}
    return (g(**a)-2*g(**P)+g(**b))/(H*H)


class ExpandedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = X.all_cases()

    def test_counts_and_splits(self):
        self.assertEqual(sum(len(v) for v in self.cases.values()), 1001)
        splits = collections.Counter()
        for suite, items in self.cases.items():
            for item in items:
                splits[X.release_split(item) if suite == "manufactured" else item.truth.split] += 1
        self.assertEqual(dict(splits), {"train": 705, "validation": 148, "test": 148})

    def test_no_case_repeats_an_existing_one(self):
        self.assertEqual(X.sealed_overlap(), [])

    def test_variants_are_distinct(self):
        keys = [X._key(c.truth.spec) for suite in ("pdebench", "pdebench_ext", "inventory") for c in self.cases[suite]]
        keys += [str({k: v for k, v in t.to_dict().items() if k not in ("task_id", "template_group")})
                 for t in self.cases["manufactured"]]
        self.assertEqual(len(set(keys)), 1001)

    def test_lifted_dirichlet_data_satisfy_the_equations(self):
        lifted = [c for c in self.cases["pdebench"] if c.truth.spec.get("bc_left")]
        self.assertGreater(len(lifted), 60)
        for case in lifted:
            s = case.truth.spec
            x = np.linspace(s["a"], s["b"], 9)[1:-1]
            t = np.full_like(x, 0.5*(s["T"] or 1.0))

            def u(x, t):
                return eq.exact(s, x, t)
            values = eq.boundary(s, 0.3)
            self.assertAlmostEqual(float(u(s["a"], 0.3)), values[0], places=12)
            self.assertAlmostEqual(float(u(s["b"], 0.3)), values[1], places=12)
            if case.truth.problem in ("2a", "3a", "4a"):
                residual = d(u, "t", x=x, t=t)-s["D"]*d2(u, "x", x=x, t=t)-eq.forcing(s, x, t)
                if case.truth.problem == "3a":
                    residual += s["v"]*d(u, "x", x=x, t=t)
                if case.truth.problem == "4a":
                    residual += u(x, t)*d(u, "x", x=x, t=t)
                self.assertLess(float(np.abs(residual).max()), 2e-5*max(1.0, float(np.abs(eq.forcing(s, x, t)).max())),
                                case.case_id)


if __name__ == "__main__":
    unittest.main()
