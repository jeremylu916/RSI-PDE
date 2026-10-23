"""pde_benchmarks.tex suite: exact data, splits, hidden labels, evaluator, and sandbox contract."""

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pde_rsi import equations_bench as eq
from pde_rsi.codegen import check_program
from pde_rsi.pdebench import (CLASS, LIBRARY, PDEBenchProtocol, SPLITS, evaluate, evaluation_nodes, make_cases,
                              reference, run_case, score_model, true_parameters)
from pde_rsi.runner import Budget
from pde_rsi.solvers import Solution

# Returns the right shapes, IC and BC, and u = 0 elsewhere: contract-valid but inaccurate.
SHAPE_PROGRAM = '''
import numpy as np

def solve(problem):
    d = problem.domain
    model = {"family": "heat", "parameters": {"D": 1.0}, "method": "numerical"}
    if problem.spatial_dimension == 0:
        t = np.linspace(0, d["t"][1], 11)
        u = np.zeros(11)
        u[0] = problem.initial()
        float(problem.forcing(0.5)) + float(problem.forcing(np.array([0.5]))[0])
        assert problem.forcing(t).shape == t.shape
        return {"t": t, "u": u, "model": model}
    x = np.linspace(d["x"][0], d["x"][1], 33)
    if problem.spatial_dimension == 2:
        y = np.linspace(d["y"][0], d["y"][1], 33)
        t = np.linspace(0, d["t"][1], 5)
        X, Y = np.meshgrid(x, y)
        u = np.zeros((5, 33, 33))
        u[0] = problem.initial(X, Y)
        u[:, 0, :] = u[:, -1, :] = u[:, :, 0] = u[:, :, -1] = 0.0
        problem.forcing(X, Y, 0.5)
        float(problem.forcing(0.3, 0.2, 0.1))
        problem.initial_velocity(X, Y)
        return {"x": x, "y": y, "t": t, "u": u, "model": model}
    if not problem.time_dependent:
        u = np.zeros(33)
        u[0], u[-1] = problem.boundary(0.0)
        problem.forcing(x)
        float(problem.forcing(0.5*(d["x"][0]+d["x"][1])))
        return {"x": x, "u": u, "model": model}
    t = np.linspace(0, d["t"][1], 9)
    u = np.zeros((9, 33))
    u[0] = problem.initial(x)
    for n in range(9):
        u[n, 0], u[n, -1] = problem.boundary(t[n])
    problem.forcing(x, 0.0)
    float(problem.forcing(x[3], t[2]))
    assert problem.forcing(x[:, None], t[None, :]).shape == (33, 9)
    float(problem.boundary(t[1])[0])
    return {"x": x, "t": t, "u": u, "model": model}
'''


def all_cases():
    return [case for split in SPLITS for case in make_cases(split)]


def _test_splits_cover_every_class_and_hide_identity():
    for split in ("evolution", "development"):
        cases = make_cases(split)
        assert {c.family for c in cases} == set(LIBRARY)
        for case in cases:
            view = case.public_view()
            text = repr(view)
            assert case.family not in text and case.truth.problem not in view.values()
            assert "spec" not in view and "A" not in view
    ids = [c.task_id for c in all_cases()]
    assert len(ids) == len(set(ids))


def _test_manufactured_profiles_follow_tex_splits():
    for split, profile in (("evolution", "sine"), ("development", "polynomial"), ("test", "mixed_sine")):
        for case in make_cases(split):
            if case.truth.problem in ("1a", "2a", "3a", "4a", "5", "7", "8"):
                assert case.truth.spec["profile"] == profile


def _test_initial_and_boundary_data_match_exact_solution():
  for case in [c for c in all_cases() if c.truth.dimension == 1 and not c.truth.steady]:
    s = case.truth.spec
    x = np.linspace(s["a"], s["b"], 9)
    assert np.allclose(eq.initial(s, x), eq.exact(s, x, 0.0), atol=1e-12)
    for t in (0.0, 0.5*s["T"], s["T"]):
      assert np.allclose(eq.boundary(s, t), [eq.exact(s, s["a"], t), eq.exact(s, s["b"], t)], atol=1e-12)


def _test_heat_forcing_matches_operator():
    case = next(c for c in make_cases("evolution") if c.truth.problem == "2a")
    s, h = case.truth.spec, 1e-4
    x, t = np.linspace(0.1, s["b"]-0.1, 5), 0.6*s["T"]
    ut = (eq.exact(s, x, t+h)-eq.exact(s, x, t-h))/(2*h)
    uxx = (eq.exact(s, x+h, t)-2*eq.exact(s, x, t)+eq.exact(s, x-h, t))/h**2
    assert np.allclose(ut-s["D"]*uxx, eq.forcing(s, x, t), atol=1e-5)


def _test_burgers_cole_hopf_peak_slope():
    case = next(c for c in make_cases("development") if c.truth.problem == "4b")
    x = np.linspace(-1e-3, 1e-3, 201)
    slope = np.gradient(eq.burgers_cole_hopf(case.truth.spec, x, 1.6037/math.pi), x)[100]
    assert abs(slope+152.005) < 0.5  # Basdevant et al. (1986): max |u_x| = 152.00516


def _test_exact_solution_scores_zero_and_shapes_are_checked():
    for split in ("evolution", "development"):
        for case in make_cases(split):
            task = case.truth
            if task.problem == "4b":
                continue  # covered by the quadrature test; its reference takes ~30 s
            x, t = evaluation_nodes(task)
            u = reference(task)
            if task.dimension == 0:
                u[0] = 0.0
            solution = Solution(x, t, u, "exact", x if task.dimension == 2 else None)
            if task.dimension == 1:
                solution.u[:, 0] = [eq.boundary(task.spec, ti)[0] for ti in t]
                solution.u[:, -1] = [eq.boundary(task.spec, ti)[1] for ti in t]
            result = evaluate(task, solution)
            assert result["relative_l2"] < 1e-12 and result["passed"], (task.task_id, result)


def _test_score_model_and_ratio_parameter():
    case = next(c for c in make_cases("evolution") if c.truth.problem == "3b")
    s = case.truth.spec
    good = score_model(case, {"family": "convection_diffusion", "parameters": {"v": 2*s["v"], "D": 2*s["D"]}})
    assert good["family_correct"] and good["parameter_relative_errors"]["v/D"] < 1e-12
    assert not score_model(case, {"family": "poisson"})["family_correct"]
    assert set(true_parameters(case.truth)) == {"v/D"}


def _test_classes_cover_library():
    assert set(CLASS.values()) == set(LIBRARY)


@unittest.skipUnless(shutil.which("bwrap"), "bubblewrap is required")
def _test_contract_program_passes_policy_and_runs_every_case_in_sandbox():
    tmp_path = Path(tempfile.mkdtemp())
    check_program(SHAPE_PROGRAM, "identify")
    cases = [c for c in make_cases("evolution")]
    budget = Budget(len(cases), 3600, max_cpu_seconds=3600)
    for case in cases:
        result = run_case(case, SHAPE_PROGRAM, budget, tmp_path/f"{case.case_id}.npz", gpu=False)
        assert result["status"] == "ok", (case.task_id, result.get("message"))
        assert result["constraints_pass"], (case.task_id, result)
        assert result["identified_family"] == "heat"
        assert (tmp_path/f"{case.case_id}.npz").exists()


def _test_protocol_hides_validation_labels():
    protocol = PDEBenchProtocol()
    tasks = protocol.load_tasks(None, None, ())
    rows = [{"case_id": c.case_id, "split": c.split, "task_id": c.task_id, "true_family": c.family,
             "family_correct": False} for c in tasks["evolution"][:1]+tasks["development"][:1]]
    diagnostics = protocol.diagnostics(rows)
    assert "true_family" in diagnostics[0] and "true_family" not in diagnostics[1]
    reasons = protocol.reasons(["family regression: heat", f"instance regression: {tasks['development'][0].task_id}"],
                               tasks["development"])
    assert reasons == ["family regression: a hidden equation class",
                       f"instance regression: {tasks['development'][0].case_id}"]


def _test_self_test_covers_every_output_kind():
    protocol = PDEBenchProtocol()
    chosen = protocol.self_test_tasks(make_cases("evolution"))
    assert {(c.truth.dimension, c.truth.steady) for c in chosen} == {(1, False), (1, True), (0, False), (2, False)}


def _gpu_available():
    try:
        from pde_rsi.codegen import gpu_sandbox_self_test
        return bool(shutil.which("bwrap")) and gpu_sandbox_self_test()["cuda"]
    except Exception:
        return False


def _test_cuda_training_runs_inside_sandbox():
    if not _gpu_available():
        raise unittest.SkipTest("no GPU sandbox on this host")
    program = SHAPE_PROGRAM.replace("import numpy as np", "import numpy as np\nimport torch").replace(
        "    d = problem.domain", "    d = problem.domain\n    net = torch.nn.Linear(2, 4).to(problem.device)\n"
        "    net(torch.rand(8, 2, device=problem.device)).sum().backward()")
    case = make_cases("development")[0]
    result = run_case(case, program, Budget(1, 600, max_cpu_seconds=600), gpu=True)
    assert result["status"] == "ok", result.get("message")


# A contract-shaped program that never reads the hidden layout fields (hidden-time-dependence variant).
HIDDEN_PROGRAM = """
import numpy as np

def solve(problem):
    d = problem.domain
    model = {"family": "heat", "parameters": {"D": 1.0}, "method": "numerical"}
    t = np.linspace(0, d["t"][1], 9)
    if problem.spatial_dimension == 0:
        u = np.zeros(9)
        u[0] = problem.initial()
        float(problem.forcing(0.5))
        return {"t": t, "u": u, "model": model}
    x = np.linspace(d["x"][0], d["x"][1], 33)
    if problem.spatial_dimension == 2:
        X, Y = np.meshgrid(x, x)
        u = np.zeros((9, 33, 33))
        u[0] = problem.initial(X, Y)
        u[:, 0, :] = u[:, -1, :] = u[:, :, 0] = u[:, :, -1] = 0.0
        return {"x": x, "y": x, "t": t, "u": u, "model": model}
    u = np.zeros((9, 33))
    u[0] = problem.initial(x)
    for n in range(9):
        u[n, 0], u[n, -1] = problem.boundary(t[n])
    float(problem.forcing(x[3], t[2]))
    return {"x": x, "t": t, "u": u, "model": model}
"""


def _test_hidden_time_dependence_poses_every_case_in_time():
    for split in SPLITS:
        hidden = make_cases(split, hide_time_dependence=True)
        plain = {c.task_id: c for c in make_cases(split)}
        for case in hidden:
            view = case.public_view()
            assert "time_dependent" not in view and "coordinates" not in view
            assert "t" in view["domain"] and case.data()["t"] is not None and not case.truth.steady
            if case.truth.problem in ("1a", "1b", "1c", "3b", "3c"):
                s, x = case.truth.spec, np.linspace(case.truth.spec["a"], case.truth.spec["b"], 7)
                u = eq.exact(s, x[None, :], np.linspace(0, s["T"], 5)[:, None])
                assert np.allclose(u, u[0]) and np.allclose(eq.initial(s, x), u[0])
                assert plain[case.task_id].truth.steady and case.family == plain[case.task_id].family
                assert reference(case.truth).shape == (500, 512)
    from pde_rsi.pdebench import HIDDEN_TIME_CONTRACT
    assert "time_dependent" not in HIDDEN_TIME_CONTRACT and "NOT told" in HIDDEN_TIME_CONTRACT


@unittest.skipUnless(shutil.which("bwrap"), "bubblewrap is required")
def _test_hidden_time_program_runs_every_kind_in_sandbox():
    check_program(HIDDEN_PROGRAM, "identify")
    cases = make_cases("evolution", hide_time_dependence=True)
    budget = Budget(len(cases), 3600, max_cpu_seconds=3600)
    for case in cases:
        result = run_case(case, HIDDEN_PROGRAM, budget, gpu=False)
        assert result["status"] == "ok" and result["constraints_pass"], (case.task_id, result.get("message"))


def _caputo(fun, alpha, t):
    """Caputo derivative of a scalar function of t by singular-weight quadrature of u'(s)."""
    from scipy.integrate import quad
    from scipy.special import gamma
    h = 1e-6
    value, _ = quad(lambda s_: (fun(s_+h)-fun(max(s_-h, 0.0)))/(s_+h-max(s_-h, 0.0)), 0, t, weight="alg",
                    wvar=(0, -alpha), limit=200)
    return value/gamma(1-alpha)


def _operator_residual(case):
    """Max relative mismatch between forcing and the true operator applied to the exact solution."""
    task, s, h = case.truth, case.truth.spec, 1e-4
    if task.dimension == 0:
        return max(abs(_caputo(lambda t_: float(eq.exact(s, 0.0, t_)), s["alpha"], t)-float(eq.forcing(s, 0.0, t)))
                   for t in (0.3, 0.7, 1.0))
    worst = 0.0
    xs = np.linspace(s["a"], s["b"], 6)[1:-1]
    times = [0.0] if task.steady else [0.4*s["T"], 0.8*s["T"]]
    for t in times:
        for x in xs:
            if task.dimension == 2:
                y = s["a"]+0.37*(s["b"]-s["a"])
                U = lambda x_, t_, y_=y: float(eq.exact(s, x_, t_, y_))
                lap = (U(x+h, t)-2*U(x, t)+U(x-h, t))/h**2+(U(x, t, y+h)-2*U(x, t)+U(x, t, y-h))/h**2
                lhs = (U(x, t+h)-2*U(x, t)+U(x, t-h))/h**2-s["D"]*lap
                f = float(eq.forcing(s, x, t, y))
            else:
                U = lambda x_, t_: float(eq.exact(s, x_, t_))
                uxx = (U(x+h, t)-2*U(x, t)+U(x-h, t))/h**2
                ux = (U(x+h, t)-U(x-h, t))/(2*h)
                f, fam = float(eq.forcing(s, x, t)), task.family
                if fam == "poisson":
                    lhs = -s["D"]*uxx
                elif task.steady:
                    lhs = s["v"]*ux-s["D"]*uxx
                else:
                    ut = (U(x, t+h)-U(x, t-h))/(2*h)
                    history = lambda order: _caputo(lambda tt: U(x, tt), order, t)
                    lhs = {"heat": lambda: ut-s["D"]*uxx,
                           "convection_diffusion": lambda: ut+s["v"]*ux-s["D"]*uxx,
                           "burgers": lambda: ut+U(x, t)*ux-s["D"]*uxx,
                           "caputo": lambda: history(s["alpha"])-s["D"]*uxx,
                           "multi_term": lambda: s["w1"]*history(s["alpha1"])+s["w2"]*history(s["alpha2"])-s["D"]*uxx,
                           }.get(fam, lambda: history(float(eq.order(s, np.array([x]), t)[0]))-s["D"]*uxx)()
            worst = max(worst, abs(lhs-f)/(1+abs(f)))
    return worst


def _test_fresh_test_split_is_consistent_and_disjoint():
    cases = make_cases("test2")
    assert sorted(c.truth.problem for c in cases) == sorted(CLASS) and all(c.case_id.startswith("case-s") for c in cases)
    seen = {json.dumps({k: v for k, v in c.truth.spec.items() if k != "profile"}, sort_keys=True)
            for split in SPLITS for c in make_cases(split)}
    for case in cases:
        s = case.truth.spec
        assert json.dumps({k: v for k, v in s.items() if k != "profile"}, sort_keys=True) not in seen, case.task_id
        assert _operator_residual(case) < 1e-4, (case.task_id, _operator_residual(case))
        if case.truth.dimension == 1:
            T = 0.0 if case.truth.steady else s["T"]
            assert np.allclose(eq.boundary(s, T), [eq.exact(s, s["a"], T), eq.exact(s, s["b"], T)], atol=1e-12)
            if not case.truth.steady:
                x = np.linspace(s["a"], s["b"], 9)
                assert np.allclose(eq.initial(s, x), eq.exact(s, x, 0.0), atol=1e-12)
        assert np.isfinite(reference(case.truth)).all()


class PDEBenchTest(unittest.TestCase):
    def test_fresh_test_split_is_consistent_and_disjoint(self):
        _test_fresh_test_split_is_consistent_and_disjoint()

    def test_hidden_time_dependence_poses_every_case_in_time(self):
        _test_hidden_time_dependence_poses_every_case_in_time()

    def test_hidden_time_program_runs_every_kind_in_sandbox(self):
        _test_hidden_time_program_runs_every_kind_in_sandbox()

    def test_self_test_covers_every_output_kind(self):
        _test_self_test_covers_every_output_kind()

    def test_cuda_training_runs_inside_sandbox(self):
        _test_cuda_training_runs_inside_sandbox()

    def test_splits_cover_every_class_and_hide_identity(self):
        _test_splits_cover_every_class_and_hide_identity()

    def test_manufactured_profiles_follow_tex_splits(self):
        _test_manufactured_profiles_follow_tex_splits()

    def test_initial_and_boundary_data_match_exact_solution(self):
        _test_initial_and_boundary_data_match_exact_solution()

    def test_heat_forcing_matches_operator(self):
        _test_heat_forcing_matches_operator()

    def test_burgers_cole_hopf_peak_slope(self):
        _test_burgers_cole_hopf_peak_slope()

    def test_exact_solution_scores_zero_and_shapes_are_checked(self):
        _test_exact_solution_scores_zero_and_shapes_are_checked()

    def test_score_model_and_ratio_parameter(self):
        _test_score_model_and_ratio_parameter()

    def test_classes_cover_library(self):
        _test_classes_cover_library()

    def test_contract_program_passes_policy_and_runs_every_case_in_sandbox(self):
        _test_contract_program_passes_policy_and_runs_every_case_in_sandbox()

    def test_protocol_hides_validation_labels(self):
        _test_protocol_hides_validation_labels()


if __name__ == "__main__":
    unittest.main()
