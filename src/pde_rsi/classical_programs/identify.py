"""Classical identification baseline: fit each library equation to the data, choose by BIC, then solve finely.

Coarse implicit solvers (backward Euler; IMEX Burgers; L1 Caputo with constant, time-, or space-time-
variable order) are fitted by Nelder-Mead on transformed parameters. The class with the lowest BIC is
re-solved on a finer grid and returned with its parameters.
"""
import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import solve_banded
from scipy.optimize import minimize
from scipy.special import gamma

COARSE = (32, 48)
FINE = (128, 160)
STARTS = {"heat": {"D": 0.1}, "convection_diffusion": {"D": 0.1, "v": 0.3}, "burgers": {"D": 0.1},
          "caputo": {"D": 0.1, "alpha": 0.5}, "multi_term": {"D": 0.1, "alpha1": 0.4, "alpha2": 0.75, "w1": 0.5},
          "variable_order_t": {"D": 0.5, "c": 0.5}, "variable_order_xt": {"D": 0.5, "c": 0.5}}


def to_free(name, value):
    if name in ("alpha", "alpha1", "alpha2", "c"):
        return math.log(value / (1 - value))
    if name == "v":
        return value
    return math.log(value)


def to_param(name, value):
    if name in ("alpha", "alpha1", "alpha2", "c"):
        return 0.02 + 0.96 / (1 + math.exp(-value))
    if name == "v":
        return value
    return math.exp(min(max(value, -12.0), 3.0))


def l1_weights(t, n, alpha):
    left = t[n] - t[:n]
    right = t[n] - t[1:n + 1]
    steps = np.diff(t[:n + 1])
    a = np.asarray(alpha, dtype=float)
    if a.ndim == 0:
        return (left ** (1 - a) - right ** (1 - a)) / (gamma(2 - a) * steps)
    return (left[:, None] ** (1 - a) - right[:, None] ** (1 - a)) / (gamma(2 - a) * steps[:, None])


def simulate(problem, family, p, nx, nt):
    x = np.linspace(0.0, problem.length, nx + 1)
    t = np.linspace(0.0, problem.final_time, nt + 1)
    dx = x[1] - x[0]
    inner = x[1:-1]
    d = p["D"] / dx ** 2
    u = np.zeros((nt + 1, nx + 1))
    u[0] = problem.initial(x)
    u[0, 0] = u[0, -1] = 0.0
    for n in range(1, nt + 1):
        f = problem.forcing(inner, t[n])
        lower = np.full(nx - 1, -d)
        upper = np.full(nx - 1, -d)
        diag = np.full(nx - 1, 2 * d)
        if family in ("heat", "convection_diffusion", "burgers"):
            dt = t[n] - t[n - 1]
            diag = diag + 1 / dt
            rhs = u[n - 1, 1:-1] / dt + f
            if family == "convection_diffusion":
                c = p["v"] / (2 * dx)
                lower = lower - c
                upper = upper + c
            if family == "burgers":
                prev = u[n - 1]
                rhs = rhs - prev[1:-1] * (prev[2:] - prev[:-2]) / (2 * dx)
        else:
            if family == "caputo":
                w = l1_weights(t, n, p["alpha"])
            elif family == "multi_term":
                w = p["w1"] * l1_weights(t, n, p["alpha1"]) + l1_weights(t, n, p["alpha2"])
            elif family == "variable_order_t":
                w = l1_weights(t, n, p["c"] * math.sin(t[n]) ** 2)
            else:
                w = l1_weights(t, n, p["c"] * np.sin(inner) * t[n] ** 2)
            if w.ndim == 1:
                w = w[:, None] * np.ones(nx - 1)
            history = np.sum(w[:-1] * np.diff(u[:n, 1:-1], axis=0), axis=0) if n > 1 else 0.0
            diag = diag + w[-1]
            rhs = f + w[-1] * u[n - 1, 1:-1] - history
        ab = np.zeros((3, nx - 1))
        ab[0, 1:] = upper[:-1]
        ab[1, :] = diag
        ab[2, :-1] = lower[1:]
        u[n, 1:-1] = solve_banded((1, 1), ab, rhs)
        if not np.all(np.isfinite(u[n])):
            raise FloatingPointError("unstable")
    return x, t, u


def misfit(problem, family, p, grid):
    try:
        x, t, u = simulate(problem, family, p, *grid)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return 1e30
    values = RegularGridInterpolator((t, x), u)(np.column_stack([problem.data_t, problem.data_x]))
    return float(np.mean((values - problem.data_u) ** 2))


def fit(problem, family):
    names = list(STARTS[family])
    start = np.array([to_free(n, STARTS[family][n]) for n in names])

    def objective(z):
        return misfit(problem, family, {n: to_param(n, v) for n, v in zip(names, z)}, COARSE)

    best = minimize(objective, start, method="Nelder-Mead",
                    options={"maxfev": 40 * len(names), "xatol": 1e-3, "fatol": 1e-14})
    return {n: to_param(n, v) for n, v in zip(names, best.x)}, float(best.fun)


def solve(problem):
    n = len(problem.data_u)
    choices = []
    for family in problem.library:
        if family not in STARTS:
            continue
        params, error = fit(problem, family)
        bic = n * math.log(max(error, 1e-300)) + len(params) * math.log(n)
        choices.append((bic, family, params))
    _, family, params = min(choices)
    x, t, u = simulate(problem, family, params, *FINE)
    return {"x": x, "t": t, "u": u, "model": {"family": family, "parameters": params}}
