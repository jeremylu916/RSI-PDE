"""Classical solver: nonuniform L1 Caputo quadrature with implicit centered diffusion.

Handles any positive weighted sum of Caputo orders in (0, 1).
"""
import math
import numpy as np
from scipy.linalg import solve_banded

NX = 32
NT = 48
GRADING = 1.0


def l1_weights(times, n, alpha):
    left = times[n] - times[:n]
    right = times[n] - times[1:n + 1]
    p = 1 - alpha
    difference = np.empty(n)
    nonzero = right > 0
    difference[~nonzero] = left[~nonzero] ** p
    ratio = np.diff(times[:n + 1])[nonzero] / left[nonzero]
    difference[nonzero] = left[nonzero] ** p * (-np.expm1(p * np.log1p(-ratio)))
    return difference / (math.gamma(2 - alpha) * np.diff(times[:n + 1]))


def solve(problem):
    x = np.linspace(0.0, problem.length, NX + 1)
    dx = x[1] - x[0]
    t = problem.final_time * np.linspace(0.0, 1.0, NT + 1) ** GRADING
    d = problem.diffusivity / dx**2
    u = np.zeros((NT + 1, NX + 1))
    u[0] = problem.initial(x)
    u[0, [0, -1]] = 0.0
    for n in range(1, NT + 1):
        coefficients = sum(w * l1_weights(t, n, a) for a, w in zip(problem.orders, problem.weights))
        leading = coefficients[-1]
        history = coefficients[:-1] @ np.diff(u[:n, 1:-1], axis=0) if n > 1 else 0.0
        rhs = problem.forcing(x[1:-1], t[n]) + leading * u[n - 1, 1:-1] - history
        ab = np.zeros((3, NX - 1))
        ab[0, 1:] = -d
        ab[1, :] = leading + 2 * d
        ab[2, :-1] = -d
        u[n, 1:-1] = solve_banded((1, 1), ab, rhs)
    return {"x": x, "t": t, "u": u}
