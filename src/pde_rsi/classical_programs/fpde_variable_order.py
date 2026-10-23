"""Classical solver: uniform L1 quadrature with the order alpha(t_n) at each step; implicit centered diffusion."""
import math
import numpy as np
from scipy.linalg import solve_banded

NX = 32
NT = 48


def l1_weights(t, n, alpha):
    left = t[n] - t[:n]
    right = t[n] - t[1:n + 1]
    return (left ** (1 - alpha) - right ** (1 - alpha)) / (math.gamma(2 - alpha) * np.diff(t[:n + 1]))


def solve(problem):
    x = np.linspace(0.0, problem.length, NX + 1)
    t = np.linspace(0.0, problem.final_time, NT + 1)
    d = problem.diffusivity / (x[1] - x[0]) ** 2
    u = np.zeros((NT + 1, NX + 1))
    u[0] = problem.initial(x)
    for n in range(1, NT + 1):
        weights = l1_weights(t, n, problem.alpha(t[n]))
        rhs = problem.forcing(x[1:-1], t[n]) + weights[-1] * u[n - 1, 1:-1]
        if n > 1:
            rhs = rhs - weights[:-1] @ np.diff(u[:n, 1:-1], axis=0)
        ab = np.zeros((3, NX - 1))
        ab[0, 1:] = -d
        ab[1, :] = weights[-1] + 2 * d
        ab[2, :-1] = -d
        u[n, 1:-1] = solve_banded((1, 1), ab, rhs)
    return {"x": x, "t": t, "u": u}
