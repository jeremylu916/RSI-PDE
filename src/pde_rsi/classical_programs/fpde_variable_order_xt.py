"""Classical solver: uniform L1 quadrature with the order alpha(x_i, t_n) at each grid point and step; implicit centered diffusion."""
import numpy as np
from scipy.linalg import solve_banded
from scipy.special import gamma

NX = 32
NT = 48


def solve(problem):
    x = np.linspace(0.0, problem.length, NX + 1)
    t = np.linspace(0.0, problem.final_time, NT + 1)
    d = problem.diffusivity / (x[1] - x[0]) ** 2
    u = np.zeros((NT + 1, NX + 1))
    u[0] = problem.initial(x)
    for n in range(1, NT + 1):
        alpha = problem.alpha(x[1:-1], t[n])  # one order per interior point
        left = (t[n] - t[:n])[:, None]
        right = (t[n] - t[1:n + 1])[:, None]
        steps = np.diff(t[:n + 1])[:, None]
        weights = (left ** (1 - alpha) - right ** (1 - alpha)) / (gamma(2 - alpha) * steps)
        rhs = problem.forcing(x[1:-1], t[n]) + weights[-1] * u[n - 1, 1:-1]
        if n > 1:
            rhs = rhs - np.sum(weights[:-1] * np.diff(u[:n, 1:-1], axis=0), axis=0)
        ab = np.zeros((3, NX - 1))
        ab[0, 1:] = -d
        ab[1, :] = weights[-1] + 2 * d
        ab[2, :-1] = -d
        u[n, 1:-1] = solve_banded((1, 1), ab, rhs)
    return {"x": x, "t": t, "u": u}
