"""Classical solver: backward Euler in time, centered second differences in space."""
import numpy as np
from scipy.linalg import solve_banded

NX = 32
NT = 48


def solve(problem):
    x = np.linspace(0.0, problem.length, NX + 1)
    dx = x[1] - x[0]
    t = np.linspace(0.0, problem.final_time, NT + 1)
    d = problem.diffusivity / dx**2
    u = np.zeros((NT + 1, NX + 1))
    u[0] = problem.initial(x)
    u[0, [0, -1]] = 0.0
    for n in range(1, NT + 1):
        dt = t[n] - t[n - 1]
        ab = np.zeros((3, NX - 1))
        ab[0, 1:] = -d
        ab[1, :] = 1 / dt + 2 * d
        ab[2, :-1] = -d
        rhs = u[n - 1, 1:-1] / dt + problem.forcing(x[1:-1], t[n])
        u[n, 1:-1] = solve_banded((1, 1), ab, rhs)
    return {"x": x, "t": t, "u": u}
