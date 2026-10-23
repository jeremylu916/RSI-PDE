"""Classical solver: second-order centered finite differences for -D u_xx = f, u(0)=u(L)=0."""
import numpy as np
from scipy.linalg import solve_banded

NX = 32


def solve(problem):
    x = np.linspace(0.0, problem.length, NX + 1)
    dx = x[1] - x[0]
    d = problem.diffusivity / dx**2
    ab = np.zeros((3, NX - 1))
    ab[0, 1:] = -d
    ab[1, :] = 2 * d
    ab[2, :-1] = -d
    u = np.zeros((1, NX + 1))
    u[0, 1:-1] = solve_banded((1, 1), ab, problem.forcing(x[1:-1], 0.0))
    return {"x": x, "t": np.array([0.0]), "u": u}
