"""Classical solver: implicit diffusion, explicit centered transport, CFL-limited uniform steps."""
import math
import numpy as np
from scipy.linalg import solve_banded

NX = 32
NT = 48


def solve(problem):
    x = np.linspace(0.0, problem.length, NX + 1)
    dx = x[1] - x[0]
    u0 = problem.initial(x)
    # Transport speed is estimated from the initial state with a growth safety factor.
    speed = 1.5 * float(np.max(np.abs(u0)))
    nt = max(NT, math.ceil(problem.final_time * speed / (0.35 * dx)))
    t = np.linspace(0.0, problem.final_time, nt + 1)
    d = problem.diffusivity / dx**2
    u = np.zeros((nt + 1, NX + 1))
    u[0] = u0
    u[0, [0, -1]] = 0.0
    for n in range(1, nt + 1):
        dt = t[n] - t[n - 1]
        previous = u[n - 1]
        gradient = (previous[2:] - previous[:-2]) / (2 * dx)
        ab = np.zeros((3, NX - 1))
        ab[0, 1:] = -d
        ab[1, :] = 1 / dt + 2 * d
        ab[2, :-1] = -d
        rhs = previous[1:-1] / dt + problem.forcing(x[1:-1], t[n]) - previous[1:-1] * gradient
        u[n, 1:-1] = solve_banded((1, 1), ab, rhs)
    return {"x": x, "t": t, "u": u}
