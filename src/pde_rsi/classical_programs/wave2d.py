"""Classical solver: explicit leapfrog with a 5-point Laplacian for the 2D wave equation."""
import math
import numpy as np

NX = 32
NT = 48


def solve(problem):
    x = np.linspace(0.0, problem.length, NX + 1)
    y = x.copy()
    dx = x[1] - x[0]
    xx, yy = np.meshgrid(x, y)
    speed = math.sqrt(problem.diffusivity)
    nt = max(NT, math.ceil(problem.final_time * speed / (0.5 * dx)))
    t = np.linspace(0.0, problem.final_time, nt + 1)
    dt = t[1] - t[0]
    inner = (slice(1, -1), slice(1, -1))

    def laplacian(v):
        return (v[1:-1, 2:] + v[1:-1, :-2] + v[2:, 1:-1] + v[:-2, 1:-1] - 4 * v[1:-1, 1:-1]) / dx**2

    u = np.zeros((nt + 1, NX + 1, NX + 1))
    u[0] = problem.initial(xx, yy)
    u[0][[0, -1], :] = 0.0
    u[0][:, [0, -1]] = 0.0
    # Taylor first step; the source is sampled at t=dt because it may be singular at t=0.
    u[1][inner] = (u[0][inner] + dt * problem.initial_velocity(xx, yy)[inner]
                   + 0.5 * dt * dt * (problem.diffusivity * laplacian(u[0]) + problem.forcing(xx, yy, dt)[inner]))
    for n in range(1, nt):
        u[n + 1][inner] = (2 * u[n][inner] - u[n - 1][inner]
                           + dt * dt * (problem.diffusivity * laplacian(u[n]) + problem.forcing(xx, yy, t[n])[inner]))
    return {"x": x, "y": y, "t": t, "u": u}
