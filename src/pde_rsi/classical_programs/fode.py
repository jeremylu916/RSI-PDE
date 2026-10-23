"""Classical solver: L1 scheme on a uniform time grid for Caputo(alpha, u)(t) = f(t), u(0) = 0."""
import math
import numpy as np

NT = 48


def solve(problem):
    alpha = problem.orders[0]
    t = np.linspace(0.0, problem.final_time, NT + 1)
    u = np.zeros(NT + 1)
    u[0] = problem.initial()
    for n in range(1, NT + 1):
        left = t[n] - t[:n]
        right = t[n] - t[1:n + 1]
        weights = (left ** (1 - alpha) - right ** (1 - alpha)) / (math.gamma(2 - alpha) * np.diff(t[:n + 1]))
        history = weights[:-1] @ np.diff(u[:n]) if n > 1 else 0.0
        u[n] = u[n - 1] + (problem.forcing(t[n]) - history) / weights[-1]
    return {"t": t, "u": u}
