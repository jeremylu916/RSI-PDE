"""Evaluator-only analytic manufactured references, separate from numerical solvers."""

import numpy as np

from .tasks import evaluation_points


def reference(task, x, t):
    xi = np.asarray(x)/task.length
    if task.profile == "polynomial":
        spatial = task.amplitude*4*xi*(1-xi)
    elif task.profile == "mixed_sine":
        spatial = task.amplitude*(np.sin(np.pi*task.mode*xi)+0.25*np.sin(3*np.pi*task.mode*xi))
    else:
        spatial = task.amplitude*np.sin(np.pi*task.mode*xi)
    temporal = np.ones_like(np.asarray(t)) if task.family == "poisson" else 1 + np.asarray(t)**task.beta
    return temporal[:, None]*spatial[None, :]


def output_nodes(task, grid="fine"):
    space, time = evaluation_points(task, grid)
    x = np.linspace(0, task.length, space)
    t = np.array([0.0]) if task.family == "poisson" else np.linspace(0, task.final_time, time)
    return x, t


def reference2d(task, x, y, t):
    """Exact wave solution A*s(x)*s(y)*(cos(omega*t) + t**beta), shape (len(t), len(y), len(x))."""
    import math

    def profile(z):
        xi = np.asarray(z)/task.length
        if task.profile == "polynomial":
            return 4*xi*(1-xi)
        if task.profile == "mixed_sine":
            return np.sin(np.pi*task.mode*xi)+0.25*np.sin(3*np.pi*task.mode*xi)
        return np.sin(np.pi*task.mode*xi)
    omega = math.sqrt(task.diffusivity)*math.pi*task.mode*math.sqrt(2)/task.length
    t = np.asarray(t, dtype=float)
    temporal = np.cos(omega*t)+t**task.beta
    return task.amplitude*temporal[:, None, None]*profile(y)[None, :, None]*profile(x)[None, None, :]


def output_nodes2d(task, grid="fine"):
    space, time = evaluation_points(task, grid)
    x = np.linspace(0, task.length, space)
    t = np.linspace(0, task.final_time, time)
    return x, x.copy(), t


def reference_group(task, x, t):
    """Exact group solution A*t**p*sin(k x), shape (len(t), len(x)); the FODE uses x = [0] and A*t**p."""
    t = np.asarray(t, dtype=float)
    temporal = task.amplitude*t**task.beta
    if task.family == "fode":
        return temporal[:, None]*np.ones((1, len(np.atleast_1d(x))))
    return temporal[:, None]*np.sin(task.mode*np.pi*np.asarray(x, dtype=float)/task.length)[None, :]


def output_nodes_group(task, grid="fine"):
    space, time = evaluation_points(task, grid)
    x = np.array([0.0]) if task.family == "fode" else np.linspace(0, task.length, space)
    return x, np.linspace(0, task.final_time, time)
