"""Public forcing, initial data, and order functions for the group FPDE examples.

No evaluator reference imports. Caputo derivatives start at t=0. A variable order is evaluated at the
current point: alpha(t) = c*sin(t)**2 or alpha(x, t) = c*sin(x)*t**2, with c = orders[0].
"""

import math
import numpy as np
from scipy.special import gamma


def order(task, t, x=None):
    """Order of the Caputo derivative at time t (and position x for the space-time variable order)."""
    c = task.orders[0]
    if task.family == "fpde_variable_order":
        return c*math.sin(t)**2
    if task.family == "fpde_variable_order_xt":
        return c*np.sin(np.asarray(x, dtype=float))*t*t
    return c


def forcing_group(task, x, t):
    t = float(t)
    if t < 0:
        raise ValueError("Group forcing is defined for t >= 0")
    p = task.beta
    alpha = order(task, t, x)
    temporal = gamma(p+1)/gamma(p+1-np.asarray(alpha, dtype=float))*t**(p-np.asarray(alpha, dtype=float))
    if task.family == "fode":
        return np.full(np.shape(x), task.amplitude*float(temporal), dtype=float)
    k = task.mode*np.pi/task.length
    return task.amplitude*(temporal+task.diffusivity*k*k*t**p)*np.sin(k*np.asarray(x, dtype=float))


def initial_group(task, x):
    return np.zeros(np.shape(x))
