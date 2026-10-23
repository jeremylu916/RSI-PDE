"""Public forcing and initial conditions. No evaluator reference imports."""

import math
import numpy as np


def spatial_terms(task, x):
    k = task.mode*np.pi/task.length
    if task.profile == "polynomial":
        xi = x/task.length
        return 4*xi*(1-xi), 4*(1-2*xi)/task.length, np.full_like(x, -8/task.length**2)
    if task.profile == "mixed_sine":
        return (np.sin(k*x)+0.25*np.sin(3*k*x),
                k*np.cos(k*x)+0.75*k*np.cos(3*k*x),
                -k*k*np.sin(k*x)-2.25*k*k*np.sin(3*k*x))
    return np.sin(k*x), k*np.cos(k*x), -k*k*np.sin(k*x)


def initial(task, x):
    return task.amplitude * spatial_terms(task, x)[0]


def forcing(task, x, t):
    s, gradient, laplacian = spatial_terms(task, x)
    if task.family == "poisson":
        return -task.diffusivity * task.amplitude * laplacian
    # All time stepping evaluates forcing at positive time, including singular profiles.
    if t <= 0:
        raise ValueError("Evaluate temporal forcing at t>0")
    if task.orders:
        temporal_derivative = sum(w*math.gamma(task.beta+1)/math.gamma(task.beta+1-a)*t**(task.beta-a)
                                  for a, w in zip(task.orders, task.weights))
    else:
        temporal_derivative = task.beta*t**(task.beta-1)
    r = 1 + t**task.beta
    f = task.amplitude*(temporal_derivative*s-task.diffusivity*r*laplacian)
    if task.family == "convection_diffusion":
        f += task.velocity * task.amplitude*r*gradient
    if task.family == "burgers":
        f += task.amplitude**2*r*r*s*gradient
    return f
