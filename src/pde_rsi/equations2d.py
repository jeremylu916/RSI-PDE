"""Public forcing and initial data for the 2D wave family. No evaluator reference imports.

u_tt - diffusivity*(u_xx + u_yy) = f on [0, L]^2 with zero Dirichlet data. Arrays x and y
broadcast against each other, for example meshgrid outputs.
"""

import math
import numpy as np


def _profile(task, z):
    k = task.mode*np.pi/task.length
    z = np.asarray(z, dtype=float)
    if task.profile == "polynomial":
        xi = z/task.length
        return 4*xi*(1-xi), np.full_like(z, -8/task.length**2)
    if task.profile == "mixed_sine":
        return np.sin(k*z)+0.25*np.sin(3*k*z), -k*k*np.sin(k*z)-2.25*k*k*np.sin(3*k*z)
    return np.sin(k*z), -k*k*np.sin(k*z)


def frequency(task):
    return math.sqrt(task.diffusivity)*math.pi*task.mode*math.sqrt(2)/task.length


def initial2d(task, x, y):
    return task.amplitude*_profile(task, x)[0]*_profile(task, y)[0]


def initial_velocity2d(task, x, y):
    # The temporal factor cos(omega t) + t**beta has zero slope at t=0 for beta > 1.
    return np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)))


def forcing2d(task, x, y, t):
    if t <= 0:
        raise ValueError("Evaluate wave forcing at t>0")
    omega = frequency(task)
    temporal = math.cos(omega*t)+t**task.beta
    acceleration = -omega*omega*math.cos(omega*t)+task.beta*(task.beta-1)*t**(task.beta-2)
    sx, dxx = _profile(task, x)
    sy, dyy = _profile(task, y)
    return task.amplitude*(acceleration*sx*sy-task.diffusivity*temporal*(dxx*sy+sx*dyy))
