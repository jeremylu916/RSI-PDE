"""Forcing, initial/boundary data, and exact solutions for the pde_benchmarks.tex suite.

A case is a flat dict `spec` (see pdebench.py). This module is imported by the host evaluator and
bind-mounted read-only into the sandbox, where solver programs see only black-box closures over
forcing, initial, and boundary; they cannot import or read this file. It must not import pde_rsi.

Conventions follow pde_benchmarks.tex: 0**0 = 1; Caputo derivatives start at t = 0, and a variable
order is evaluated at the current point (x, t).
"""

import math

import numpy as np
from scipy.special import gamma

PI = math.pi


def tpow(t, e):
    """t**e with the convention 0**0 = 1 (e may be an array)."""
    t, e = np.asarray(t, dtype=float), np.asarray(e, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(e == 0, 1.0, np.power(t, e))


def profile(spec, x):
    """Spatial profile s and its first two derivatives on (0, L) (training/validation/reserved)."""
    L = spec["b"]-spec["a"]
    z = (np.asarray(x, dtype=float)-spec["a"])/L
    if spec["profile"] == "sine":
        k = PI/L
        return np.sin(PI*z), k*np.cos(PI*z), -k*k*np.sin(PI*z)
    if spec["profile"] == "polynomial":
        return 4*z*(1-z), 4/L*(1-2*z), np.full_like(z, -8/L**2)
    if spec["profile"] == "mixed_sine":
        k = PI/L
        return (np.sin(PI*z)+0.25*np.sin(3*PI*z), k*np.cos(PI*z)+0.75*k*np.cos(3*PI*z),
                -k*k*np.sin(PI*z)-2.25*k*k*np.sin(3*PI*z))
    raise ValueError(f"Unknown profile {spec['profile']!r}")


def caputo_power(order, power, t):
    """Caputo derivative of t**power (power > 0) of the given (possibly array-valued) order."""
    order = np.asarray(order, dtype=float)
    return gamma(power+1)/gamma(power+1-order)*tpow(t, power-order)


def order(spec, x, t):
    """Fractional order alpha at (x, t) for the variable-order problems; the constant order otherwise."""
    if spec["problem"] == "10":
        return spec["c"]*np.sin(t)**2*np.ones_like(np.asarray(x, dtype=float))
    if spec["problem"] == "11":
        return spec["c"]*t*t*np.sin(np.asarray(x, dtype=float))
    return spec["alpha"]


def heat_mode_amplitude(spec, t):
    """a(t) of Case 2c: a' + lambda a = cos(gamma t), a(0) = 0."""
    lam, g = spec["D"]*PI**2/(spec["b"]-spec["a"])**2, spec["gamma"]
    return (lam*np.cos(g*t)+g*np.sin(g*t)-lam*np.exp(-lam*t))/(lam*lam+g*g)


def layer(spec, x):
    """(exp(Pe x/L) - 1)/(exp(Pe) - 1), evaluated stably for either sign of Pe."""
    L = spec["b"]-spec["a"]
    pe, z = spec["v"]*L/spec["D"], (np.asarray(x, dtype=float)-spec["a"])/L
    if pe > 0:
        return (np.exp(pe*(z-1))-math.exp(-pe))/(-math.expm1(-pe))
    return np.expm1(pe*z)/math.expm1(pe)


LIFTED = ("1a", "2a", "3a", "4a", "7", "8")


def lift(spec, x):
    """Affine Dirichlet lift b0 + (b1 - b0)(x - a)/L of the manufactured 1-D cases and its slope (0 by default)."""
    b0, b1 = spec.get("bc_left", 0.0), spec.get("bc_right", 0.0)
    slope = (b1-b0)/(spec["b"]-spec["a"])
    return b0+slope*(np.asarray(x, dtype=float)-spec["a"]), slope


HERMITE_Z = np.linspace(-10.0, 10.0, 4001)


def burgers_cole_hopf(spec, x, t):
    """Case 4b: u(x, t) by Cole-Hopf; trapezoid in z (eta = 2 sqrt(D t) z) with log-sum-exp scaling."""
    x = np.asarray(x, dtype=float)
    if t <= 0:
        return -np.sin(PI*x)
    D = spec["D"]
    y = x[:, None]-2*math.sqrt(D*t)*HERMITE_Z[None, :]
    exponent = -np.cos(PI*y)/(2*PI*D)-HERMITE_Z[None, :]**2
    weight = np.exp(exponent-exponent.max(axis=1, keepdims=True))
    return -np.sum(np.sin(PI*y)*weight, axis=1)/np.sum(weight, axis=1)


def exact(spec, x, t, y=None):
    """Exact solution at points (x, t[, y]); arrays broadcast. Steady cases ignore t; the FODE ignores x."""
    p, A = spec["problem"], spec.get("A")
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    if p == "1a":
        return A*profile(spec, x)[0]+lift(spec, x)[0]+0*t
    if p == "1b":
        return np.sin(2*PI*x)+0.1*np.sin(50*PI*x)+0*t
    if p == "1c":
        return np.arctan(spec["sharpness"]*(x-spec["x0"]))+0*t
    if p == "3b":
        return spec.get("U", 1.0)*layer(spec, x)+0*t
    if p == "3c":
        L = spec["b"]-spec["a"]
        return (x-spec["a"]-L*layer(spec, x))/spec["v"]+0*t
    if p in ("2a", "3a", "4a", "7", "8"):
        return A*profile(spec, x)[0]*(1+tpow(t, spec["beta"]))+lift(spec, x)[0]
    if p == "2b":
        L = spec["b"]-spec["a"]
        return np.exp(-spec["D"]*PI**2/L**2*t)*np.sin(PI*(x-spec["a"])/L)
    if p == "2c":
        return heat_mode_amplitude(spec, t)*np.sin(PI*(x-spec["a"])/(spec["b"]-spec["a"]))
    if p == "3d":
        sigma2 = spec["sigma0"]**2+2*spec["D"]*t
        return spec["sigma0"]/np.sqrt(sigma2)*np.exp(-(x-spec["x0"]-spec["v"]*t)**2/(2*sigma2))
    if p == "4b":
        x, t = np.broadcast_arrays(x, t)
        out = np.empty(x.shape)
        flat_x, flat_t, flat_out = x.reshape(-1), t.reshape(-1), out.reshape(-1)
        for value in np.unique(flat_t):
            mask = flat_t == value
            flat_out[mask] = burgers_cole_hopf(spec, flat_x[mask], float(value))
        return out
    if p == "4c":
        ul, ur, D = spec["ul"], spec["ur"], spec["D"]
        return (ul+ur)/2-(ul-ur)/2*np.tanh((ul-ur)*(x-(ul+ur)/2*t)/(4*D))
    if p == "5":
        omega = PI*math.sqrt(2*spec["D"])/(spec["b"]-spec["a"])
        return A*profile(spec, x)[0]*profile(spec, y)[0]*(np.cos(omega*t)+tpow(t, spec["beta"]))
    if p == "6":
        return tpow(t, spec["p"])+0*x
    if p in ("9", "10", "11"):
        return tpow(t, spec["p"])*np.sin(x)
    raise ValueError(f"Unknown problem {p!r}")


def forcing(spec, x, t, y=None):
    """Source term f at points (x, t[, y]) with t >= 0; the FODE ignores x and steady cases ignore t."""
    p, A = spec["problem"], spec.get("A")
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    zero = 0*x+0*t
    D = spec.get("D")
    if p == "1a":
        return -D*A*profile(spec, x)[2]+zero
    if p == "1b":
        return D*(4*PI**2*np.sin(2*PI*x)+250*PI**2*np.sin(50*PI*x))+zero
    if p == "1c":
        a, r = spec["sharpness"], x-spec["x0"]
        return 2*D*a**3*r/(1+a*a*r*r)**2+zero
    if p in ("2b", "3b", "3d", "4b", "4c"):
        return zero
    if p == "3c":
        return 1.0+zero
    if p == "2c":
        return np.sin(PI*(x-spec["a"])/(spec["b"]-spec["a"]))*np.cos(spec["gamma"]*t)
    if p in ("2a", "3a", "4a", "7", "8"):
        s, ds, d2s = profile(spec, x)
        beta, g = spec["beta"], 1+tpow(t, spec["beta"])
        if p == "7":
            rate = caputo_power(spec["alpha"], beta, t)
        elif p == "8":
            rate = spec["w1"]*caputo_power(spec["alpha1"], beta, t)+spec["w2"]*caputo_power(spec["alpha2"], beta, t)
        else:
            rate = beta*tpow(t, beta-1)
        value = A*(rate*s-D*g*d2s)
        ell, slope = lift(spec, x)
        if p == "3a":
            value = value+A*spec["v"]*g*ds+spec["v"]*slope
        if p == "4a":  # (w + ell)(w_x + ell') with w = A s g
            w, wx = A*s*g, A*ds*g
            value = value+w*wx+w*slope+ell*wx+ell*slope
        return value
    if p == "5":
        omega = PI*math.sqrt(2*D)/(spec["b"]-spec["a"])
        (sx, _, dxx), (sy, _, dyy) = profile(spec, x), profile(spec, y)
        beta = spec["beta"]
        acceleration = -omega**2*np.cos(omega*t)+beta*(beta-1)*tpow(t, beta-2)
        return A*acceleration*sx*sy-D*A*(np.cos(omega*t)+tpow(t, beta))*(dxx*sy+sx*dyy)
    if p == "6":
        return caputo_power(spec["alpha"], spec["p"], t)+zero
    if p in ("9", "10", "11"):
        return (caputo_power(order(spec, x, t), spec["p"], t)+tpow(t, spec["p"]))*np.sin(x)
    raise ValueError(f"Unknown problem {p!r}")


def initial(spec, x, y=None):
    """u at t = 0 (the FODE ignores x)."""
    if spec["problem"] == "5":
        return exact(spec, x, 0.0, y)
    return exact(spec, x, 0.0)


def initial_velocity(spec, x, y):
    """u_t at t = 0 for the wave problem (zero for every profile since beta >= 2)."""
    return np.zeros(np.broadcast(np.asarray(x, dtype=float), np.asarray(y, dtype=float)).shape)


def boundary(spec, t):
    """Dirichlet values [u(a, t), u(b, t)] for one-dimensional cases (taken from the exact solution)."""
    if spec["problem"] in LIFTED and (spec.get("bc_left", 0.0) or spec.get("bc_right", 0.0)):
        return np.array([spec.get("bc_left", 0.0), spec.get("bc_right", 0.0)], dtype=float)
    if spec["problem"] in ("1a", "1b", "2a", "2b", "2c", "3a", "3c", "4a", "4b", "7", "8", "9", "10", "11"):
        return np.zeros(2)  # homogeneous; 1b, 3c, 4b, and 9-11 vanish at both ends exactly
    return np.array([float(exact(spec, spec["a"], t)), float(exact(spec, spec["b"], t))])
