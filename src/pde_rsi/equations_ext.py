"""Forcing, initial/boundary data, and closed-form solutions for pde_benchmarks.tex Problems 12-16.

Problem 12: Kuramoto-Sivashinsky (12a manufactured, 12b chaotic; 12b has no closed form).
Problem 13: 2-D incompressible Navier-Stokes (13a Taylor-Green vortex, 13b Kovasznay flow).
Problem 14: Cahn-Hilliard (14a manufactured, 14b spinodal decomposition; 14b has no closed form).
Problem 15: Maxwell's equations, 2-D TE_z time domain (15a resonant cavity mode, 15b current-driven).
Problem 16: time-harmonic electromagnetic (curl-curl) equation (16a low frequency, 16b high frequency).

A case is a flat dict `spec` (see pdebench_ext.py). Every field function returns an array whose last axis lists the
components of the case (see COMPONENTS). This module is bind-mounted read-only into the sandbox, where solver
programs see only black-box closures over forcing, initial, and boundary data. It must not import pde_rsi.
"""

import math

import numpy as np

PI = math.pi
COMPONENTS = {"12": ("u",), "13": ("u", "v", "p"), "14": ("u",), "15": ("Ex", "Ey", "Hz"), "16": ("Ex", "Ey")}
FORCING_COMPONENTS = {"12": ("f",), "13": ("fx", "fy"), "14": ("f",), "15": ("Jx", "Jy"), "16": ("fx", "fy")}


def problem_number(spec):
    return spec["problem"].rstrip("ab")


def tpow(t, e):
    t, e = np.asarray(t, dtype=float), np.asarray(e, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(e == 0, 1.0, np.power(t, e))


def _stack(*arrays):
    arrays = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in arrays])
    return np.stack(arrays, axis=-1)


# ---------------------------------------------------------------- Problem 12: Kuramoto-Sivashinsky
def _ks_manufactured(spec, x, t):
    A, k, beta = spec["A"], 2*PI/spec["b"], spec["beta"]
    g = 1+tpow(t, beta)
    u = A*np.sin(k*x)*g
    f = A*(beta*tpow(t, beta-1)+(k**4-k**2)*g)*np.sin(k*x)+A*A*k/2*g*g*np.sin(2*k*x)
    return u, f


def ks_initial_12b(spec, x):
    """Kassam-Trefethen data cos(z/16)(1 + sin(z/16)), z = x - shift, scaled by amp (defaults: 1 and 0)."""
    z = np.asarray(x, dtype=float)-spec.get("ic_shift", 0.0)
    return spec.get("ic_amp", 1.0)*np.cos(z/16)*(1+np.sin(z/16))


# ---------------------------------------------------------------- Problem 13: Navier-Stokes
def kovasznay_lambda(spec):
    re = 1/spec["nu"]
    return re/2-math.sqrt(re*re/4+4*PI*PI)


def _navier_stokes(spec, x, y, t):
    if spec["problem"] == "13a":  # Taylor-Green vortex with wavenumber k
        k, nu = spec["k"], spec["nu"]
        F = np.exp(-2*nu*k*k*t)
        return (np.cos(k*x)*np.sin(k*y)*F, -np.sin(k*x)*np.cos(k*y)*F,
                -0.25*(np.cos(2*k*x)+np.cos(2*k*y))*F*F)
    lam = kovasznay_lambda(spec)  # 13b, steady
    e = np.exp(lam*x)
    return 1-e*np.cos(2*PI*y), lam/(2*PI)*e*np.sin(2*PI*y), 0.5*(1-np.exp(2*lam*x))+0*y


# ---------------------------------------------------------------- Problem 14: Cahn-Hilliard
def _cahn_hilliard_manufactured(spec, x, y, t):
    """u = A cos(kx) cos(ky) g(t), k = 2 pi / L, for u_t = M Lap(u^3 - u - eps^2 Lap u) + f."""
    A, k, beta, M, eps = spec["A"], 2*PI/spec["b"], spec["beta"], spec["M"], spec["eps"]
    g = 1+tpow(t, beta)
    c = np.cos(k*x)*np.cos(k*y)
    u = A*c*g
    grad2 = (A*g*k)**2*(np.sin(k*x)**2*np.cos(k*y)**2+np.cos(k*x)**2*np.sin(k*y)**2)
    lap_u = -2*k*k*u
    lap_mu = 3*u*u*lap_u+6*u*grad2-lap_u-eps*eps*4*k**4*u
    f = A*c*beta*tpow(t, beta-1)-M*lap_mu
    return u, f


def ch_initial_14b(spec, x, y):
    """Smooth deterministic perturbation of the mixed state: a sum of a few Fourier modes."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    total = 0*x+0*y
    for amplitude, kx, ky, phase in spec["modes"]:
        total = total+amplitude*np.cos(kx*x+ky*y+phase)
    return total


# ---------------------------------------------------------------- Problem 15: Maxwell TE_z
def _maxwell(spec, x, y, t):
    """(Ex, Ey, Hz) and the current (Jx, Jy) for E_t = c curl H - J, H_t = -c curl E on (0, a) x (0, b)."""
    a, b, c = spec["b"], spec["d"], spec["c"]
    if spec["problem"] == "15a":  # resonant cavity mode (m, n); J = 0
        m, n = spec["m"], spec["n"]
        omega = c*PI*math.sqrt((m/a)**2+(n/b)**2)
        ex = -(c*n*PI/(b*omega))*np.cos(m*PI*x/a)*np.sin(n*PI*y/b)*np.sin(omega*t)
        ey = (c*m*PI/(a*omega))*np.sin(m*PI*x/a)*np.cos(n*PI*y/b)*np.sin(omega*t)
        hz = np.cos(m*PI*x/a)*np.cos(n*PI*y/b)*np.cos(omega*t)
        zero = 0*x+0*y+0*t
        return (ex, ey, hz), (zero, zero)
    A, beta = spec["A"], spec["beta"]  # 15b: E = A t^beta (cos(pi x/a) sin(pi y/b), -sin(pi x/a) cos(pi y/b))
    s, ds = tpow(t, beta), beta*tpow(t, beta-1)
    big_s = tpow(t, beta+1)/(beta+1)
    cx, sx, cy, sy = np.cos(PI*x/a), np.sin(PI*x/a), np.cos(PI*y/b), np.sin(PI*y/b)
    ex, ey = A*s*cx*sy, -A*s*sx*cy
    amp = c*A*PI*(1/a+1/b)
    hz = amp*big_s*cx*cy
    hz_x, hz_y = -amp*big_s*(PI/a)*sx*cy, -amp*big_s*(PI/b)*cx*sy
    jx = c*hz_y-A*ds*cx*sy
    jy = -c*hz_x+A*ds*sx*cy
    return (ex, ey, hz), (jx, jy)


# ---------------------------------------------------------------- Problem 16: time-harmonic electromagnetic
def _curl_curl(spec, x, y):
    """E = (A cos(p pi x) sin(n pi y), B sin(m pi x) cos(q pi y)) on (0, 1)^2 and f = curl curl E - k^2 E."""
    A, B, p, n, m, q, k = (spec[key] for key in ("A", "B", "p", "n", "m", "q", "k"))
    ex = A*np.cos(p*PI*x)*np.sin(n*PI*y)
    ey = B*np.sin(m*PI*x)*np.cos(q*PI*y)
    ccx = -B*m*q*PI*PI*np.cos(m*PI*x)*np.sin(q*PI*y)+A*n*n*PI*PI*np.cos(p*PI*x)*np.sin(n*PI*y)
    ccy = B*m*m*PI*PI*np.sin(m*PI*x)*np.cos(q*PI*y)-A*n*p*PI*PI*np.sin(p*PI*x)*np.cos(n*PI*y)
    return (ex, ey), (ccx-k*k*ex, ccy-k*k*ey)


# ---------------------------------------------------------------- dispatch
def has_closed_form(spec):
    return spec["problem"] not in ("12b", "14b")


def exact(spec, x, t, y=None):
    """Closed-form solution at points (x, t[, y]); last axis = components. Not defined for 12b and 14b."""
    p = spec["problem"]
    x, t = np.asarray(x, dtype=float), np.asarray(t, dtype=float)
    y = None if y is None else np.asarray(y, dtype=float)
    if p == "12a":
        return _stack(_ks_manufactured(spec, x, t)[0])
    if p in ("13a", "13b"):
        return _stack(*_navier_stokes(spec, x, y, t))
    if p == "14a":
        return _stack(_cahn_hilliard_manufactured(spec, x, y, t)[0])
    if p in ("15a", "15b"):
        return _stack(*_maxwell(spec, x, y, t)[0])
    if p in ("16a", "16b"):
        return _stack(*_curl_curl(spec, x, y)[0])
    raise ValueError(f"No closed form for problem {p!r}")


def forcing(spec, x, t, y=None):
    """Source term at points; last axis = forcing components (FORCING_COMPONENTS)."""
    p = spec["problem"]
    x, t = np.asarray(x, dtype=float), np.asarray(t, dtype=float)
    y = None if y is None else np.asarray(y, dtype=float)
    zero = 0*x+0*t+(0 if y is None else 0*y)
    if p == "12a":
        return _stack(_ks_manufactured(spec, x, t)[1])
    if p in ("12b", "14b"):
        return _stack(zero)
    if p in ("13a", "13b"):
        return _stack(zero, zero)
    if p == "14a":
        return _stack(_cahn_hilliard_manufactured(spec, x, y, t)[1])
    if p in ("15a", "15b"):
        return _stack(*_maxwell(spec, x, y, t)[1])
    if p in ("16a", "16b"):
        return _stack(*_curl_curl(spec, x, y)[1])
    raise ValueError(f"Unknown problem {p!r}")


def initial(spec, x, y=None):
    """Fields at t = 0 (time-dependent cases); last axis = components."""
    p = spec["problem"]
    if p == "12b":
        return _stack(ks_initial_12b(spec, x))
    if p == "14b":
        return _stack(ch_initial_14b(spec, x, y))
    return exact(spec, x, 0.0, y)


def boundary(spec, x, y):
    """Dirichlet velocity (u, v) on the boundary of the Kovasznay domain (Case 13b)."""
    u, v, _ = _navier_stokes(spec, np.asarray(x, dtype=float), np.asarray(y, dtype=float), 0.0)
    return _stack(u, v)
