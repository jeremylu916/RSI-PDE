"""Evaluator-only reference solvers for the pdebench-ext cases without a closed form (12b, 14b).

Both are periodic and solved pseudospectrally with ETDRK4 (Cox-Matthews; coefficients by the contour integrals of
Kassam and Trefethen 2005). The solution is evaluated on the scoring grid by exact Fourier interpolation, so the
only reference errors are the spatial and temporal truncation, which scripts/build_pdebench_ext_references.py
estimates by a second solve at lower resolution.
"""

import numpy as np


def etdrk4_coefficients(L, h, contour_points=64):
    r = np.exp(1j*np.pi*(np.arange(1, contour_points+1)-0.5)/contour_points)
    LR = h*L[..., None]+r
    E, E2 = np.exp(h*L), np.exp(h*L/2)
    Q = h*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=-1))
    f1 = h*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=-1))
    f2 = h*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=-1))
    f3 = h*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=-1))
    return E, E2, Q, f1, f2, f3


def etdrk4(v, L, nonlinear, h, steps, coefficients=None):
    E, E2, Q, f1, f2, f3 = coefficients or etdrk4_coefficients(L, h)
    for _ in range(steps):
        Nv = nonlinear(v)
        a = E2*v+Q*Nv
        Na = nonlinear(a)
        b = E2*v+Q*Na
        Nb = nonlinear(b)
        c = E2*a+Q*(2*Nb-Nv)
        Nc = nonlinear(c)
        v = E*v+Nv*f1+2*(Na+Nb)*f2+Nc*f3
    return v


def fourier_matrix(n, length, points):
    """Rows evaluate the length-periodic trigonometric interpolant of n samples at the given points."""
    k = np.fft.fftfreq(n, d=1.0/n)*2*np.pi/length
    if n % 2 == 0:
        k[n//2] = 0.0  # drop the Nyquist mode, which an interpolant evaluated off the grid cannot represent
    return np.exp(1j*np.outer(points, k))/n


def kuramoto_sivashinsky(spec, n, times, x_eval, substeps):
    """u(t, x_eval) for u_t + u u_x + u_xx + u_xxxx = 0 on (0, L), periodic; times equally spaced from 0."""
    from .equations_ext import ks_initial_12b
    length = spec["b"]
    x = length*np.arange(n)/n
    k = np.fft.fftfreq(n, d=1.0/n)*2*np.pi/length
    L = k**2-k**4
    dealias = np.abs(np.fft.fftfreq(n, d=1.0/n)) < n/3
    g = -0.5j*k

    def nonlinear(v):
        return g*np.fft.fft(np.real(np.fft.ifft(v))**2)*dealias
    v = np.fft.fft(ks_initial_12b(spec, x))
    h = (times[1]-times[0])/substeps
    coefficients = etdrk4_coefficients(L, h)
    matrix = fourier_matrix(n, length, x_eval)
    out = [np.real(matrix@v)]
    for _ in range(len(times)-1):
        v = etdrk4(v, L, nonlinear, h, substeps, coefficients)
        out.append(np.real(matrix@v))
    return np.array(out)


def cahn_hilliard(spec, n, times, x_eval, substeps):
    """u(t, y_eval, x_eval) for u_t = M Lap(u^3 - u - eps^2 Lap u) on (0, L)^2, periodic."""
    from .equations_ext import ch_initial_14b
    length, M, eps = spec["b"], spec["M"], spec["eps"]
    x = length*np.arange(n)/n
    X, Y = np.meshgrid(x, x)
    k = np.fft.fftfreq(n, d=1.0/n)*2*np.pi/length
    KX, KY = np.meshgrid(k, k)
    k2 = KX**2+KY**2
    L = M*k2-M*eps*eps*k2**2
    m = np.abs(np.fft.fftfreq(n, d=1.0/n)) < n/3
    dealias = m[None, :] & m[:, None]

    def nonlinear(v):
        u = np.real(np.fft.ifft2(v))
        return -M*k2*np.fft.fft2(u**3)*dealias
    v = np.fft.fft2(ch_initial_14b(spec, X, Y))
    h = (times[1]-times[0])/substeps
    coefficients = etdrk4_coefficients(L, h)
    matrix = fourier_matrix(n, length, x_eval)

    def evaluate(v):
        return np.real(matrix@v@matrix.T)
    out = [evaluate(v)]
    for _ in range(len(times)-1):
        v = etdrk4(v, L, nonlinear, h, substeps, coefficients)
        out.append(evaluate(v))
    return np.array(out)
