"""Input fields, coefficients, and forcing of the PDE Data Inventory suite (PDE_Data_Inventory_Overleaf.pdf).

A case is a dict `spec` (see inventory.py). This module is imported by the host (references, evaluator) and
bind-mounted read-only into the sandbox, where solver programs see only black-box closures built from it, so
both sides use exactly the same continuous definition of every input. It must not import pde_rsi.

Continuous definitions of gridded inputs (the inputs of DiffusionPDE/FNO datasets are gridded):
- periodic grid data (Burgers u0, Navier-Stokes w0): trigonometric interpolation of the samples;
- cell-centred data (Darcy permeability): piecewise constant on the cells;
- node data with boundary nodes (Poisson / Helmholtz forcing): bicubic spline interpolation;
- the 4-D coefficient: bilinear interpolation of a-hat on its 65 x 65 (q, z) grid, a(x) = a-hat(|x'|^2, z).
"""

import math

import numpy as np
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator

TWO_PI = 2*math.pi


# ---------------------------------------------------------------- gridded inputs
def _trig_coefficients(values):
    """FFT coefficients of periodic samples with the Nyquist mode split symmetrically (real interpolant)."""
    values = np.asarray(values, dtype=float)
    c = np.fft.fftn(values)/values.size
    return c


def _trig_matrix(points, n, length):
    """exp(i k x) for the n symmetric frequencies of a length-periodic grid (Nyquist halved and mirrored)."""
    k = np.fft.fftfreq(n, d=1.0/n)
    matrix = np.exp(1j*TWO_PI/length*np.outer(np.ravel(points), k))
    if n % 2 == 0:  # split the Nyquist mode cos(pi n x / L) evenly between +n/2 and -n/2
        matrix[:, n//2] = np.cos(math.pi*n/length*np.ravel(points))
    return matrix


def periodic_1d(values, length, x, origin=0.0):
    """Trigonometric interpolant of periodic samples values[j] = u(origin + j length / n)."""
    values = np.asarray(values, dtype=float)
    x = np.asarray(x, dtype=float)
    c = _trig_coefficients(values)
    return np.real(_trig_matrix(x-origin, len(values), length)@c).reshape(x.shape)


def periodic_2d(values, length, x, y):
    """Trigonometric interpolant of values[i, j] = w(x_j, y_i) on the periodic [0, length)^2 grid.
    Meshgrid inputs (rows of x equal, columns of y equal) are evaluated separably; others pointwise."""
    values = np.asarray(values, dtype=float)
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    ny, nx = values.shape
    c = _trig_coefficients(values)
    if x.ndim == 2 and np.allclose(x, x[:1]) and np.allclose(y, y[:, :1]):
        ex, ey = _trig_matrix(x[0], nx, length), _trig_matrix(y[:, 0], ny, length)
        return np.real(ey@c@ex.T)
    flat_x, flat_y = x.ravel(), y.ravel()
    out = np.empty(flat_x.size)
    for start in range(0, flat_x.size, 2048):
        ex = _trig_matrix(flat_x[start:start+2048], nx, length)
        ey = _trig_matrix(flat_y[start:start+2048], ny, length)
        out[start:start+2048] = np.real(np.einsum("pi,ij,pj->p", ey, c, ex))
    return out.reshape(x.shape)


def cell_constant(values, x, y):
    """Piecewise-constant field on a K x K cell grid of [0, 1]^2: values[i, j] on cell (row i = y, col j = x)."""
    values = np.asarray(values, dtype=float)
    k = values.shape[0]
    i = np.clip(np.floor(np.asarray(y, dtype=float)*k).astype(int), 0, k-1)
    j = np.clip(np.floor(np.asarray(x, dtype=float)*k).astype(int), 0, k-1)
    return values[i, j]


def node_spline(values, x, y):
    """Bicubic spline through node values on the uniform S x S grid of [0, 1]^2 (row i = y, col j = x)."""
    values = np.asarray(values, dtype=float)
    s = values.shape[0]
    nodes = np.linspace(0.0, 1.0, s)
    spline = RectBivariateSpline(nodes, nodes, values, kx=3, ky=3)
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return spline.ev(np.clip(y, 0, 1), np.clip(x, 0, 1)).reshape(x.shape)


# ---------------------------------------------------------------- random-input fields (inventory sections 2, 7, 8, 9)
def wave_vectors(count=28):
    """The lowest nonzero integer wave vectors in the half-plane kx > 0 or (kx = 0, ky > 0), by |k|^2 then lexicographically."""
    vectors = [(kx, ky) for kx in range(0, 8) for ky in range(-8, 9)
               if (kx > 0 or ky > 0) and (kx, ky) != (0, 0)]
    vectors.sort(key=lambda k: (k[0]**2+k[1]**2, k[0], k[1]))
    return vectors[:count]


def allen_cahn_kappa(xi, x, y):
    """Section 7: kappa = 0.005 exp(0.6 g), g = Z^-1 sum_j a_j [xi_{2j-1} cos(k_j.x) + xi_{2j} sin(k_j.x)]."""
    k = wave_vectors(28)
    a = np.array([(1+kx*kx+ky*ky)**-2.0 for kx, ky in k])
    g = 0.0
    for j, (kx, ky) in enumerate(k):
        phase = kx*np.asarray(x)+ky*np.asarray(y)
        g = g+a[j]*(xi[2*j]*np.cos(phase)+xi[2*j+1]*np.sin(phase))
    return 0.005*np.exp(0.6*g/a.sum())


def adr_kappa(xi, m, x, y):
    """Section 8: kappa_m = 0.018 exp(0.6 g_m) with the variance-fixing weights omega_j^(m) = s_m b_j."""
    if m == 0:
        return 0.018+0*np.asarray(x)*np.asarray(y)
    k = wave_vectors(28)
    a = np.array([(1+kx*kx+ky*ky)**-2.0 for kx, ky in k])
    abar = a/a.sum()
    b = np.array([(1+kx*kx+ky*ky)**-0.75 for kx, ky in k])
    s = math.sqrt((abar**2).sum()/(b[:m]**2).sum())
    g = 0.0
    for j in range(m):
        kx, ky = k[j]
        phase = kx*np.asarray(x)+ky*np.asarray(y)
        g = g+s*b[j]*(xi[2*j]*np.cos(phase)+xi[2*j+1]*np.sin(phase))
    return 0.018*np.exp(0.6*g)


BURGERS2D_MODES = (lambda x, y: np.sin(x), lambda x, y: np.sin(y), lambda x, y: np.cos(x+y), lambda x, y: np.cos(x-y),
                   lambda x, y: np.sin(2*x+y), lambda x, y: np.sin(x+2*y), lambda x, y: np.cos(2*x-y),
                   lambda x, y: np.cos(x-2*y))
# Normalization of phi_j (inventory section 2 leaves the constant to the implementation): unit L2 norm on the
# 2pi torus, i.e. phi_j = mode_j / (pi sqrt 2), since each mode has L2 norm pi sqrt 2 on [0, 2pi]^2.
BURGERS2D_NORM = math.pi*math.sqrt(2.0)
TAU = (-0.4, -0.2, 0.0, 0.2, -0.4, -0.2, 0.0, 0.2)


def ramp(z, tau):
    return min(1.0, max(0.0, (z-tau)/0.30))


def burgers2d_forcing(xi, x, y):
    """Section 2: f = 1.5 sum_j s_tau_j(xi_j) phi_j(x, y)."""
    return 1.5*sum(ramp(xi[j], TAU[j])*BURGERS2D_MODES[j](np.asarray(x), np.asarray(y))/BURGERS2D_NORM
                   for j in range(8))


def burgers2d_kappa(xi, x, y):
    """Section 2: kappa = 5e-3 [1 + 0.30/6 sum_j xi_{8+j} chi_j(x, y)]."""
    x, y = np.asarray(x), np.asarray(y)
    chi = (np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(x+y), np.sin(x-y))
    return 5e-3*(1+0.30/6*sum(xi[8+j]*chi[j] for j in range(6)))


def burgers2d_initial(xi, x, y):
    x, y = np.asarray(x), np.asarray(y)
    return (0.12*np.cos(x)+0.10*np.sin(y)+0.08*np.cos(x-y)+0.05*np.sin(2*x+y)+0.015*xi[16]*np.sin(2*x-y)
            + 0.015*xi[17]*np.cos(x+2*y)+0.012*xi[18]*np.sin(2*x+2*y)+0.012*xi[19]*np.cos(2*x-2*y))


def ac_initial(x, y):
    """Sections 7 and 8: u0 = 0.35 cos x + 0.30 sin y + 0.20 cos(x - y) + 0.10 sin(2x + y)."""
    x, y = np.asarray(x), np.asarray(y)
    return 0.35*np.cos(x)+0.30*np.sin(y)+0.20*np.cos(x-y)+0.10*np.sin(2*x+y)


def adr_f0(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return np.sin(x+y)+0.5*np.cos(2*x-y)


def elliptic4d_coefficient(ahat, x1, x2, x3, z):
    """Section 9: a(x) = bilinear interpolation of a-hat(q, z) on the 65 x 65 grid of [0, 1] x [-1, 1], q = |x'|^2."""
    ahat = np.asarray(ahat, dtype=float)
    q = np.clip(np.asarray(x1)**2+np.asarray(x2)**2+np.asarray(x3)**2, 0.0, 1.0)
    interpolator = RegularGridInterpolator((np.linspace(0, 1, ahat.shape[0]), np.linspace(-1, 1, ahat.shape[1])), ahat)
    q, z = np.broadcast_arrays(q, np.clip(np.asarray(z, dtype=float), -1, 1))
    return interpolator(np.stack([q.ravel(), z.ravel()], axis=-1)).reshape(q.shape)


def elliptic4d_source(x1, x2, x3, z):
    x0 = (0.35, 0.15, -0.10, 0.25)
    r2 = (np.asarray(x1)-x0[0])**2+(np.asarray(x2)-x0[1])**2+(np.asarray(x3)-x0[2])**2+(np.asarray(z)-x0[3])**2
    return np.exp(-25.0*r2)


# ---------------------------------------------------------------- per-case black boxes (spec -> callables)
def coefficient(spec, *coords):
    kind = spec["kind"]
    if kind == "darcy":
        return cell_constant(spec["fields"]["a"], *coords)
    if kind == "allen_cahn":
        return allen_cahn_kappa(spec["xi"], *coords)
    if kind == "adr":
        return adr_kappa(spec["xi"], spec["m"], *coords)
    if kind == "burgers2d":
        return burgers2d_kappa(spec["xi"], *coords)
    if kind == "elliptic4d":
        return elliptic4d_coefficient(spec["fields"]["ahat"], *coords)
    raise ValueError(f"{kind} has no coefficient field")


def forcing(spec, *coords):
    """Source term of the TRUE equation at the given coordinates (in the case's coordinate order)."""
    kind = spec["kind"]
    if kind in ("burgers_gauss", "burgers_random", "convdiff"):
        x, _ = coords
        return 0.0*np.asarray(x, dtype=float)
    if kind == "darcy":
        x, y = coords
        return np.ones(np.broadcast(np.asarray(x), np.asarray(y)).shape)
    if kind in ("poisson", "helmholtz"):
        return node_spline(spec["fields"]["f"], *coords)
    if kind == "ns":
        x, y, _ = coords
        return 0.1*(np.sin(TWO_PI*(np.asarray(x)+np.asarray(y)))+np.cos(TWO_PI*(np.asarray(x)+np.asarray(y))))
    if kind == "kolmogorov":
        x, y, _ = coords
        return -4.0*np.cos(4.0*np.asarray(y))+0.0*np.asarray(x)
    if kind == "allen_cahn":
        x, y, _ = coords
        return 0.0*np.asarray(x)*np.asarray(y)
    if kind == "burgers2d":
        x, y, _ = coords
        return burgers2d_forcing(spec["xi"], x, y)
    if kind == "adr":
        x, y, _ = coords
        return spec["gamma"]*adr_f0(x, y)
    if kind == "elliptic4d":
        return elliptic4d_source(*coords)
    raise ValueError(kind)


def initial(spec, *space):
    kind = spec["kind"]
    if kind == "burgers_gauss":
        # inventory 1.1: the Gaussian through its periodic representation on [a, b) (u(b) = u(a) exactly)
        (x,) = space
        a, length = spec["a"], spec["b"]-spec["a"]
        wrapped = a+np.mod(np.asarray(x, dtype=float)-a, length)
        center, width = spec.get("ic_center", 3.0), spec.get("ic_width", 1.0)  # inventory 1.1: center 3, width 1
        return np.exp(-(wrapped-center)**2/(2.0*width*width))
    if kind == "burgers_random":
        (x,) = space
        return periodic_1d(spec["fields"]["u0"], spec["b"]-spec["a"], x, spec["a"])
    if kind == "convdiff":
        (x,) = space
        return convdiff_exact(spec, np.asarray(x, dtype=float), 0.0)
    if kind in ("ns", "kolmogorov"):
        return periodic_2d(spec["fields"]["w0"], spec["b"]-spec["a"], *space)
    if kind in ("allen_cahn", "adr"):
        return ac_initial(*space)
    if kind == "burgers2d":
        return burgers2d_initial(spec["xi"], *space)
    raise ValueError(f"{kind} is steady")


def convdiff_exact(spec, x, t):
    """Section 10 with a Gaussian initial profile: the whole-line solution (Dirichlet data taken from it)."""
    s2 = spec["sigma0"]**2+2*spec["D"]*np.asarray(t, dtype=float)
    return spec["sigma0"]/np.sqrt(s2)*np.exp(-(x-spec["x0"]-spec["c"]*np.asarray(t, dtype=float))**2/(2*s2))


def boundary(spec, t):
    """Dirichlet values [u(a, t), u(b, t)] for the 1-D non-periodic case."""
    if spec["kind"] == "convdiff":
        return np.array([float(convdiff_exact(spec, spec["a"], t)), float(convdiff_exact(spec, spec["b"], t))])
    raise ValueError(f"{spec['kind']} has no Dirichlet boundary function")
