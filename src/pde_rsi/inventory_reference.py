"""High-accuracy reference solvers for the PDE Data Inventory suite (host side only; never in the sandbox).

Each solver works from the same continuous inputs the solver programs see (equations_inventory.py) and returns
the solution on a requested evaluation grid. Accuracy is established by self-convergence (tests and
scripts/build_inventory_references.py), and pass tolerances are set well above the measured reference error.
"""

import math

import numpy as np
from scipy import sparse
from scipy.fft import dstn, idstn
from scipy.sparse.linalg import spsolve

from . import equations_inventory as eq


# ---------------------------------------------------------------- 1-D periodic viscous Burgers (sections 1.1, 1.2)
def burgers_periodic(spec, n, x_eval, t_eval, substeps=40):
    """ETDRK4 (Kassam-Trefethen) for u_t + (u^2/2)_x = nu u_xx on the periodic [a, b); returns u[t, x]."""
    a, b, nu = spec["a"], spec["b"], spec["D"]
    length = b-a
    x = a+length*np.arange(n)/n
    u = eq.initial(spec, x) if spec["kind"] == "burgers_gauss" else eq.periodic_1d(spec["fields"]["u0"], length, x, a)
    k = 2*math.pi/length*np.fft.fftfreq(n, d=1.0/n)
    dealias = np.abs(np.fft.fftfreq(n, d=1.0/n)) < n/3
    out = np.empty((len(t_eval), len(x_eval)))
    v = np.fft.fft(u)

    def nonlinear(v):
        w = np.real(np.fft.ifft(v))
        return -0.5j*k*np.fft.fft(w*w)*dealias

    def evaluate(v):
        coeff = v/n
        phase = np.exp(1j*np.outer(np.asarray(x_eval)-a, k))
        values = np.real(phase@coeff)
        return values

    out[0] = evaluate(v)
    for index in range(1, len(t_eval)):
        h = (t_eval[index]-t_eval[index-1])/substeps
        L = -nu*k*k
        E, E2 = np.exp(h*L), np.exp(h*L/2)
        roots = np.exp(1j*math.pi*(np.arange(1, 33)-0.5)/32)
        LR = h*L[:, None]+roots[None, :]
        Q = h*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=1))
        f1 = h*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1))
        f2 = h*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1))
        f3 = h*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1))
        for _ in range(substeps):
            Nv = nonlinear(v)
            a_ = E2*v+Q*Nv
            Na = nonlinear(a_)
            b_ = E2*v+Q*Na
            Nb = nonlinear(b_)
            c_ = E2*a_+Q*(2*Nb-Nv)
            Nc = nonlinear(c_)
            v = E*v+Nv*f1+2*(Na+Nb)*f2+Nc*f3
        out[index] = evaluate(v)
    return out


# ---------------------------------------------------------------- 2-D Darcy (section 3)
def darcy(spec, refine, x_eval, y_eval):
    """Cell-centred finite volumes on the 128*refine grid, harmonic face averaging, -div(a grad u) = 1, u = 0."""
    a_cells = np.asarray(spec["fields"]["a"], dtype=float)
    k = a_cells.shape[0]
    m = k*refine
    a = np.kron(a_cells, np.ones((refine, refine)))  # a[i, j]: row i = y, column j = x
    h = 1.0/m
    index = np.arange(m*m).reshape(m, m)
    rows, cols, vals = [], [], []
    diag = np.zeros((m, m))

    def couple(sl_a, sl_b, coef):
        nonlocal rows, cols, vals
        rows.append(index[sl_a].ravel())
        cols.append(index[sl_b].ravel())
        vals.append(-coef.ravel())
    # interior faces in x and y (harmonic means)
    fx = 2*a[:, :-1]*a[:, 1:]/(a[:, :-1]+a[:, 1:])/h**2
    fy = 2*a[:-1, :]*a[1:, :]/(a[:-1, :]+a[1:, :])/h**2
    couple((slice(None), slice(None, -1)), (slice(None), slice(1, None)), fx)
    couple((slice(None), slice(1, None)), (slice(None), slice(None, -1)), fx)
    couple((slice(None, -1), slice(None)), (slice(1, None), slice(None)), fy)
    couple((slice(1, None), slice(None)), (slice(None, -1), slice(None)), fy)
    diag[:, :-1] += fx
    diag[:, 1:] += fx
    diag[:-1, :] += fy
    diag[1:, :] += fy
    # Dirichlet faces at distance h/2
    diag[:, 0] += 2*a[:, 0]/h**2
    diag[:, -1] += 2*a[:, -1]/h**2
    diag[0, :] += 2*a[0, :]/h**2
    diag[-1, :] += 2*a[-1, :]/h**2
    rows.append(index.ravel())
    cols.append(index.ravel())
    vals.append(diag.ravel())
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(m*m, m*m))
    u = spsolve(A.tocsc(), np.ones(m*m)).reshape(m, m)
    centers = (np.arange(m)+0.5)*h
    padded = np.zeros((m+2, m+2))
    padded[1:-1, 1:-1] = u
    nodes = np.concatenate([[0.0], centers, [1.0]])
    from scipy.interpolate import RegularGridInterpolator
    interpolate = RegularGridInterpolator((nodes, nodes), padded)
    yy, xx = np.meshgrid(y_eval, x_eval, indexing="ij")
    return interpolate(np.stack([yy.ravel(), xx.ravel()], axis=-1)).reshape(yy.shape)


# ---------------------------------------------------------------- 2-D Poisson / Helmholtz (section 4)
def poisson_helmholtz(spec, m, x_eval, y_eval):
    """Delta u + k^2 u = f on (0,1)^2, u = 0: sine-series solve with f sampled on the m+1 node grid."""
    k2 = spec.get("k", 0.0)**2
    nodes = np.arange(1, m)/m
    xx, yy = np.meshgrid(nodes, nodes)
    f = eq.node_spline(spec["fields"]["f"], xx, yy)
    coef = dstn(f, type=1)
    p = np.arange(1, m)
    lam = -(math.pi*p[:, None])**2-(math.pi*p[None, :])**2+k2
    u_hat = coef/lam
    # evaluate the sine series exactly at the evaluation nodes (separable)
    bx = np.sin(math.pi*np.outer(x_eval, p))
    by = np.sin(math.pi*np.outer(y_eval, p))
    # dstn type 1 coefficients: f_ij = sum_pq c_pq sin(pi p x_i) sin(pi q y_j) with c = dst(f)/(2m)^2*4 ...
    c = u_hat/(2*m)**2*4  # normalization of DST-I: inverse is dst(.)/(2m) per axis
    return by@c@bx.T


# ---------------------------------------------------------------- periodic 2-D spectral machinery (sections 2, 5-8)
class Spectral2D:
    def __init__(self, n, length):
        self.n, self.length = n, length
        k = 2*math.pi/length*np.fft.fftfreq(n, d=1.0/n)
        self.kx, self.ky = np.meshgrid(k, k)  # [row = y, column = x]
        self.k2 = self.kx**2+self.ky**2
        mode = np.fft.fftfreq(n, d=1.0/n)
        mx, my = np.meshgrid(mode, mode)
        self.dealias = (np.abs(mx) < n/3) & (np.abs(my) < n/3)
        self.x = length*np.arange(n)/n
        self.X, self.Y = np.meshgrid(self.x, self.x)

    def to_grid(self, values_hat):
        return np.real(np.fft.ifft2(values_hat))

    def derivative(self, field_hat, axis):
        return np.real(np.fft.ifft2(1j*(self.kx if axis == "x" else self.ky)*field_hat))

    def sample(self, field, n_out):
        """Values at the n_out x n_out sub-grid (n must be a multiple of n_out)."""
        step = self.n//n_out
        return field[::step, ::step]


def _resample_periodic(values, n):
    """Trigonometric resampling of a periodic n0 x n0 grid onto n x n (zero padding or truncation)."""
    return eq.periodic_2d(values, 1.0, *np.meshgrid(np.arange(n)/n, np.arange(n)/n))


def vorticity(spec, n, frame_times, n_out, dt=1e-4):
    """w_t + v.grad w = nu Lap w - mu w + q, -Lap psi = w, v = (psi_y, -psi_x): integrating-factor RK4."""
    s = Spectral2D(n, spec["b"]-spec["a"])
    w = eq.initial(spec, s.X, s.Y)
    q_hat = np.fft.fft2(eq.forcing(spec, s.X, s.Y, 0.0))
    lin = -spec["D"]*s.k2-spec["mu"]
    k2 = s.k2.copy()
    k2[0, 0] = 1.0

    def rhs(w_hat):
        psi_hat = w_hat/k2
        psi_hat[0, 0] = 0
        vx = s.derivative(psi_hat, "y")
        vy = -s.derivative(psi_hat, "x")
        wx, wy = s.derivative(w_hat, "x"), s.derivative(w_hat, "y")
        return -np.fft.fft2(vx*wx+vy*wy)*s.dealias+q_hat
    return _integrate_if(np.fft.fft2(w), rhs, lin, frame_times, dt, s, n_out)


def _integrate_if(v, rhs, lin, frame_times, dt, s, n_out):
    frames = [s.sample(s.to_grid(v), n_out)]
    t = 0.0
    for target in frame_times[1:]:
        steps = max(1, int(round((target-t)/dt)))
        h = (target-t)/steps
        E, E2 = np.exp(lin*h), np.exp(lin*h/2)
        for _ in range(steps):
            k1 = rhs(v)
            k2_ = rhs(E2*(v+h/2*k1))
            k3 = rhs(E2*v+h/2*k2_)
            k4 = rhs(E*v+h*E2*k3)
            v = E*v+h/6*(E*k1+2*E2*(k2_+k3)+k4)
        t = target
        frames.append(s.sample(s.to_grid(v), n_out))
    return np.array(frames)


def scalar_periodic(spec, n, frame_times, n_out, dt=5e-4):
    """Allen-Cahn / 2-D Burgers / ADR: u_t = -cx u u_x - cy u u_y + div(kappa grad u) + lam (u - u^3) + f (RK4)."""
    s = Spectral2D(n, spec["b"]-spec["a"])
    kappa = eq.coefficient(spec, s.X, s.Y)
    force = eq.forcing(spec, s.X, s.Y, 0.0)
    cx, cy, lam = spec.get("cx", 0.0), spec.get("cy", 0.0), spec.get("lam", 0.0)

    def rhs(u):
        u_hat = np.fft.fft2(u)
        ux, uy = s.derivative(u_hat, "x"), s.derivative(u_hat, "y")
        flux_x_hat, flux_y_hat = np.fft.fft2(kappa*ux), np.fft.fft2(kappa*uy)
        diffusion = s.derivative(flux_x_hat, "x")+s.derivative(flux_y_hat, "y")
        advection = cx*u*ux+cy*u*uy
        # dealias the quadratic/cubic products
        nonlinear = np.real(np.fft.ifft2(np.fft.fft2(-advection+lam*(u-u**3))*s.dealias))
        return diffusion+nonlinear+force
    u = eq.initial(spec, s.X, s.Y)
    frames = [s.sample(u, n_out)]
    t = 0.0
    for target in frame_times[1:]:
        steps = max(1, int(round((target-t)/dt)))
        h = (target-t)/steps
        for _ in range(steps):
            k1 = rhs(u)
            k2 = rhs(u+h/2*k1)
            k3 = rhs(u+h/2*k2)
            k4 = rhs(u+h*k3)
            u = u+h/6*(k1+2*k2+2*k3+k4)
        t = target
        frames.append(s.sample(u, n_out))
    return np.array(frames)


# ---------------------------------------------------------------- 4-D elliptic with invariant coefficient (section 9)
def elliptic4d_grid(n):
    """Cell centres of the n^4 grid on [-1, 1]^4 with the unit-ball mask |x'| < 1 in the first three coordinates."""
    centres = -1+(np.arange(n)+0.5)*2.0/n
    c3 = np.stack(np.meshgrid(centres, centres, centres, indexing="ij"), axis=-1).reshape(-1, 3)
    inside = (c3**2).sum(axis=1) < 1.0
    return centres, inside.reshape(n, n, n)


def elliptic4d(spec, n, tol=1e-11, maxiter=20000):
    """-div(a grad u) = s0 on B^3 x (-1, 1), u = 0 on the boundary: cell-centred finite volumes on the masked
    n^4 grid, harmonic face averaging, Dirichlet faces at distance h/2, Jacobi-preconditioned CG (float64).
    Returns (points[N, 4], u[N]) for the active cells in C order over (x1, x2, x3, z)."""
    from scipy.sparse.linalg import cg
    centres, ball = elliptic4d_grid(n)
    h = 2.0/n
    active = np.broadcast_to(ball[..., None], (n, n, n, n))
    index = -np.ones((n, n, n, n), dtype=np.int64)
    count = int(active.sum())
    index[active] = np.arange(count)
    X1, X2, X3, Z = np.meshgrid(centres, centres, centres, centres, indexing="ij")
    a = np.where(active, eq.coefficient(spec, X1, X2, X3, Z), 0.0)
    rhs = eq.forcing(spec, X1[active], X2[active], X3[active], Z[active])
    rows, cols, vals = [], [], []
    diag = np.zeros(count)
    for axis in range(4):
        lo = [slice(None)]*4
        hi = [slice(None)]*4
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        a_lo, a_hi = a[tuple(lo)], a[tuple(hi)]
        act_lo, act_hi = active[tuple(lo)], active[tuple(hi)]
        both = act_lo & act_hi
        face = 2*a_lo[both]*a_hi[both]/(a_lo[both]+a_hi[both])/h**2
        i_lo, i_hi = index[tuple(lo)][both], index[tuple(hi)][both]
        rows += [i_lo, i_hi]
        cols += [i_hi, i_lo]
        vals += [-face, -face]
        np.add.at(diag, i_lo, face)
        np.add.at(diag, i_hi, face)
        # Dirichlet faces: an active cell next to an inactive cell or the edge of the box
        for cell_act, other_act, cell_a, cell_index in ((act_lo, act_hi, a_lo, index[tuple(lo)]),
                                                         (act_hi, act_lo, a_hi, index[tuple(hi)])):
            wall = cell_act & ~other_act
            np.add.at(diag, cell_index[wall], 2*cell_a[wall]/h**2)
        first = [slice(None)]*4
        last = [slice(None)]*4
        first[axis], last[axis] = 0, -1
        for edge in (tuple(first), tuple(last)):
            edge_act = active[edge]
            np.add.at(diag, index[edge][edge_act], 2*a[edge][edge_act]/h**2)
    rows.append(np.arange(count))
    cols.append(np.arange(count))
    vals.append(diag)
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(count, count))
    M = sparse.diags(1.0/diag)
    u, info = cg(A, rhs, rtol=tol, maxiter=maxiter, M=M)
    if info != 0:
        raise RuntimeError(f"CG did not converge (info={info})")
    points = np.stack([X1[active], X2[active], X3[active], Z[active]], axis=-1)
    return points, u
