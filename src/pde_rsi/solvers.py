"""Finite differences, backward Euler, IMEX Burgers, nonuniform Caputo L1, 2D leapfrog waves, group FPDEs."""

from dataclasses import dataclass
import math
import numpy as np
from scipy.linalg import solve_banded

from .equations import forcing, initial
from .equations2d import forcing2d, initial2d, initial_velocity2d
from .equations_group import forcing_group, order


@dataclass
class Solution:
    x: np.ndarray
    t: np.ndarray
    u: np.ndarray
    method: str
    y: np.ndarray | None = None  # 2D solutions store u[t, y, x]

    def save_npz(self, path):
        from .storage import project_path
        path = project_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"x": self.x, "t": self.t, "u": self.u, "method": np.asarray(self.method)}
        if self.y is not None:
            arrays["y"] = self.y
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load_npz(cls, path):
        from .storage import project_path
        with np.load(project_path(path), allow_pickle=False) as archive:
            if set(archive.files) not in ({"x", "t", "u", "method"}, {"x", "y", "t", "u", "method"}):
                raise ValueError("Solution archive requires exactly x, t, u, method (and y for 2D)")
            method = archive["method"]
            if method.shape != () or method.dtype.kind not in ("U", "S"):
                raise ValueError("Method description must be a scalar string")
            arrays = [archive[name] for name in ("x", "t", "u")]
            y = archive["y"] if "y" in archive.files else None
            if any(array.dtype.kind not in "fiu" for array in arrays+([y] if y is not None else [])):
                raise ValueError("Solution arrays must contain real numbers")
            return cls(*arrays, str(method.item()), y)


def band_matrix(n, lower, diagonal, upper):
    ab = np.zeros((3, n))
    ab[0, 1:] = upper
    ab[1, :] = diagonal
    ab[2, :-1] = lower
    return ab


def l1_weights(times, n, alpha):
    """Weights on intervals [t_k,t_{k+1}], k=0,...,n-1.

    Factored differences and expm1 avoid cancellation on strongly graded grids.
    """
    left = times[n] - times[:n]
    right = times[n] - times[1:n+1]
    p = 1-alpha
    difference = np.empty(n)
    nonzero = right > 0
    difference[~nonzero] = left[~nonzero]**p
    ratio = np.diff(times[:n+1])[nonzero]/left[nonzero]
    difference[nonzero] = left[nonzero]**p*(-np.expm1(p*np.log1p(-ratio)))
    return difference / (math.gamma(2-alpha)*np.diff(times[:n+1]))


def solve_wave2d(task, settings):
    """Explicit leapfrog with a 5-point Laplacian; Taylor first step; CFL number 0.5."""
    nx = settings.nx
    x = np.linspace(0, task.length, nx+1)
    dx = x[1]-x[0]
    xx, yy = np.meshgrid(x, x)
    speed = math.sqrt(task.diffusivity)
    nt = max(settings.nt, math.ceil(task.final_time*speed/(0.5*dx)))
    if nt > 4096:
        raise ValueError("Wave CFL requirement exceeds the allowed temporal grid")
    times = np.linspace(0, task.final_time, nt+1)
    dt = times[1]-times[0]
    inner = (slice(1, -1), slice(1, -1))

    def laplacian(v):
        return (v[1:-1, 2:]+v[1:-1, :-2]+v[2:, 1:-1]+v[:-2, 1:-1]-4*v[1:-1, 1:-1])/dx**2

    u = np.zeros((nt+1, nx+1, nx+1))
    u[0] = initial2d(task, xx, yy)
    u[0][[0, -1], :] = u[0][:, [0, -1]] = 0
    # The source may be singular at t=0, so the first step samples it at t=dt.
    u[1][inner] = (u[0][inner]+dt*initial_velocity2d(task, xx, yy)[inner]
                   + 0.5*dt*dt*(task.diffusivity*laplacian(u[0])+forcing2d(task, xx, yy, dt)[inner]))
    for n in range(1, nt):
        u[n+1][inner] = (2*u[n][inner]-u[n-1][inner]
                         + dt*dt*(task.diffusivity*laplacian(u[n])+forcing2d(task, xx, yy, times[n])[inner]))
    return Solution(x, times, u, "leapfrog 2D wave", y=x.copy())


def solve_group(task, settings):
    """Graded-in-time L1 schemes for the group FPDE examples.

    A variable order uses its value at the current step (and grid point) in that step's weights.
    """
    times = task.final_time*np.linspace(0, 1, settings.nt+1)**settings.grading
    if task.family == "fode":
        u = np.zeros(settings.nt+1)
        for n in range(1, settings.nt+1):
            weights = l1_weights(times, n, order(task, times[n]))
            history = weights[:-1] @ np.diff(u[:n]) if n > 1 else 0.0
            u[n] = u[n-1]+(forcing_group(task, 0.0, times[n])-history)/weights[-1]
        return Solution(np.array([0.0]), times, u[:, None], "uniform L1 FODE")
    nx = settings.nx
    x = np.linspace(0, task.length, nx+1)
    diffusion = task.diffusivity/(x[1]-x[0])**2
    u = np.zeros((settings.nt+1, nx+1))
    for n in range(1, settings.nt+1):
        if task.family == "fpde_variable_order_xt":
            alphas = order(task, times[n], x[1:-1])
            weights = np.stack([l1_weights(times, n, a) for a in alphas], axis=1)  # (n, interior points)
            history = np.sum(weights[:-1]*np.diff(u[:n, 1:-1], axis=0), axis=0) if n > 1 else 0.0
        else:
            weights = l1_weights(times, n, order(task, times[n]))
            history = weights[:-1] @ np.diff(u[:n, 1:-1], axis=0) if n > 1 else 0.0
        leading = weights[-1]
        rhs = forcing_group(task, x[1:-1], times[n])+leading*u[n-1, 1:-1]-history
        u[n, 1:-1] = solve_banded((1, 1), band_matrix(nx-1, -diffusion, leading+2*diffusion, -diffusion), rhs)
    return Solution(x, times, u, f"L1 {task.family}")


def solve(task, settings):
    task.validate()
    settings.validate()
    if task.family == "wave2d":
        return solve_wave2d(task, settings)
    if task.family in ("fode", "fpde_caputo", "fpde_variable_order", "fpde_variable_order_xt"):
        return solve_group(task, settings)
    nx = settings.nx
    x = np.linspace(0, task.length, nx+1)
    dx = x[1]-x[0]
    interior = x[1:-1]
    diffusion = task.diffusivity/dx**2
    if task.family == "poisson":
        matrix = band_matrix(nx-1, -diffusion, 2*diffusion, -diffusion)
        u = np.zeros((1, nx+1))
        u[0, 1:-1] = solve_banded((1, 1), matrix, forcing(task, interior, 0))
        return Solution(x, np.array([0.0]), u, "centered finite difference Poisson")
    nt = settings.nt
    # Explicit Burgers transport requires a CFL bound. The uniform time grid is
    # chosen from the specified manufactured envelope, never an evaluator reference.
    if task.family == "burgers":
        envelope = 1.25 if task.profile == "mixed_sine" else 1.0
        speed_bound = envelope*task.amplitude*(1+task.final_time**task.beta)
        nt = max(nt, math.ceil(task.final_time*speed_bound/(0.35*dx)))
        if nt > 1024:
            raise ValueError("Burgers CFL requirement exceeds the allowed temporal grid")
    grading = settings.grading if task.orders else 1.0
    times = task.final_time*np.linspace(0, 1, nt+1)**grading
    u = np.zeros((nt+1, nx+1))
    u[0] = initial(task, x)
    u[0, [0, -1]] = 0
    for n in range(1, nt+1):
        dt = times[n]-times[n-1]
        f = forcing(task, interior, times[n])
        if task.orders:
            coefficients = sum(w*l1_weights(times, n, alpha)
                               for alpha, w in zip(task.orders, task.weights))
            leading = coefficients[-1]
            history = coefficients[:-1] @ np.diff(u[:n, 1:-1], axis=0) if n > 1 else 0
            rhs = f + leading*u[n-1, 1:-1] - history
            matrix = band_matrix(nx-1, -diffusion, leading+2*diffusion, -diffusion)
        else:
            lower, upper = -diffusion, -diffusion
            diagonal = 1/dt+2*diffusion
            rhs = u[n-1, 1:-1]/dt + f
            if task.family == "convection_diffusion":
                if settings.advection == "central":
                    lower -= task.velocity/(2*dx)
                    upper += task.velocity/(2*dx)
                elif task.velocity >= 0:
                    lower -= task.velocity/dx
                    diagonal += task.velocity/dx
                else:
                    upper += task.velocity/dx
                    diagonal -= task.velocity/dx
            if task.family == "burgers":
                previous = u[n-1]
                if settings.advection == "central":
                    gradient = (previous[2:]-previous[:-2])/(2*dx)
                else:
                    gradient = np.where(previous[1:-1] >= 0,
                                        (previous[1:-1]-previous[:-2])/dx,
                                        (previous[2:]-previous[1:-1])/dx)
                rhs -= previous[1:-1]*gradient
            matrix = band_matrix(nx-1, lower, diagonal, upper)
        u[n, 1:-1] = solve_banded((1, 1), matrix, rhs, check_finite=True)
    method = "nonuniform Caputo L1" if task.orders else "IMEX Burgers" if task.family == "burgers" else "backward Euler"
    return Solution(x, times, u, method)
