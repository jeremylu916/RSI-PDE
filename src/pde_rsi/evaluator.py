"""Independent reference comparison, physical diagnostics, and validation gates."""

import math
import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
from scipy.special import roots_jacobi

from .equations import forcing
from .equations2d import forcing2d
from .reference import output_nodes, output_nodes2d, output_nodes_group, reference, reference2d, reference_group
from .tasks import GROUP_FAMILIES, WAVE_FAMILIES

# v3 added the 2D wave path and v4 the group FPDE paths. v5 moves every family to the fine evaluation
# grid (1D 512 x 500, wave 128 x 128 x 100); grid="legacy" reproduces v2-v4 scores exactly.
EVALUATOR_VERSION = "manufactured-evaluator-v5-fine-grid"
LEGACY_EVALUATOR_VERSION = "manufactured-evaluator-v4-group"


def physical_residual(task, solution):
    """Final-time residual using a derivative/quadrature independent of solver stepping.

    Diagnostic only: interpolation and coarse-grid differentiation have their own
    error. In particular a small discrete residual is not an accuracy certificate.
    """
    x, t, u = solution.x, solution.t, solution.u
    dx = x[1]-x[0]
    laplacian = (u[-1, 2:]-2*u[-1, 1:-1]+u[-1, :-2])/dx**2
    f = forcing(task, x[1:-1], 0.0 if task.family == "poisson" else t[-1])
    if task.family == "poisson":
        residual = -task.diffusivity*laplacian-f
    else:
        if task.orders:
            derivative = PchipInterpolator(t, u[:, 1:-1], axis=0).derivative()
            temporal = np.zeros(len(x)-2)
            for alpha, weight in zip(task.orders, task.weights):
                nodes, quadrature = roots_jacobi(32, -alpha, 0)
                samples = derivative((nodes+1)*t[-1]/2)
                temporal += weight*(t[-1]/2)**(1-alpha)/math.gamma(1-alpha)*(quadrature @ samples)
        else:
            temporal = np.gradient(u[:, 1:-1], t, axis=0, edge_order=2)[-1]
        residual = temporal-task.diffusivity*laplacian-f
        if task.family in ("convection_diffusion", "burgers"):
            gradient = (u[-1, 2:]-u[-1, :-2])/(2*dx)
            velocity = task.velocity if task.family == "convection_diffusion" else u[-1, 1:-1]
            residual += velocity*gradient
    return float(np.linalg.norm(residual)/max(np.linalg.norm(f), 1e-14))


def physical_residual2d(task, x, y, t, u):
    """Final-time residual of u_tt - D*Laplacian(u) - f; diagnostic only."""
    dx = x[1]-x[0]
    acceleration = np.gradient(np.gradient(u, t, axis=0, edge_order=2), t, axis=0, edge_order=2)[-1, 1:-1, 1:-1]
    final = u[-1]
    laplacian = ((final[1:-1, 2:]-2*final[1:-1, 1:-1]+final[1:-1, :-2])/dx**2
                 + (final[2:, 1:-1]-2*final[1:-1, 1:-1]+final[:-2, 1:-1])/dx**2)
    xx, yy = np.meshgrid(x[1:-1], y[1:-1])
    f = forcing2d(task, xx, yy, t[-1])
    residual = acceleration-task.diffusivity*laplacian-f
    return float(np.linalg.norm(residual)/max(np.linalg.norm(f), 1e-14))


def evaluate2d(task, solution, runtime_seconds=0.0, wall_seconds=0.0, peak_memory_mb=0.0, grid="fine"):
    if solution.y is None:
        raise ValueError("2D solutions require x, y, t, and u")
    x, y, t, u = map(np.asarray, (solution.x, solution.y, solution.t, solution.u))
    if x.ndim != 1 or y.ndim != 1 or t.ndim != 1 or u.shape != (len(t), len(y), len(x)) or min(len(x), len(y)) < 3:
        raise ValueError("Invalid solution dimensions; expected u[t, y, x]")
    if not all(np.isfinite(a).all() for a in (x, y, t, u)) or not np.isrealobj(u):
        raise ValueError("Non-finite or complex solution")
    for name, axis in (("x", x), ("y", y)):
        if not np.all(np.diff(axis) > 0) or not np.allclose(np.diff(axis), axis[1]-axis[0], rtol=1e-9, atol=1e-13):
            raise ValueError(f"Expected strictly increasing uniform {name} grid")
        if not np.isclose(axis[0], 0, atol=1e-12, rtol=0) or not np.isclose(axis[-1], task.length, atol=1e-12, rtol=0):
            raise ValueError(f"Solution does not cover the {name} domain")
    if (len(t) < 3 or not np.all(np.diff(t) > 0) or abs(t[0]) > 1e-12 or
            not np.isclose(t[-1], task.final_time, atol=1e-12, rtol=0)):
        raise ValueError("Solution does not cover the requested time horizon")
    eval_x, eval_y, eval_t = output_nodes2d(task, grid)
    tt, yy, xx = np.meshgrid(eval_t, eval_y, eval_x, indexing="ij")
    predicted = RegularGridInterpolator((t, y, x), u)(np.stack([tt, yy, xx], axis=-1))
    truth = reference2d(task, eval_x, eval_y, eval_t)
    error = predicted-truth
    space_error = trapezoid(trapezoid(error**2, eval_x, axis=2), eval_y, axis=1)
    space_norm = trapezoid(trapezoid(truth**2, eval_x, axis=2), eval_y, axis=1)
    integrated_error, integrated_norm = trapezoid(space_error, eval_t), trapezoid(space_norm, eval_t)
    relative_l2 = float(np.sqrt(integrated_error/max(integrated_norm, 1e-30)))
    final_l2 = float(np.sqrt(space_error[-1]/max(space_norm[-1], 1e-30)))
    boundary_error = float(max(np.max(np.abs(u[:, :, [0, -1]])), np.max(np.abs(u[:, [0, -1], :]))))
    initial_error = float(np.max(np.abs(u[0]-reference2d(task, x, y, [0])[0])))
    resources_ok = wall_seconds <= task.max_seconds and peak_memory_mb <= task.memory_mb
    accuracy = relative_l2 <= task.relative_l2_tolerance and final_l2 <= task.relative_l2_tolerance
    constraints = boundary_error <= task.boundary_tolerance and initial_error <= task.initial_tolerance
    return {
        "task_id": task.task_id, "family": task.family, "split": task.split, "status": "ok",
        "relative_l2": relative_l2, "final_relative_l2": final_l2,
        "absolute_l2": float(np.sqrt(integrated_error)), "final_absolute_l2": float(np.sqrt(space_error[-1])),
        "absolute_linf": float(np.max(np.abs(error))),
        "boundary_error": boundary_error, "initial_error": initial_error,
        "physical_residual_relative": physical_residual2d(task, x, y, t, u),
        "runtime_seconds": runtime_seconds, "wall_seconds": wall_seconds, "peak_memory_mb": peak_memory_mb,
        "accuracy_pass": accuracy, "constraints_pass": constraints, "resources_pass": resources_ok,
        "passed": accuracy and constraints and resources_ok,
        "method": solution.method, "nx": len(x)-1, "ny": len(y)-1, "nt": len(t)-1,
        "evaluator_version": EVALUATOR_VERSION if grid == "fine" else LEGACY_EVALUATOR_VERSION,
    }


def evaluate(task, solution, runtime_seconds=0.0, wall_seconds=0.0, peak_memory_mb=0.0, grid="fine"):
    if task.family in WAVE_FAMILIES:
        return evaluate2d(task, solution, runtime_seconds, wall_seconds, peak_memory_mb, grid)
    group = task.family in GROUP_FAMILIES
    if task.family == "fode":
        x = np.asarray(solution.x)
        if x.shape != (1,) or x[0] != 0:
            raise ValueError("FODE solutions use x = [0] and u[t, 0]")
    x, t, u = map(np.asarray, (solution.x, solution.t, solution.u))
    if x.ndim != 1 or t.ndim != 1 or u.shape != (len(t), len(x)) or (len(x) < 3 and task.family != "fode"):
        raise ValueError("Invalid solution dimensions")
    if not all(np.isfinite(a).all() for a in (x, t, u)) or not np.isrealobj(u):
        raise ValueError("Non-finite or complex solution")
    if task.family != "fode":
        if not np.all(np.diff(x) > 0) or not np.allclose(np.diff(x), x[1]-x[0], rtol=1e-9, atol=1e-13):
            raise ValueError("Expected strictly increasing uniform spatial grid")
        if not np.isclose(x[0], 0, atol=1e-12, rtol=0) or not np.isclose(x[-1], task.length, atol=1e-12, rtol=0):
            raise ValueError("Solution does not cover the spatial domain")
    if task.family == "poisson":
        if len(t) != 1 or abs(t[0]) > 1e-12:
            raise ValueError("Poisson output must contain t=[0]")
    elif (len(t) < 3 or not np.all(np.diff(t) > 0) or abs(t[0]) > 1e-12 or
          not np.isclose(t[-1], task.final_time, atol=1e-12, rtol=0)):
        raise ValueError("Solution does not cover the requested time horizon")
    exact = reference_group if group else reference
    eval_x, eval_t = output_nodes_group(task, grid) if group else output_nodes(task, grid)
    if task.family == "poisson":
        predicted = np.interp(eval_x, x, u[0])[None, :]
    elif task.family == "fode":
        predicted = np.interp(eval_t, t, u[:, 0])[:, None]
    else:
        tt, xx = np.meshgrid(eval_t, eval_x, indexing="ij")
        predicted = RegularGridInterpolator((t, x), u)(np.stack([tt, xx], axis=-1))
    truth = exact(task, eval_x, eval_t)
    error = predicted-truth
    if task.family == "fode":
        space_error, space_norm = error[:, 0]**2, truth[:, 0]**2
    else:
        space_error = trapezoid(error**2, eval_x, axis=1)
        space_norm = trapezoid(truth**2, eval_x, axis=1)
    integrated_error = space_error[0] if len(eval_t) == 1 else trapezoid(space_error, eval_t)
    integrated_norm = space_norm[0] if len(eval_t) == 1 else trapezoid(space_norm, eval_t)
    relative_l2 = float(np.sqrt(integrated_error/max(integrated_norm, 1e-30)))
    final_l2 = float(np.sqrt(space_error[-1]/max(space_norm[-1], 1e-30)))
    boundary_error = 0.0 if task.family == "fode" else float(np.max(np.abs(u[:, [0, -1]])))
    initial_error = 0.0 if task.family == "poisson" else float(np.max(np.abs(u[0]-exact(task, x, [0])[0])))
    resources_ok = wall_seconds <= task.max_seconds and peak_memory_mb <= task.memory_mb
    return {
        "task_id": task.task_id, "family": task.family, "split": task.split, "status": "ok",
        "relative_l2": relative_l2, "final_relative_l2": final_l2,
        "absolute_l2": float(np.sqrt(integrated_error)),
        "final_absolute_l2": float(np.sqrt(space_error[-1])),
        "absolute_linf": float(np.max(np.abs(error))),
        "boundary_error": boundary_error, "initial_error": initial_error,
        "physical_residual_relative": None if group else physical_residual(task, solution),
        "runtime_seconds": runtime_seconds, "wall_seconds": wall_seconds, "peak_memory_mb": peak_memory_mb,
        "accuracy_pass": relative_l2 <= task.relative_l2_tolerance and final_l2 <= task.relative_l2_tolerance,
        "constraints_pass": boundary_error <= task.boundary_tolerance and initial_error <= task.initial_tolerance,
        "resources_pass": resources_ok,
        "passed": (relative_l2 <= task.relative_l2_tolerance and final_l2 <= task.relative_l2_tolerance and
                   boundary_error <= task.boundary_tolerance and initial_error <= task.initial_tolerance and resources_ok),
        "method": solution.method, "nx": len(x)-1, "nt": len(t)-1,
        "evaluator_version": EVALUATOR_VERSION if grid == "fine" else LEGACY_EVALUATOR_VERSION,
    }


def failed_result(task, status, message, wall_seconds):
    return {"task_id": task.task_id, "family": task.family, "split": task.split, "status": status,
            "message": message[:1000], "relative_l2": 1e6, "final_relative_l2": 1e6,
            "absolute_l2": None, "final_absolute_l2": None,
            "accuracy_pass": False, "constraints_pass": False, "resources_pass": False, "passed": False,
            "runtime_seconds": wall_seconds, "wall_seconds": wall_seconds, "peak_memory_mb": 0.0,
            "evaluator_version": EVALUATOR_VERSION}


def summarize(results):
    if not results:
        raise ValueError("No evaluation results")
    families = {}
    for family in sorted({r["family"] for r in results}):
        subset = [r for r in results if r["family"] == family]
        families[family] = {"count": len(subset), "pass_rate": sum(r["passed"] for r in subset)/len(subset),
                            "mean_relative_l2": float(np.mean([r["relative_l2"] for r in subset])),
                            "max_relative_l2": max(r["relative_l2"] for r in subset),
                            "solver_seconds": sum(r["runtime_seconds"] for r in subset)}
    return {"count": len(results), "pass_rate": sum(r["passed"] for r in results)/len(results),
            "macro_relative_l2": float(np.mean([r["mean_relative_l2"] for r in families.values()])),
            "solver_seconds": sum(r["runtime_seconds"] for r in results), "families": families}


def acceptance_gate(baseline_results, candidate_results, minimum_gain=0.005, regression_tolerance=0.05,
                    pass_first=False):
    """pass_first orders candidates by passed cases first: a candidate that passes more cases and fails none that
    passed is accepted despite an aggregate gain below minimum_gain, family-mean regressions, or regressions of
    cases failing in both versions. Execution, IC/BC and resource failures still reject, and cases passing in both
    versions keep the instance-regression check, protecting their margin to the tolerance."""
    for records in (baseline_results, candidate_results):
        if len(records) != len({r["task_id"] for r in records}):
            raise ValueError("Development comparisons require unique task IDs")
    versions = {r.get("evaluator_version", EVALUATOR_VERSION) for r in baseline_results+candidate_results}
    if len(versions) != 1:
        raise ValueError("Development comparisons require matching evaluator versions")
    if {r["task_id"] for r in baseline_results} != {r["task_id"] for r in candidate_results}:
        raise ValueError("Development comparisons must use the same tasks")
    baseline, candidate = summarize(baseline_results), summarize(candidate_results)
    reasons = []
    if any(r["status"] != "ok" or not r["constraints_pass"] or not r["resources_pass"] for r in candidate_results):
        reasons.append("execution, IC/BC, or resource failure")
    if candidate["macro_relative_l2"] >= baseline["macro_relative_l2"]*(1-minimum_gain):
        reasons.append("insufficient aggregate accuracy gain")
    for family, before in baseline["families"].items():
        after = candidate["families"][family]
        if after["mean_relative_l2"] > max(before["mean_relative_l2"]*(1+regression_tolerance), 1e-12):
            reasons.append(f"family regression: {family}")
        if after["pass_rate"] < before["pass_rate"]:
            reasons.append(f"pass-rate regression: {family}")
    # Paired instance regression prevents a family average from hiding a severe failure.
    before_by_id = {r["task_id"]: r for r in baseline_results}
    for after in candidate_results:
        before = before_by_id[after["task_id"]]
        if after["relative_l2"] > max(before["relative_l2"]*1.25, 1e-10):
            reasons.append(f"instance regression: {after['task_id']}")
        if after["final_relative_l2"] > max(before["final_relative_l2"]*1.25, 1e-10):
            reasons.append(f"final-time instance regression: {after['task_id']}")
    decision = {"accepted": not reasons, "reasons": reasons,
                "relative_gain": 1-candidate["macro_relative_l2"]/max(baseline["macro_relative_l2"], 1e-30)}
    if pass_first and reasons:
        after_by_id = {r["task_id"]: r for r in candidate_results}
        gained = sorted(i for i, r in after_by_id.items() if r["passed"] and not before_by_id[i]["passed"])
        lost = sorted(i for i, r in after_by_id.items() if before_by_id[i]["passed"] and not r["passed"])
        if gained and not lost:
            protected = {i for i, r in after_by_id.items() if r["passed"] and before_by_id[i]["passed"]}
            kept = [r for r in reasons if r == "execution, IC/BC, or resource failure"
                    or (r.split(": ", 1)[0] in ("instance regression", "final-time instance regression")
                        and r.split(": ", 1)[1] in protected)]
            decision.update(accepted=not kept, reasons=kept,
                            pass_first={"gained": gained, "waived": [r for r in reasons if r not in kept]})
    return decision
