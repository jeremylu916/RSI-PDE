"""Export every pdebench case as a folder: agent inputs, hidden truth, and the final agents' outputs.

Usage (after sourcing scripts/env.sh):
    python scripts/export_pdebench_cases.py --runs runs/code-agents-pdebench-*-20261017 --output tasks/pdebench

tasks/pdebench/<case_id>/
    README.md                tex case, hidden truth, and every model's single-run / RSI result
    input/agent_view.json    exactly what the model and program are told (no class, coefficients, or tex case)
    input/library.json       candidate equation classes
    input/data.npz           noisy samples (data_x, data_y, data_t present only where the case has them), noise_std
    truth/truth.json         tex case, true class and coefficients, full hidden spec (never given to agents)
    truth/problem.tex        the true equation and exact solution with this case's numbers
    truth/reference.npz      exact solution on the evaluation grid (x, t, u; u[t, y, x] for the 2-D case)
    outputs/<model>/<arm>/   solution.npz, solver.py (the arm's frozen program), metrics.json
Sealed-test cases (case-r*) show outputs only after scripts/pdebench_sealed_test.py has run.
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np

from pde_rsi.pdebench import (CLASS, LIBRARY, SPLITS, TITLES, evaluation_nodes, make_cases, reference,
                              true_parameters)
from pde_rsi.storage import PROJECT_ROOT, project_path

MODELS = {"us.anthropic.claude-sonnet-4-5-20250929-v1:0": "sonnet45", "openai.gpt-oss-120b-1:0": "gpt-oss",
          "us.openai.gpt-5.5": "gpt55", "us.openai.gpt-6.1-sol": "gpt61", "us.anthropic.claude-opus-5-5": "opus55"}
LABELS = {"sonnet45": "Claude Sonnet 4.5", "gpt-oss": "GPT-OSS-120B", "gpt55": "GPT-5.5", "gpt61": "GPT-6.1",
          "opus55": "Claude Opus 5.5"}
SPLIT_NAMES = {"evolution": "training", "development": "validation", "test": "sealed test"}
EQUATION = {
    "poisson": r"-D\,u_{xx} = f(x)",
    "heat": r"u_t - D\,u_{xx} = f(x,t)",
    "convection_diffusion": r"u_t + v\,u_x - D\,u_{xx} = f(x,t)",
    "burgers": r"u_t + u\,u_x - D\,u_{xx} = f(x,t)",
    "wave2d": r"u_{tt} - D\,(u_{xx}+u_{yy}) = f(x,y,t)",
    "fode": r"{}^{C}_{0}D_t^{\alpha}u(t) = f(t)",
    "caputo": r"{}^{C}_{0}D_t^{\alpha}u - D\,u_{xx} = f(x,t)",
    "multi_term": r"w_1\,{}^{C}_{0}D_t^{\alpha_1}u + w_2\,{}^{C}_{0}D_t^{\alpha_2}u - D\,u_{xx} = f(x,t)",
    "variable_order_t": r"{}^{C}_{0}D_t^{\alpha(t)}u - D\,u_{xx} = f(x,t),\quad \alpha(t)=c\,\sin^2 t",
    "variable_order_xt": r"{}^{C}_{0}D_t^{\alpha(x,t)}u - D\,u_{xx} = f(x,t),\quad \alpha(x,t)=c\,t^2\sin x",
}
STEADY_CD = r"v\,u_x - D\,u_{xx} = f(x)"


def g(value):
    return f"{value:.4g}"


def profile_tex(s, var="x"):
    L = g(s["b"]-s["a"])
    return {"sine": rf"\sin(\pi {var}/{L})", "polynomial": rf"4\frac{{{var}}}{{{L}}}\Bigl(1-\frac{{{var}}}{{{L}}}\Bigr)",
            "mixed_sine": rf"\bigl[\sin(\pi {var}/{L})+\tfrac14\sin(3\pi {var}/{L})\bigr]"}[s["profile"]]


def exact_tex(s):
    p = s["problem"]
    if p == "1a":
        return rf"u(x) = {g(s['A'])}\,{profile_tex(s)}"
    if p == "1b":
        return r"u(x) = \sin(2\pi x) + 0.1\sin(50\pi x)"
    if p == "1c":
        return rf"u(x) = \arctan\bigl({g(s['sharpness'])}(x-{g(s['x0'])})\bigr)"
    if p == "3b":
        return rf"u(x) = \frac{{e^{{\mathrm{{Pe}}\,x/L}}-1}}{{e^{{\mathrm{{Pe}}}}-1}},\ \mathrm{{Pe}}={g(s['v']*(s['b']-s['a'])/s['D'])}"
    if p == "3c":
        return (rf"u(x) = \frac{{1}}{{v}}\Bigl[x-L\frac{{e^{{\mathrm{{Pe}}\,x/L}}-1}}{{e^{{\mathrm{{Pe}}}}-1}}\Bigr],"
                rf"\ \mathrm{{Pe}}={g(s['v']*(s['b']-s['a'])/s['D'])}")
    if p in ("2a", "3a", "4a", "7", "8"):
        return rf"u(x,t) = {g(s['A'])}\,{profile_tex(s)}\,\bigl(1+t^{{{g(s['beta'])}}}\bigr)"
    if p == "2b":
        return rf"u(x,t) = e^{{-D\pi^2 t/L^2}}\sin(\pi x/L),\ L={g(s['b'])}"
    if p == "2c":
        return (rf"u(x,t) = a(t)\sin(\pi x/L),\ a(t)=\frac{{\lambda\cos\gamma t+\gamma\sin\gamma t-\lambda e^{{-\lambda t}}}}"
                rf"{{\lambda^2+\gamma^2}},\ \lambda=D\pi^2/L^2,\ \gamma={g(s['gamma'])},\ f=\sin(\pi x/L)\cos(\gamma t)")
    if p == "3d":
        return (rf"u(x,t) = \frac{{\sigma_0}}{{\sigma(t)}}\exp\Bigl(-\frac{{(x-{g(s['x0'])}-vt)^2}}{{2\sigma(t)^2}}\Bigr),"
                rf"\ \sigma(t)^2={g(s['sigma0'])}^2+2Dt")
    if p == "4b":
        return r"u(x,0) = -\sin(\pi x),\ u(\pm1,t)=0;\ u \text{ by Cole--Hopf (Basdevant et al.)}"
    if p == "4c":
        return (rf"u(x,t) = \tfrac{{u_\ell+u_r}}{{2}}-\tfrac{{u_\ell-u_r}}{{2}}\tanh\Bigl(\frac{{(u_\ell-u_r)(x-ct)}}{{4D}}\Bigr),"
                rf"\ u_\ell={g(s['ul'])},\ u_r={g(s['ur'])},\ c=\tfrac{{u_\ell+u_r}}{{2}}")
    if p == "5":
        return (rf"u(x,y,t) = {g(s['A'])}\,{profile_tex(s, 'x')}\,{profile_tex(s, 'y')}\,"
                rf"\bigl[\cos(\omega t)+t^{{{g(s['beta'])}}}\bigr],\ \omega=\pi\sqrt{{2D}}/L")
    if p == "6":
        return rf"u(t) = t^{{{g(s['p'])}}}"
    return rf"u(x,t) = t^{{{g(s['p'])}}}\sin x"


def problem_tex(case):
    s, task = case.truth.spec, case.truth
    numbers = ", ".join(f"{k}={g(v)}" for k, v in true_parameters(task).items())
    if s["problem"] == "3b":
        numbers += f", v={g(s['v'])}, D={g(s['D'])}"
    equation = STEADY_CD if task.steady and task.family == "convection_diffusion" else EQUATION[task.family]
    if task.dimension == 0:
        domain = rf"t\in(0,{g(s['T'])}],\ u(0)=0"
    elif task.dimension == 2:
        domain = rf"(x,y)\in(0,{g(s['b'])})^2,\ t\in(0,{g(s['T'])}],\ u=0\text{{ on }}\partial\Omega"
    else:
        domain = rf"x\in({g(s['a'])},{g(s['b'])})" + ("" if task.steady else rf",\ t\in(0,{g(s['T'])}]")
    return (f"% {case.case_id} ({SPLIT_NAMES[case.split]}): pde_benchmarks.tex Case {s['problem']} "
            f"({TITLES[s['problem']]}); hidden truth, never given to agents\n\\begin{{align*}}\n"
            f"&{equation},\\qquad {domain},\\\\\n&{numbers},\\\\\n&{exact_tex(s)}.\n\\end{{align*}}\n")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT/"tasks/pdebench")
    args = parser.parse_args()
    output = project_path(args.output)
    if output.exists():
        shutil.rmtree(output)
    runs = [(MODELS[json.loads((r/"manifest.json").read_text())["model_config"]["model"]], r)
            for r in map(project_path, sorted(args.runs))]
    index_rows = []
    for split in SPLITS:
        for case in make_cases(split):
            task, s, folder = case.truth, case.truth.spec, output/case.case_id
            data = case.data()
            write_json(folder/"input/agent_view.json", {**case.public_view(), "library": LIBRARY})
            write_json(folder/"input/library.json", LIBRARY)
            np.savez_compressed(folder/"input/data.npz", noise_std=data["noise_std"], data_u=data["u"],
                                **{f"data_{k}": data[k] for k in ("x", "y", "t") if data[k] is not None})
            params = true_parameters(task)
            write_json(folder/"truth/truth.json", {"tex_case": s["problem"], "title": TITLES[s["problem"]],
                                                   "true_class": task.family, "true_parameters": params,
                                                   "spec": s})
            (folder/"truth/problem.tex").write_text(problem_tex(case))
            x, t = evaluation_nodes(task)
            extra = {"y": x} if task.dimension == 2 else {}
            np.savez_compressed(folder/"truth/reference.npz", x=x, t=t, u=reference(task), **extra)
            numbers = ", ".join(f"{k}={g(v)}" for k, v in params.items())
            lines = [f"# Case `{case.case_id}` ({SPLIT_NAMES[split]})", "",
                     f"pde_benchmarks.tex Case **{s['problem']}**: {TITLES[s['problem']]}. Hidden truth: "
                     f"**{task.family}**, {numbers}.", "",
                     "Agents see only `input/`. `truth/` is for inspection and was never given to agents.", "",
                     "| Model | Arm | Chosen class | Coefficients | Method | Relative L2 | Passed |",
                     "| --- | --- | --- | --- | --- | ---: | :---: |"]
            for key, run in runs:
                for arm in ("single", "rsi"):
                    source = run/arm/("sealed_test" if split == "test" else "")
                    results_path = source/"results.json" if split == "test" else run/arm/"frozen_results.json"
                    if not results_path.exists():
                        continue
                    result = next(r for r in json.loads(results_path.read_text()) if r["case_id"] == case.case_id)
                    target = folder/"outputs"/key/arm
                    target.mkdir(parents=True)
                    solution = source/"solutions"/f"{case.case_id}.npz"
                    if solution.exists():
                        shutil.copyfile(solution, target/"solution.npz")
                    shutil.copyfile(run/arm/"programs/frozen/identify.py", target/"solver.py")
                    write_json(target/"metrics.json", {**result, "source_run": str(run.relative_to(PROJECT_ROOT))})
                    coefficients = ", ".join(f"{k}={g(v)}" for k, v in (result.get("identified_parameters") or {}).items())
                    chosen = result.get("identified_family") or "—"
                    mark = " ✓" if result.get("family_correct") else (" ✗" if chosen != "—" else "")
                    rel = result["relative_l2"] if result["status"] == "ok" else None
                    lines.append(f"| {LABELS[key]} | {'single run' if arm == 'single' else 'RSI'} | {chosen}{mark} | "
                                 f"{coefficients or '—'} | {result.get('method') or '—'} | "
                                 f"{f'{rel:.2e}' if rel is not None and math.isfinite(rel) else result['status']} | "
                                 f"{'yes' if result['passed'] else 'no'} |")
            (folder/"README.md").write_text("\n".join(lines)+"\n")
            index_rows.append(f"| [`{case.case_id}`]({case.case_id}/README.md) | {SPLIT_NAMES[split]} | "
                              f"{s['problem']} | {TITLES[s['problem']]} | {task.family} | {numbers} |")
    (output/"README.md").write_text("\n".join([
        "# pde_benchmarks.tex cases (student protocol)", "",
        "Each case is one case of `pde_benchmarks.tex` with its own numbers. Agents receive `input/` (the case view,",
        "the equation library, and 256 noisy samples), identify the class and coefficients, choose a method",
        "(numerical or PINN), and write the solver from scratch. Training and validation cases were used by the",
        "improvement loop; sealed-test cases were scored only after the agents were frozen. Each run has two agents:",
        "the single run (one model request, solver frozen as returned) and RSI (gated rounds with inherited workflow",
        "and memory). Outputs come from:",
        "", *[f"- `{key}`: `{run.relative_to(PROJECT_ROOT)}`" for key, run in runs], "",
        "| Case | Split | tex case | Description | True class | True coefficients |",
        "| --- | --- | --- | --- | --- | --- |", *index_rows, ""]))
    print(output)


if __name__ == "__main__":
    main()
