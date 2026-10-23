"""Export every student-protocol case as a folder: agent inputs, hidden truth, and final agent outputs.

Usage (after sourcing scripts/env.sh):
    python scripts/export_student_cases.py --runs runs/code-agents-student-pilot-*-20261016 --output tasks/student

tasks/student/<case_id>/
    README.md                summary: true equation, and every model's single/RSI result
    input/agent_view.json    exactly what the model and program are told (no family, coefficients, or truth)
    input/library.json       candidate equation classes
    input/data.npz           data_x, data_t, data_u (noisy samples) and noise_std
    truth/truth.json         hidden true class, coefficients, and full task record (never given to agents)
    truth/problem.tex        the true equation with this case's numbers
    truth/reference.npz      exact solution on the evaluator grid (x, t, u)
    outputs/<model>/<arm>/   solution.npz, solver.py (the arm's single program), metrics.json
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np

from pde_rsi.identification import LIBRARY, TRUE_CLASS, StudentProtocol, true_parameters
from pde_rsi.reference import output_nodes, output_nodes_group, reference, reference_group
from pde_rsi.storage import PROJECT_ROOT, project_path

MODELS = {"us.anthropic.claude-sonnet-4-5-20250929-v1:0": "sonnet45", "openai.gpt-oss-120b-1:0": "gpt-oss",
          "us.openai.gpt-5.5": "gpt55", "us.openai.gpt-6.1-sol": "gpt61", "us.anthropic.claude-opus-5-5": "opus55"}
LABELS = {"sonnet45": "Claude Sonnet 4.5", "gpt-oss": "GPT-OSS-120B", "gpt55": "GPT-5.5", "gpt61": "GPT-6.1",
          "opus55": "Claude Opus 5.5"}
TEX = {
    "heat": r"u_t - D\,u_{xx} = f",
    "convection_diffusion": r"u_t + v\,u_x - D\,u_{xx} = f",
    "burgers": r"u_t + u\,u_x - D\,u_{xx} = f",
    "caputo": r"{}^{C}_{0}D_t^{\alpha}u - D\,u_{xx} = f",
    "multi_term": r"w_1\,{}^{C}_{0}D_t^{\alpha_1}u + {}^{C}_{0}D_t^{\alpha_2}u - D\,u_{xx} = f",
    "variable_order_t": r"{}^{C}_{0}D_t^{\alpha(t)}u - D\,u_{xx} = f,\quad \alpha(t)=c\,\sin^2 t",
    "variable_order_xt": r"{}^{C}_{0}D_t^{\alpha(x,t)}u - D\,u_{xx} = f,\quad \alpha(x,t)=c\,t^2\sin x",
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def exact_text(task):
    L = f"{task.length:.4g}"
    if task.family.startswith("fpde_"):
        return rf"u(x,t) = {task.amplitude:.4g}\,t^{{{task.beta:.4g}}}\,\sin(\pi x/{L})"
    s = rf"\sin(\pi x/{L})" if task.profile == "sine" else rf"4\frac{{x}}{{{L}}}\Bigl(1-\frac{{x}}{{{L}}}\Bigr)"
    return rf"u(x,t) = {task.amplitude:.4g}\,{s}\,\bigl(1+t^{{{task.beta:.4g}}}\bigr)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT/"tasks/student")
    args = parser.parse_args()
    output = project_path(args.output)
    if output.exists():
        shutil.rmtree(output)
    protocol = StudentProtocol()
    cases = [c for items in protocol.load_tasks(PROJECT_ROOT, None, ()).values() for c in items]
    runs = [(MODELS[json.loads((r/"manifest.json").read_text())["model_config"]["model"]], r)
            for r in map(project_path, sorted(args.runs))]
    index_rows = []
    for case in cases:
        task, folder = case.truth, output/case.case_id
        data = case.data()
        write_json(folder/"input/agent_view.json", case.public_view())
        write_json(folder/"input/library.json", LIBRARY)
        np.savez_compressed(folder/"input/data.npz", data_x=data["x"], data_t=data["t"], data_u=data["u"],
                            noise_std=data["noise_std"])
        cls, params = TRUE_CLASS[task.family], true_parameters(task)
        write_json(folder/"truth/truth.json", {"true_class": cls, "true_parameters": params,
                                               "benchmark_family": task.family, "task_record": task.to_dict()})
        numbers = ", ".join(f"{k}={v:.4g}" for k, v in params.items())
        (folder/"truth/problem.tex").write_text(
            f"% {case.case_id} ({case.split}): hidden truth, never given to agents\n\\begin{{align*}}\n"
            f"&{TEX[cls]},\\qquad x\\in(0,{task.length:.4g}),\\ t\\in(0,{task.final_time:.4g}],\\\\\n"
            f"&u(0,t)=u(L,t)=0,\\qquad {numbers},\\\\\n&{exact_text(task)}.\n\\end{{align*}}\n")
        x, t = output_nodes_group(task) if task.family.startswith("fpde_") else output_nodes(task)
        exact = reference_group if task.family.startswith("fpde_") else reference
        np.savez_compressed(folder/"truth/reference.npz", x=x, t=t, u=exact(task, x, t))
        lines = [f"# Case `{case.case_id}` ({'training' if case.split == 'evolution' else 'validation'})", "",
                 f"Hidden truth: **{cls}**, {numbers}; L = {task.length:.4g}, T = {task.final_time:.4g}.", "",
                 "Agents see only `input/`. `truth/` is for inspection and was never given to agents.", "",
                 "| Model | Arm | Chosen equation | Coefficients | Method | Relative L2 | Passed |",
                 "| --- | --- | --- | --- | --- | ---: | :---: |"]
        for key, run in runs:
            for arm in ("single", "rsi"):
                result = next(r for r in json.loads((run/arm/"frozen_results.json").read_text())
                              if r["case_id"] == case.case_id)
                target = folder/"outputs"/key/arm
                target.mkdir(parents=True)
                solution = run/arm/"solutions"/f"{case.case_id}.npz"
                if solution.exists():
                    shutil.copyfile(solution, target/"solution.npz")
                shutil.copyfile(run/arm/"programs/frozen/identify.py", target/"solver.py")
                write_json(target/"metrics.json", {**result, "source_run": str(run.relative_to(PROJECT_ROOT))})
                coefficients = ", ".join(f"{k}={v:.4g}" for k, v in (result.get("identified_parameters") or {}).items())
                chosen = result.get("identified_family") or "—"
                mark = " ✓" if result.get("family_correct") else (" ✗" if chosen != "—" else "")
                rel = result["relative_l2"] if result["status"] == "ok" else None
                lines.append(f"| {LABELS[key]} | {'single run' if arm == 'single' else 'RSI'} | {chosen}{mark} | "
                             f"{coefficients or '—'} | {result.get('method') or '—'} | "
                             f"{f'{rel:.2e}' if rel is not None and math.isfinite(rel) else result['status']} | "
                             f"{'yes' if result['passed'] else 'no'} |")
        (folder/"README.md").write_text("\n".join(lines)+"\n")
        index_rows.append(f"| [`{case.case_id}`]({case.case_id}/README.md) | {case.split} | {cls} | {numbers} | "
                          f"{task.length:.3g} | {task.final_time:.3g} |")
    (output/"README.md").write_text("\n".join([
        "# Student-protocol cases", "",
        "Each case hides one true equation with coefficients drawn for it alone. Agents receive `input/` (the case",
        "view, the equation library, and 256 noisy samples), choose the equation and a method (numerical or PINN),",
        "and write the solver from scratch. Outputs are the final single-run and RSI agents of the runs:",
        "", *[f"- `{key}`: `{run.relative_to(PROJECT_ROOT)}`" for key, run in runs], "",
        "| Case | Split | True class | True coefficients | L | T |", "| --- | --- | --- | --- | ---: | ---: |",
        *index_rows, ""]))
    print(output)


if __name__ == "__main__":
    main()
