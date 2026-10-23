"""Per-PDE results of frozen agents: pass rate (single vs RSI) and the RSI solvers' time and accuracy per model.

Each tex problem pools its cases over training, validation, and the sealed test (1-3 cases per problem).
Solve time is the in-sandbox solve() time recorded when each frozen program was scored (every arm on its own
dedicated core of the same node type).
Usage (after sourcing scripts/env.sh):
    python scripts/report_per_pde.py runs/code-agent-results-pdebench2h-20261017 \
        --runs runs/code-agents-pdebench2h-*-20261017
Writes per_pde_pass_rate.{md,csv}, per_pde_rsi_time_accuracy.{md,csv}, and per_pde_rsi_heatmap.{png,svg}.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm  # noqa: E402
import numpy as np  # noqa: E402

from pde_rsi.pdebench import RELATIVE_L2_TOLERANCE, TITLES  # noqa: E402

MODELS = {"us.anthropic.claude-sonnet-4-5-20250929-v1:0": "Sonnet 4.5", "openai.gpt-oss-120b-1:0": "GPT-OSS",
          "us.openai.gpt-5.5": "GPT-5.5", "us.openai.gpt-6.1-sol": "GPT-6.1", "us.anthropic.claude-opus-5-5": "Opus 5.5"}
ORDER = ["1a", "1b", "1c", "2a", "2b", "2c", "3a", "3b", "3c", "3d", "4a", "4b", "4c", "5", "6", "7", "8", "9", "10", "11"]
SHORT = {"1a": "Poisson, manufactured", "1b": "Poisson, multiscale", "1c": "Poisson, interior layer",
         "2a": "Heat, manufactured", "2b": "Heat, free decay", "2c": "Heat, periodic source",
         "3a": "Conv.-diff., manufactured", "3b": "Conv.-diff., steady layer", "3c": "Conv.-diff., layer f=1",
         "3d": "Conv.-diff., Gaussian pulse", "4a": "Burgers, manufactured", "4b": "Burgers, -sin(pi x)",
         "4c": "Burgers, viscous shock", "5": "2-D wave", "6": "Fractional ODE", "7": "Caputo, manufactured",
         "8": "Multi-term Caputo", "9": "Caputo, zero IC", "10": "Var. order a(t)", "11": "Var. order a(x,t)"}
# Reference palette: sequential blue ramp; diverging blue <-> red with a gray midpoint.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#a32a29"]
SURFACE, INK, MUTED = "#fcfcfb", "#1f1f1e", "#6b6a64"


def load(run_dirs):
    rows = {}
    for run in run_dirs:
        manifest = json.loads((run/"manifest.json").read_text())
        model = MODELS[manifest["model_config"]["model"]]
        for arm in ("single", "rsi"):
            results = json.loads((run/arm/"frozen_results.json").read_text())
            results += json.loads((run/arm/"sealed_test/results.json").read_text())
            for r in results:
                rows.setdefault((model, arm, r["problem"]), []).append(r)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    rows = load(sorted(args.runs))
    models = [m for m in MODELS.values() if any(k[0] == m for k in rows)]

    # 1. pass rate per PDE, single vs RSI
    lines = ["Cases passed per tex problem (training + validation + sealed test), single run / RSI.",
             "Bold marks the better arm when they differ.", "",
             "| tex case | Cases | " + " | ".join(f"{m} single / RSI" for m in models) + " |",
             "| --- | ---: | " + "---: | "*len(models)]
    table = []
    totals = {(m, a): [0, 0] for m in models for a in ("single", "rsi")}
    for problem in ORDER:
        n = len(rows[(models[0], "single", problem)])
        cells, record = [], {"tex_case": problem, "description": TITLES[problem], "cases": n}
        for m in models:
            s = sum(r["passed"] for r in rows[(m, "single", problem)])
            r_ = sum(r["passed"] for r in rows[(m, "rsi", problem)])
            for arm, value in (("single", s), ("rsi", r_)):
                totals[(m, arm)][0] += value
                totals[(m, arm)][1] += n
            cells.append(f"**{s}** / {r_}" if s > r_ else f"{s} / **{r_}**" if r_ > s else f"{s} / {r_}")
            record[f"{m} single passed"], record[f"{m} RSI passed"] = s, r_
        lines.append(f"| {problem} {SHORT[problem]} | {n} | " + " | ".join(cells) + " |")
        table.append(record)
    lines.append("| **All 49 cases** | 49 | " + " | ".join(
        f"{totals[(m, 'single')][0]} ({totals[(m, 'single')][0]/49:.0%}) / {totals[(m, 'rsi')][0]} "
        f"({totals[(m, 'rsi')][0]/49:.0%})" for m in models) + " |")
    better = {m: (sum(1 for rec in table if rec[f"{m} RSI passed"] > rec[f"{m} single passed"]),
                  sum(1 for rec in table if rec[f"{m} RSI passed"] < rec[f"{m} single passed"])) for m in models}
    lines += ["", "Problems where RSI passes more / fewer cases than the single run: " +
              "; ".join(f"{m} {b}/{w}" for m, (b, w) in better.items()) + "."]
    (args.results/"per_pde_pass_rate.md").write_text("\n".join(lines)+"\n")
    with (args.results/"per_pde_pass_rate.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print("\n".join(lines))

    # 2. RSI solvers: time and accuracy per PDE and model
    time = np.full((len(ORDER), len(models)), np.nan)
    error = np.full((len(ORDER), len(models)), np.nan)
    passed = np.zeros((len(ORDER), len(models)), dtype=int)
    count = np.zeros((len(ORDER), len(models)), dtype=int)
    out = ["Each model's frozen RSI solver per tex problem: median in-sandbox solve time and median relative L2 over",
           "the problem's cases (passed / cases in brackets; 'fail' = no executed case).", "",
           "| tex case | " + " | ".join(f"{m} time | {m} rel. L2" for m in models) + " |",
           "| --- | " + "---: | "*(2*len(models))]
    records = []
    for i, problem in enumerate(ORDER):
        cells, record = [], {"tex_case": problem, "description": TITLES[problem]}
        for j, m in enumerate(models):
            items = rows[(m, "rsi", problem)]
            ok = [r for r in items if r["status"] == "ok" and math.isfinite(r["relative_l2"])]
            passed[i, j], count[i, j] = sum(r["passed"] for r in items), len(items)
            if ok:
                time[i, j] = statistics.median(r["runtime_seconds"] for r in ok)
                error[i, j] = statistics.median(r["relative_l2"] for r in ok)
                cells += [f"{time[i, j]:.2f} s", f"{error[i, j]:.1e} ({passed[i, j]}/{count[i, j]})"]
            else:
                cells += ["fail", f"fail (0/{count[i, j]})"]
            record.update({f"{m} median solve seconds": None if np.isnan(time[i, j]) else round(time[i, j], 4),
                           f"{m} median relative L2": None if np.isnan(error[i, j]) else error[i, j],
                           f"{m} passed": int(passed[i, j]), f"{m} cases": int(count[i, j])})
        out.append(f"| {problem} {SHORT[problem]} | " + " | ".join(cells) + " |")
        records.append(record)
    summary = []
    for j, m in enumerate(models):
        items = [r for p in ORDER for r in rows[(m, "rsi", p)]]
        ok = [r for r in items if r["status"] == "ok"]
        summary.append(f"{m}: total solve time {sum(r['runtime_seconds'] for r in ok):.0f} s over {len(ok)} executed "
                       f"cases, {sum(r['passed'] for r in items)}/49 passed")
    out += ["", "Totals (RSI solvers, 49 cases): " + "; ".join(summary) + "."]
    (args.results/"per_pde_rsi_time_accuracy.md").write_text("\n".join(out)+"\n")
    with (args.results/"per_pde_rsi_time_accuracy.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("\n".join(out))

    # heatmaps: solve time (sequential, log) and relative L2 (diverging around the pass tolerance)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK})
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 8.4), facecolor=SURFACE, sharey=True)
    labels = [f"{p}  {SHORT[p]}" for p in ORDER]
    sequential = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    sequential.set_bad("#e4e3de")
    finite = time[np.isfinite(time)]
    time_norm = Normalize(vmin=0.0, vmax=float(finite.max()))  # linear seconds from zero
    image = left.imshow(time, cmap=sequential, aspect="auto", norm=time_norm)
    figure.colorbar(image, ax=left, fraction=0.04, pad=0.02).set_label("Median solve time (s)")
    diverging = LinearSegmentedColormap.from_list("div", DIVERGING)
    diverging.set_bad("#e4e3de")
    center = math.log10(RELATIVE_L2_TOLERANCE)
    # Color is clipped to 2.5 decades either side of the tolerance so one blow-up cannot wash out the scale;
    # every cell still prints its true value.
    logs = np.clip(np.log10(np.where(np.isfinite(error), np.maximum(error, 1e-300), np.nan)), center-2.5, center+2.5)
    error_norm = TwoSlopeNorm(vcenter=center, vmin=center-2.5, vmax=center+2.5)
    image = right.imshow(logs, cmap=diverging, aspect="auto", norm=error_norm)
    bar = figure.colorbar(image, ax=right, fraction=0.04, pad=0.02)
    ticks = [center-2.5, center-1.5, center-0.5, center, center+0.5, center+1.5, center+2.5]
    bar.set_ticks(ticks, labels=[("<= " if t == ticks[0] else ">= " if t == ticks[-1] else "")+f"{10**t:.0e}" for t in ticks])
    bar.set_label("Median relative L2 (log scale); gray = pass tolerance 2e-3, blue = below, red = above")
    for axis, values, kind in ((left, time, "time"), (right, error, "error")):
        axis.set_xticks(range(len(models)), models)
        axis.set_yticks(range(len(ORDER)), labels)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for i in range(len(ORDER)):
            for j in range(len(models)):
                if np.isnan(values[i, j]):
                    text, rgba = "fail", (0.9, 0.9, 0.9, 1)
                elif kind == "time":
                    text = f"{values[i, j]:.2f}" if values[i, j] < 10 else f"{values[i, j]:.0f}"
                    rgba = sequential(time_norm(values[i, j]))
                else:
                    text = f"{values[i, j]:.0e}\n{passed[i, j]}/{count[i, j]}"
                    rgba = diverging(error_norm(logs[i, j]))
                # text ink from the cell's rendered luminance, so labels stay legible on every step
                luminance = 0.2126*rgba[0]+0.7152*rgba[1]+0.0722*rgba[2]
                axis.text(j, i, text, ha="center", va="center", fontsize=7,
                          color="#ffffff" if luminance < 0.45 else INK)
    left.set_xlabel("RSI solver (model)")
    right.set_xlabel("RSI solver (model)")
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(args.results/f"per_pde_rsi_heatmap.{suffix}", dpi=200, facecolor=SURFACE)
    print(args.results/"per_pde_rsi_heatmap.png")


if __name__ == "__main__":
    main()
