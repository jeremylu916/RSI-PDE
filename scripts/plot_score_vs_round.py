"""Plot each family's validation relative L2 of the active program against improvement round.

Values come from the runs listed in the results directory's protocol_checks.json: after round k, the
active program is the last accepted candidate (or the starting program), and its family score is the
mean relative L2 over that family's validation cases.

Usage (after sourcing scripts/env.sh):
    python scripts/plot_score_vs_round.py runs/code-agent-results-v2-20261014 --families burgers poisson wave2d
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Categorical slots 1-5 of the reference palette, in fixed model order.
MODELS = [("Claude", "Claude Sonnet 4.5", "#2a78d6", "o"), ("GPT-OSS", "GPT-OSS-120B", "#eb6834", "s"),
          ("GPT-5.5", "GPT-5.5", "#1baf7a", "^"), ("GPT-6.1", "GPT-6.1", "#eda100", "D"),
          ("Claude Opus 5.5", "Claude Opus 5.5", "#e87ba4", "v")]
ARMS = [("single", "Single run", "-"), ("rsi", "RSI", "--")]
FAMILY_LABELS = {"poisson": "Poisson", "heat": "Heat", "convection_diffusion": "Convection-diffusion",
                 "burgers": "Burgers", "caputo": "Caputo", "multi_term": "Multi-term Caputo", "wave2d": "Wave 2D",
                 "fode": "FODE", "fpde_caputo": "Caputo FPDE", "fpde_variable_order": "Var.-order \u03b1(t)",
                 "fpde_variable_order_xt": "Var.-order \u03b1(x,t)",
                 "variable_order_t": "Var.-order \u03b1(t)", "variable_order_xt": "Var.-order \u03b1(x,t)"}
MODEL_KEYS = {"us.anthropic.claude-sonnet-4-5-20250929-v1:0": "Claude", "openai.gpt-oss-120b-1:0": "GPT-OSS",
              "us.openai.gpt-5.5": "GPT-5.5", "us.openai.gpt-6.1-sol": "GPT-6.1",
              "us.anthropic.claude-opus-5-5": "Claude Opus 5.5"}
STARTS = {"seed": "classical", "classical": "classical", "scratch": "scratch"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e4e3de"
FAILED = 1e5  # Rounds without any executing program carry the evaluator's 1e6 failure penalty.


def family_scores(results):
    """Mean validation relative L2 per family of the active program after each round."""
    by_family = {}
    for row in results:
        by_family.setdefault(row["family"], []).append(row["relative_l2"])
    return {family: sum(values)/len(values) for family, values in by_family.items()}


def load(directory):
    rows = []
    for name in json.loads((directory/"protocol_checks.json").read_text()):
        run = directory.parent/name
        manifest = json.loads((run/"manifest.json").read_text())
        for arm in ("single", "rsi"):
            active = family_scores(json.loads((run/arm/"initial_validation.json").read_text()))
            one_shot = (run/arm/"one_shot.json").exists()  # the single run: frozen after its one request
            accepted = {event["iteration"] for event in map(json.loads, (run/arm/"events.jsonl").open())
                        if event.get("gate", {}).get("accepted")}
            for round_index in range(manifest["configuration"]["rounds"]+1):
                if one_shot and round_index == 1:
                    active = family_scores([r for r in json.loads((run/arm/"frozen_results.json").read_text())
                                            if r["split"] == "development"])
                elif round_index in accepted:
                    path = run/arm/f"candidate-validation-{round_index:03d}.json"
                    active = family_scores(json.loads(path.read_text()))
                for family, value in active.items():
                    rows.append({"benchmark": family, "start": STARTS[manifest["start"]],
                                 "model": MODEL_KEYS[manifest["model_config"]["model"]], "arm": arm,
                                 "round": round_index, "active_validation_macro_relative_l2": value})
    return rows


def series(rows, benchmark, start, model, arm):
    points = sorted((int(r["round"]), float(r["active_validation_macro_relative_l2"])) for r in rows
                    if r["benchmark"] == benchmark and r["start"] == start and r["model"] == model and r["arm"] == arm)
    return [(x, y) for x, y in points if y < FAILED]


def style(axis):
    axis.set_facecolor(SURFACE)
    axis.set_yscale("log")
    axis.grid(True, which="major", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.set_xticks(range(7))


def panels_for(families):
    return [(family, start, f"{FAMILY_LABELS[family]}, {start} start")
            for family in families for start in ("classical", "scratch")]


def present(rows, PANELS):
    """Panels with data; the start is dropped from labels when every run used the same start."""
    panels = [panel for panel in PANELS if any((r["benchmark"], r["start"]) == panel[:2] for r in rows)]
    if len({start for _, start, _ in panels}) == 1:
        panels = [(family, start, FAMILY_LABELS[family]) for family, start, _ in panels]
    return panels


def overview(rows, output, PANELS):
    """All models in one panel per condition; arms overlap where results coincide."""
    panels = present(rows, PANELS)
    figure, axes = plt.subplots(max(1, (len(panels)+1)//2), 2, figsize=(10, 3.6*((len(panels)+1)//2)), sharex=True,
                                facecolor=SURFACE, squeeze=False)
    for axis, (benchmark, start, label) in zip(axes.flat, panels):
        style(axis)
        for key, _, color, marker in MODELS:
            for arm, _, line in ARMS:
                points = series(rows, benchmark, start, key, arm)
                if points:
                    axis.plot(*zip(*points), linestyle=line, color=color, linewidth=2, marker=marker,
                              markersize=6, markeredgecolor=SURFACE, markeredgewidth=1, clip_on=False)
        axis.set_ylabel(f"{label}\nvalidation relative L2")
    for axis in axes[-1]:
        axis.set_xlabel("Improvement round")
    handles = [Line2D([], [], color=color, marker=marker, linewidth=2, markersize=6, markeredgecolor=SURFACE,
                      label=name) for _, name, color, marker in MODELS]
    handles += [Line2D([], [], color=MUTED, linestyle=line, linewidth=2, label=name) for _, name, line in ARMS]
    figure.legend(handles=handles, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 0.0))
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    save(figure, output/"score_vs_round_overview")


def small_multiples(rows, output, PANELS):
    """One panel per model and condition, so the single-run and RSI curves are always distinguishable."""
    arms = [("single", "Single run", "-", "#2a78d6", "o"), ("rsi", "RSI", "--", "#eb6834", "s")]
    panels = present(rows, PANELS)
    figure, axes = plt.subplots(len(panels), len(MODELS), figsize=(15, 2.6*len(panels)+0.1), sharex=True,
                                sharey="row", facecolor=SURFACE, squeeze=False)
    for row, (benchmark, start, label) in enumerate(panels):
        for column, (key, name, _, _) in enumerate(MODELS):
            axis = axes[row, column]
            style(axis)
            for arm, _, line, color, marker in arms:
                points = series(rows, benchmark, start, key, arm)
                if points:
                    axis.plot(*zip(*points), linestyle=line, color=color, linewidth=2, marker=marker,
                              markersize=5, markeredgecolor=SURFACE, markeredgewidth=1, clip_on=False)
            if column == 0:
                axis.set_ylabel(f"{label}\nvalidation rel. L2")
            if row == 0:
                axis.xaxis.set_label_position("top")
                axis.set_xlabel(name, fontsize=10)
            elif row == len(panels)-1:
                axis.set_xlabel("Improvement round")
    handles = [Line2D([], [], color=color, linestyle=line, marker=marker, linewidth=2, markersize=6,
                      markeredgecolor=SURFACE, label=name) for _, name, line, color, marker in arms]
    figure.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    save(figure, output/"score_vs_round")


def save(figure, stem):
    for suffix in ("png", "svg"):
        figure.savefig(f"{stem}.{suffix}", dpi=200, facecolor=SURFACE)
    plt.close(figure)
    print(f"{stem}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--families", nargs="+", default=["burgers", "poisson", "wave2d"],
                        choices=sorted(FAMILY_LABELS))
    args = parser.parse_args()
    rows = load(args.results)
    panels = panels_for(args.families)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    small_multiples(rows, args.results, panels)
    overview(rows, args.results, panels)


if __name__ == "__main__":
    main()
