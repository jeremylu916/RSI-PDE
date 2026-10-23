"""Validation macro relative L2 and validation passes of the active program after each round (single run vs RSI).

The single run is one request; it is drawn at its frozen level across the RSI arm's rounds.

Usage (after sourcing scripts/env.sh):
    python scripts/plot_convergence.py runs/code-agent-results-pdebench-20261017
Reads per_round.csv from scripts/report_code_agents.py; writes convergence.{png,svg} into the results directory.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

MODELS = [("Claude", "Claude Sonnet 4.5", "#2a78d6", "o"), ("GPT-OSS", "GPT-OSS-120B", "#eb6834", "s"),
          ("GPT-5.5", "GPT-5.5", "#1baf7a", "^"), ("GPT-6.1", "GPT-6.1", "#eda100", "D"),
          ("Claude Opus 5.5", "Claude Opus 5.5", "#e87ba4", "v")]
ARMS = [("single", "Single run", "-"), ("rsi", "RSI", "--")]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e4e3de"
FAILED = 1e5  # the starting stub and rounds without a working program carry the 1e6 failure penalty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--cases", type=int, default=16, help="validation cases per arm")
    parser.add_argument("--x", choices=("round", "minutes"), default="round",
                        help="x axis: improvement round, or elapsed wall-clock minutes (time-budgeted runs)")
    args = parser.parse_args()
    rows = list(csv.DictReader((args.results/"per_round.csv").open()))
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    for axis in (left, right):
        axis.set_facecolor(SURFACE)
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.set_xlabel("Improvement round" if args.x == "round" else "Elapsed wall-clock time (minutes)")
    left.set_yscale("log")
    for key, _, color, marker in MODELS:
        for arm, _, line in ARMS:
            def position(r):
                return int(r["round"]) if args.x == "round" else float(r["elapsed_minutes"])
            points = sorted((position(r), float(r["active_validation_macro_relative_l2"]),
                             float(r["active_validation_pass_rate"])) for r in rows
                            if r["model"] == key and r["arm"] == arm and (args.x == "round" or r.get("elapsed_minutes")))
            if arm == "single" and points:  # the one-shot single run: its frozen level across RSI's range
                span = [position(r) for r in rows if r["model"] == key and r["arm"] == "rsi"
                        and (args.x == "round" or r.get("elapsed_minutes"))]
                points = points[:1]+[(points[-1][0], *points[-1][1:]), (max(span+[points[-1][0]]), *points[-1][1:])]
            working = [(x, y) for x, y, _ in points if y < FAILED]
            style = dict(linestyle=line, color=color, linewidth=2, marker=marker, markersize=6,
                         markeredgecolor=SURFACE, markeredgewidth=1, clip_on=False)
            if args.x == "minutes":  # step plots: the active program changes only at accepted rounds
                style.update(drawstyle="steps-post", marker=None)
            if working:
                left.plot(*zip(*working), **style)
            right.plot([x for x, _, _ in points], [p*args.cases for _, _, p in points], **style)
    left.set_ylabel("Validation macro relative L2 (active program)")
    right.set_ylabel(f"Validation cases passed (of {args.cases})")
    right.set_ylim(-0.5, args.cases+0.5)
    handles = [Line2D([], [], color=color, marker=marker, linewidth=2, markersize=6, markeredgecolor=SURFACE,
                      label=name) for _, name, color, marker in MODELS]
    handles += [Line2D([], [], color=MUTED, linestyle=line, linewidth=2, label=name) for _, name, line in ARMS]
    figure.legend(handles=handles, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 0.0))
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    stem = "convergence" if args.x == "round" else "convergence_minutes"
    for suffix in ("png", "svg"):
        figure.savefig(args.results/f"{stem}.{suffix}", dpi=200, facecolor=SURFACE)
    print(args.results/f"{stem}.png")


if __name__ == "__main__":
    main()
