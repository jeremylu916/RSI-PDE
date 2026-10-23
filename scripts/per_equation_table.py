"""Per-equation single/RSI table (mean relative L2 of the frozen agents) from a consolidated report.

Usage (after sourcing scripts/env.sh):
    python scripts/per_equation_table.py runs/code-agent-results-fine-20261016
"""

import argparse
import csv
import json
from pathlib import Path

FAMS = [("poisson", "Poisson"), ("heat", "Heat"), ("convection_diffusion", "Convection–diffusion"),
        ("burgers", "Burgers"), ("caputo", "Caputo (1-D)"), ("multi_term", "Multi-term Caputo"),
        ("wave2d", "Wave 2-D"), ("fode", "FODE"), ("fpde_caputo", "Caputo FPDE"),
        ("fpde_variable_order", "Var.-order α(t)"), ("fpde_variable_order_xt", "Var.-order α(x,t)")]
MODELS = [("Claude", "Sonnet 4.5"), ("GPT-OSS", "GPT-OSS"), ("GPT-5.5", "GPT-5.5"), ("GPT-6.1", "GPT-6.1"),
          ("Claude Opus 5.5", "Opus 5.5")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    rows = list(csv.DictReader((args.results/"per_family_l2.csv").open()))
    starting, counts, starts = {}, {}, set()
    for name in json.loads((args.results/"protocol_checks.json").read_text()):
        manifest = json.loads((args.results.parent/name/"manifest.json").read_text())
        starts.add({"seed": "classical"}.get(manifest["start"], manifest["start"]))
        for result in json.loads((args.results.parent/name/"single/frozen_results.json").read_text()):
            counts.setdefault(result["family"], set()).add(result["task_id"])
        if manifest["start"] not in ("classical", "seed"):
            continue
        by_family = {}
        for result in json.loads((args.results.parent/name/"starting_reference_results.json").read_text()):
            by_family.setdefault(result["family"], []).append(result)
        for family, items in by_family.items():
            starting[family] = (sum(r["relative_l2"] for r in items)/len(items), sum(r["passed"] for r in items),
                                len(items))

    def get(family, start, model, arm):
        return next((r for r in rows if (r["family"], r["start"], r["model"], r["arm"]) == (family, start, model, arm)),
                    None)

    def cell(family, start, model):
        single, rsi = get(family, start, model, "single"), get(family, start, model, "rsi")
        if single is None or rsi is None:
            return "n/a"
        cases = len(counts[family])

        def fmt(row):
            value, passes = row["mean_relative_l2"], round(float(row["pass_rate"])*cases)
            text = "none" if value in ("", None) else f"{float(value):.1e}"
            return text+("" if passes == cases else f" ({passes}/{cases})"), passes, (
                float(value) if value not in ("", None) else 1e9)
        (st, sp, sv), (rt, rp, rv) = fmt(single), fmt(rsi)
        if (sp, sv) == (rp, rv):
            return f"{st} / {rt}"
        return f"{st} / **{rt}**" if (rp, -rv) > (sp, -sv) else f"**{st}** / {rt}"

    lines = ["Mean relative L2 of the final agent per equation, over all public cases (single run / RSI). The better",
             "arm is in bold; passes are shown only when some cases fail; 'none' means the program produced no finite error (it did not execute).",
             ""]
    for start, title in (("classical", "Classical start"), ("scratch", "Scratch start (no starting solver)")):
        if start not in starts:
            continue
        lines += [f"### {title}", ""]
        lines.append("| Equation | " + ("Starting solver | " if start == "classical" else "")
                     + " | ".join(label for _, label in MODELS) + " |")
        lines.append("|---|" + ("---:|" if start == "classical" else "") + "---:|"*len(MODELS))
        for family, label in FAMS:
            if family not in counts:
                continue
            row = f"| {label} ({len(counts[family])} cases) | "
            if start == "classical" and family in starting:
                value, passes, cases = starting[family]
                row += f"{value:.1e}"+("" if passes == cases else f" ({passes}/{cases})")+" | "
            row += " | ".join(cell(family, start, model) for model, _ in MODELS)+" |"
            lines.append(row)
        lines.append("")
    (args.results/"per_equation_table.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
