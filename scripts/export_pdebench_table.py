"""Write the per-case single-run vs RSI results of pdebench runs as CSV and HTML files.

Usage (after sourcing scripts/env.sh, once scripts/pdebench_sealed_test.py has run):
    python scripts/export_pdebench_table.py runs/code-agent-results-pdebench-20261017 \
        --runs runs/code-agents-pdebench-*-20261017

Writes into the results directory:
    pdebench_results_long.csv   one row per case, model, and arm (all metrics, identified model)
    pdebench_results_wide.csv   one row per case; relative L2 and pass flag per model and arm
    pdebench_results.html       the wide table, colored by pass/fail, for viewing in a browser
"""

import argparse
import csv
import html
import json
from pathlib import Path

from pde_rsi.pdebench import SPLITS, TITLES, make_cases

MODELS = {"us.anthropic.claude-sonnet-4-5-20250929-v1:0": "Sonnet 4.5", "openai.gpt-oss-120b-1:0": "GPT-OSS",
          "us.openai.gpt-5.5": "GPT-5.5", "us.openai.gpt-6.1-sol": "GPT-6.1", "us.anthropic.claude-opus-5-5": "Opus 5.5"}
ORDER = ["1a", "1b", "1c", "2a", "2b", "2c", "3a", "3b", "3c", "3d", "4a", "4b", "4c", "5", "6", "7", "8", "9", "10", "11"]
SPLIT_NAMES = {"evolution": "training", "development": "validation", "test": "sealed test"}
ARMS = (("single", "single run"), ("rsi", "RSI"))


def variant(case):
    s, p = case.truth.spec, case.truth.problem
    if s["profile"]:
        return {"sine": "sine profile", "polynomial": "polynomial profile", "mixed_sine": "mixed-sine profile"}[s["profile"]]
    pe = (lambda: s["v"]*(s["b"]-s["a"])/s["D"])
    return {"1b": lambda: "D=1", "1c": lambda: f"alpha={s['sharpness']:g}",
            "2b": lambda: f"L={s['b']:g}, D={s['D']:g}", "2c": lambda: f"gamma={s['gamma']:.3g}, D={s['D']:g}",
            "3b": lambda: f"Pe={pe():g}", "3c": lambda: f"Pe={pe():g}", "3d": lambda: f"v={s['v']:g}, D={s['D']:g}",
            "4b": lambda: "D=0.01/pi", "4c": lambda: f"u_l={s['ul']:g}, u_r={s['ur']:g}, D={s['D']:g}",
            "6": lambda: f"alpha={s['alpha']:g}, p={s['p']:g}", "9": lambda: f"alpha={s['alpha']:g}, p={s['p']:g}",
            "10": lambda: f"c={s['c']:g}, p={s['p']:g}", "11": lambda: f"c={s['c']:g}, p={s['p']:g}"}[p]()


def load(run_dirs):
    results, models = {}, []
    for run in run_dirs:
        manifest = json.loads((run/"manifest.json").read_text())
        model = MODELS.get(manifest["model_config"]["model"], manifest["model_config"]["model"])
        models.append(model)
        for arm, _ in ARMS:
            rows = json.loads((run/arm/"frozen_results.json").read_text())
            sealed = run/arm/"sealed_test/results.json"
            rows += json.loads(sealed.read_text()) if sealed.exists() else []
            for row in rows:
                results[(model, arm, row["task_id"])] = row
    return [m for m in MODELS.values() if m in models], results


def cell_text(row):
    if row is None:
        return "", ""
    if row["status"] != "ok":
        return "fail", row["status"]
    return f"{row['relative_l2']:.2e}", ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    models, results = load(sorted(args.runs))
    cases = sorted((c for split in SPLITS for c in make_cases(split)),
                   key=lambda c: (ORDER.index(c.truth.problem), SPLITS.index(c.split)))

    with (args.results/"pdebench_results_long.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["tex_case", "description", "split", "variant", "case_id", "true_class", "model", "arm",
                         "status", "passed", "relative_l2", "final_relative_l2", "initial_error", "boundary_error",
                         "wall_seconds", "identified_class", "class_correct", "identified_parameters", "method"])
        for case in cases:
            for model in models:
                for arm, label in ARMS:
                    r = results.get((model, arm, case.task_id))
                    if r is None:
                        continue
                    ok = r["status"] == "ok"
                    writer.writerow([case.truth.problem, TITLES[case.truth.problem], SPLIT_NAMES[case.split],
                                     variant(case), case.case_id, case.family, model, label, r["status"], r["passed"],
                                     r["relative_l2"] if ok else "", r["final_relative_l2"] if ok else "",
                                     r.get("initial_error", "") if ok else "", r.get("boundary_error", "") if ok else "",
                                     f"{r['wall_seconds']:.2f}", r.get("identified_family") or "", r["family_correct"],
                                     json.dumps(r.get("identified_parameters") or {}), r.get("method") or ""])

    header = ["tex case", "description", "split", "variant", "true class"]
    header += [f"{model} {label} {what}" for model in models for _, label in ARMS for what in ("rel. L2", "passed", "class")]
    with (args.results/"pdebench_results_wide.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for case in cases:
            row = [case.truth.problem, TITLES[case.truth.problem], SPLIT_NAMES[case.split], variant(case), case.family]
            for model in models:
                for arm, _ in ARMS:
                    r = results.get((model, arm, case.task_id))
                    text, _ = cell_text(r)
                    row += [text, "" if r is None else ("yes" if r["passed"] else "no"),
                            "" if r is None else (r.get("identified_family") or "")]
            writer.writerow(row)
        for split in SPLITS:
            subset = [c for c in cases if c.split == split]
            row = ["passed", "", SPLIT_NAMES[split], f"{len(subset)} cases", ""]
            for model in models:
                for arm, _ in ARMS:
                    row += ["", sum(results[(model, arm, c.task_id)]["passed"] for c in subset
                                    if (model, arm, c.task_id) in results), ""]
            writer.writerow(row)

    def td(r):
        if r is None:
            return "<td></td>"
        text, status = cell_text(r)
        kind = "pass" if r["passed"] else ("crash" if status else "miss")
        mark = "" if r["family_correct"] else ' <span class="wrong" title="wrong equation class">✗</span>'
        tip = html.escape(f"chose {r.get('identified_family') or 'nothing'}; method {r.get('method') or '-'}"
                          + (f"; final-time {r['final_relative_l2']:.2e}" if r["status"] == "ok" else f"; {status}"))
        return f'<td class="{kind}" title="{tip}">{text}{mark}</td>'

    head = "".join(f'<th colspan="2">{m}</th>' for m in models)
    sub = "".join("<th>single</th><th>RSI</th>" for _ in models)
    body, previous = [], None
    for case in cases:
        p = case.truth.problem
        first = p != previous
        previous = p
        name = f"<b>{p}</b> {html.escape(TITLES[p])}" if first else ""
        cells = "".join(td(results.get((m, arm, case.task_id))) for m in models for arm, _ in ARMS)
        body.append(f'<tr class="{"first" if first else ""}"><td class="name">{name}</td>'
                    f"<td>{SPLIT_NAMES[case.split]}</td><td>{html.escape(variant(case))}</td>{cells}</tr>")
    for split in SPLITS:
        subset = [c for c in cases if c.split == split]
        cells = "".join(f"<td><b>{sum(results[(m, arm, c.task_id)]['passed'] for c in subset)}</b></td>"
                        for m in models for arm, _ in ARMS)
        body.append(f'<tr class="total"><td class="name"><b>Passed</b></td><td>{SPLIT_NAMES[split]}</td>'
                    f"<td>{len(subset)} cases</td>{cells}</tr>")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>pdebench results</title>
<style>
:root {{ --bg:#fcfcfb; --ink:#1f1f1e; --muted:#6b6a64; --line:#e4e3de; --pass:#d9f2e4; --miss:#fdf0d5; --crash:#f9dcdc; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1b1b1a; --ink:#ecebe6; --muted:#a3a29b; --line:#3a3936;
  --pass:#1f4630; --miss:#4a3b17; --crash:#4d2222; }} }}
body {{ background:var(--bg); color:var(--ink); font:13px/1.4 system-ui, sans-serif; margin:16px; }}
.wrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; white-space:nowrap; }}
th, td {{ border:1px solid var(--line); padding:3px 7px; text-align:right; font-variant-numeric:tabular-nums; }}
th {{ position:sticky; top:0; background:var(--bg); }}
td:nth-child(-n+3) {{ text-align:left; }}
tr.first td {{ border-top:2px solid var(--muted); }}
tr.total td {{ border-top:2px solid var(--muted); }}
td.pass {{ background:var(--pass); font-weight:600; }} td.miss {{ background:var(--miss); }} td.crash {{ background:var(--crash); }}
.wrong {{ color:#c0392b; }}
p {{ color:var(--muted); max-width:70em; }}
</style></head><body>
<div class="wrap"><table>
<thead><tr><th>tex case</th><th>split</th><th>variant</th>{head}</tr><tr><th></th><th></th><th></th>{sub}</tr></thead>
<tbody>{"".join(body)}</tbody></table></div>
<p>Relative space-time L2 error of each frozen agent (single run vs RSI) on every pde_benchmarks.tex case.
Green: passed (space-time and final-time relative L2 both at most 2e-3, exact IC/BC, within 90 s and 16 GB).
Yellow: ran but missed. Red: crashed or invalid output. ✗: wrong equation class identified. Hover a cell for the
chosen class, method, and final-time error. Training and validation cases were used by the improvement loop;
sealed-test cases were scored only after freezing. Opus 5.5 used a 64k output-token cap; the other models used 32k.</p>
</body></html>
"""
    (args.results/"pdebench_results.html").write_text(page)
    for name in ("pdebench_results_long.csv", "pdebench_results_wide.csv", "pdebench_results.html"):
        print(args.results/name)


if __name__ == "__main__":
    main()
