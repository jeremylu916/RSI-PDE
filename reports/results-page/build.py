"""Build the PDE-RSI results and benchmark reference page from data.json and the benchmark generators.

Run from the PDE-RSI root after sourcing scripts/env.sh (after collect.py):
    python reports/results-page/build.py
Writes reports/results-page/pde-rsi-results.html (published as the "PDE-RSI Benchmark Results" artifact).
"""

import html
import json
import math
from pathlib import Path

from pde_rsi import inventory as inv
from pde_rsi import pdebench as pb

HERE = Path(__file__).parent
D = json.loads((HERE/"data.json").read_text())
e = html.escape

TEX_NAMES = {"1a": "Poisson, manufactured", "1b": "Poisson, multiscale", "1c": "Poisson, interior layer",
             "2a": "Heat, manufactured", "2b": "Heat, free decay", "2c": "Heat, periodic source",
             "3a": "Conv.–diff., manufactured", "3b": "Conv.–diff., boundary layer",
             "3c": "Conv.–diff., layer + source", "3d": "Conv.–diff., advected Gaussian",
             "4a": "Burgers, manufactured", "4b": "Burgers, sine data", "4c": "Burgers, viscous shock",
             "5": "2-D wave", "6": "Fractional ODE", "7": "Time-fractional diffusion",
             "8": "Multi-term time-fractional", "9": "Time-fractional, zero IC", "10": "Variable order α(t)",
             "11": "Variable order α(x,t)"}
TEX_ORDER = list(TEX_NAMES)
INV = [("poisson", "Poisson"), ("helmholtz", "Helmholtz"), ("darcy", "Darcy"), ("elliptic4d", "4-D elliptic"),
       ("burgers_random", "Burgers, random IC"), ("burgers_gauss", "Burgers, Gaussian IC"),
       ("convdiff", "Linear conv.–diff."), ("ns", "Navier–Stokes"), ("kolmogorov", "Kolmogorov"),
       ("allen_cahn", "Allen–Cahn"), ("adr", "ADR"), ("burgers2d", "2-D Burgers")]
LABELS = {"v1 single run + repair": "v1 single + repair", "v2 single run + repair": "v2 single + repair",
          "single run + repair": "single + repair"}


def cell(passed, total, strong=False):
    p = passed/total if total else 0
    return f'<td class="c{" best" if strong else ""}" style="--p:{p:.3f}">{passed}<span class="of">/{total}</span></td>'


def find(entries, model, config):
    return next(x for x in entries if (x["model"], x["config"]) == (model, config))


def per_problem_table(entries, columns, problems, names, n_per):
    head = "".join(f'<th scope="col"><span class="m">{e(m)}</span>{e(LABELS.get(c, c))}</th>' for m, c in columns)
    body = []
    for key in problems:
        values = []
        for m, c in columns:
            entry = find(entries, m, c)
            n = entry.get("problems_n", {}).get(key, n_per)
            values.append((entry["problems"].get(key, 0), n))
        best = {}
        for model in dict.fromkeys(m for m, _ in columns):
            idx = [i for i, (m, _) in enumerate(columns) if m == model]
            top = max(values[i][0] for i in idx)
            best.update({i: values[i][0] == top and top > 0 and sum(values[j][0] == top for j in idx) < len(idx)
                         for i in idx})
        body.append(f'<tr><th scope="row">{e(names[key])}</th>'
                    + "".join(cell(v, n, best[i]) for i, (v, n) in enumerate(values)) + "</tr>")
    totals = "".join(f'<td class="tot">{find(entries, m, c)["total"]}<span class="of">/{find(entries, m, c)["cases"]}</span></td>'
                     for m, c in columns)
    body.append(f'<tr class="total"><th scope="row">Total</th>{totals}</tr>')
    return (f'<div class="scroll"><table class="grid"><thead><tr><th scope="col">PDE</th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def summary_table(entries, configs, per_run_cases, caption):
    rows = []
    for model in ("Opus 5.5", "GPT-5.5"):
        cells = []
        for c, _ in configs:
            c = c.replace("NATIVE", "Claude Code" if model == "Opus 5.5" else "Codex")
            x = next((y for y in entries if (y["model"], y["config"]) == (model, c)), None)
            if x is None:
                cells.append('<td class="na">—</td>')
                continue
            mean, cost, runs = x["total"]/3, x.get("cost"), x.get("per_run", "")
            detail = " · ".join(p for p in (runs, f"${cost:.2f}" if cost is not None else "") if p)
            cells.append(f'<td><div class="score"><span class="bar" style="--p:{mean/per_run_cases:.3f}"></span>'
                         f'<b>{mean:.1f}</b><span class="of">/{per_run_cases}</span></div><div class="sub">{e(detail)}</div></td>')
        rows.append(f'<tr><th scope="row">{model}</th>{"".join(cells)}</tr>')
    head = "".join(f'<th scope="col">{label}</th>' for _, label in configs)
    return (f'<div class="scroll"><table class="summary"><caption>{caption}</caption><thead><tr><th scope="col">Model</th>'
            f'{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


FRESH_CONFIGS = [("v1 single run", "Single · v1"), ("v1 single run + repair", "Single + repair · v1"),
                 ("v1 RSI", "RSI · v1"), ("v2 single run", "Single · v2"),
                 ("v2 single run + repair", "Single + repair · v2"), ("v2 RSI", "RSI · v2"), ("NATIVE", "Native agent")]
EARLY_CONFIGS = [("single run", "Single"), ("single run + repair", "Single + repair"), ("RSI", "RSI · v1"),
                 ("NATIVE", "Native agent")]


def fresh_columns():
    cols = []
    for model, native in (("Opus 5.5", "Claude Code"), ("GPT-5.5", "Codex")):
        cols += [(model, "v1 single run + repair"), (model, "v1 RSI"), (model, "v2 single run + repair"),
                 (model, "v2 RSI"), (model, native)]
    return cols


def early_columns():
    cols = []
    for model, native in (("Opus 5.5", "Claude Code"), ("GPT-5.5", "Codex")):
        cols += [(model, "single run + repair"), (model, "RSI"), (model, native)]
    return cols


def other_tex():
    variants = [("pdebench", "6 rounds"), ("pdebench2h", "2 h"), ("pdebench2h-noise100", "2 h, 100 % noise"),
                ("pdebench2h-hiddentime", "2 h, hidden time dependence")]
    head = "".join(f'<th scope="col" colspan="2">{e(label)}</th>' for _, label in variants)
    sub = "".join('<th scope="col">Single</th><th scope="col">RSI</th>' for _ in variants)
    rows = []
    for model in ["Opus 5.5", "GPT-6.1", "GPT-5.5", "Sonnet 4.5", "GPT-OSS"]:
        cells = []
        for key, _ in variants:
            d = {(r["arm"], r["split"]): r for r in D["other"][key] if r["model"] == model}
            for arm in ("single", "rsi"):
                v = [d.get((arm, s), {}).get("passes", 0) for s in ("evolution", "development", "test")]
                cells.append(f'<td class="c" style="--p:{sum(v)/49:.3f}">{v[0]}·{v[1]}·{v[2]}</td>')
        rows.append(f'<tr><th scope="row">{model}</th>{"".join(cells)}</tr>')
    return (f'<div class="scroll"><table class="grid"><thead><tr><th scope="col" rowspan="2">Model</th>{head}</tr>'
            f'<tr>{sub}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


NAMES = {"Claude Opus 5.5": "Opus 5.5", "GPT-6.1": "GPT-6.1", "GPT-5.5": "GPT-5.5", "Claude": "Sonnet 4.5", "GPT-OSS": "GPT-OSS"}


def other_student():
    rows = []
    for key, label in NAMES.items():
        r = {x["arm"]: x for x in D["other"]["student"] if x["model"] == key}
        if not r:
            continue

        def c(arm):
            x = r[arm]
            l2 = f'{float(x["relative_l2"]):.1e}' if x["relative_l2"] and float(x["relative_l2"]) < 1e5 else "fails"
            return (f'<td class="c" style="--p:{int(x["passes"])/16:.3f}">{x["passes"]}<span class="of">/16</span></td>'
                    f'<td class="l2">{l2}</td>')
        rows.append(f'<tr><th scope="row">{label}</th>{c("single")}{c("rsi")}</tr>')
    return ('<div class="scroll"><table class="grid"><thead><tr><th scope="col" rowspan="2">Model</th>'
            '<th scope="col" colspan="2">Single run</th><th scope="col" colspan="2">RSI</th></tr><tr><th scope="col">Passed</th>'
            '<th scope="col">Mean rel. L2</th><th scope="col">Passed</th><th scope="col">Mean rel. L2</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def other_fine():
    groups = [("poisson", "1-D (48 cases)"), ("wave2d", "2-D wave (8)"), ("fode", "Group FPDE (16)")]
    head = "".join(f'<th scope="col" colspan="4">{label}</th>' for _, label in groups)
    sub = "".join('<th scope="col">Classical: single</th><th scope="col">RSI</th><th scope="col">Scratch: single</th>'
                  '<th scope="col">RSI</th>' for _ in groups)
    rows = []
    for key, label in NAMES.items():
        cells = []
        for prefix, _ in groups:
            for start in ("classical", "scratch"):
                for arm in ("single", "rsi"):
                    r = next((x for x in D["other"]["fine"] if x["model"] == key and x["start"] == start and x["arm"] == arm
                              and x["families"].startswith(prefix)), None)
                    cells.append('<td class="na">—</td>' if r is None else
                                 f'<td class="c" style="--p:{int(r["passes"])/int(r["cases"]):.3f}">{r["passes"]}'
                                 f'<span class="of">/{r["cases"]}</span></td>')
        rows.append(f'<tr><th scope="row">{label}</th>{"".join(cells)}</tr>')
    return (f'<div class="scroll"><table class="grid"><thead><tr><th scope="col" rowspan="2">Model</th>{head}</tr><tr>{sub}</tr>'
            f'</thead><tbody>{"".join(rows)}</tbody></table></div>')


def diagnostic():
    rows = "".join(f'<tr><th scope="row">v1 run {d["run"]}</th><td>{d["characters"]:,}</td>'
                   f'<td class="c" style="--p:{d["public"]/24:.3f}">{d["public"]}<span class="of">/24</span></td>'
                   f'<td class="c" style="--p:{d["fresh"]/12:.3f}">{d["fresh"]}<span class="of">/12</span></td></tr>'
                   for d in D["diagnostic"])
    return ('<div class="scroll"><table class="grid"><thead><tr><th scope="col">Rejected v1 answer</th>'
            '<th scope="col">Program characters</th><th scope="col">Training + validation</th>'
            f'<th scope="col">Fresh hidden split</th></tr></thead><tbody>{rows}</tbody></table></div>')


def fmt(v):
    if isinstance(v, float):
        return "π" if abs(v-math.pi) < 1e-12 else f"{v:.4g}"
    return str(v)


def literature_numbers():
    keys = {"1b": ["D"], "1c": ["D", "sharpness", "x0"], "2b": ["b", "D"], "2c": ["T", "D", "gamma"], "3b": ["v", "D"],
            "3c": ["v", "D"], "3d": ["b", "v", "D", "x0", "sigma0"], "4b": ["D"], "4c": ["b", "ul", "ur", "D"],
            "6": ["alpha", "p"], "9": ["alpha", "p"], "10": ["c", "p"], "11": ["c", "p"]}
    pretty = {"sharpness": "α", "gamma": "γ", "sigma0": "σ₀", "x0": "x₀", "b": "L", "ul": "u_ℓ", "ur": "u_r", "alpha": "α"}
    splits = [("evolution", "Training"), ("development", "Validation"), ("test", "Sealed test"), ("test2", "Sealed test2"),
              ("test3", "Fresh test3")]
    rows = []
    for case, names in keys.items():
        cells = []
        for split, _ in splits:
            entry = next((v for p, v in pb.FIXED[split] if p == case), None)
            if entry is None:
                cells.append('<td class="na">—</td>')
                continue
            text = (f"D = {entry['D']*math.pi:.3g}/π" if case == "4b" else
                    ", ".join(f"{pretty.get(k, k)} = {fmt(entry[k])}" for k in names if k in entry))
            cells.append(f"<td>{e(text)}</td>")
        rows.append(f'<tr><th scope="row">{case} {e(TEX_NAMES[case])}</th>{"".join(cells)}</tr>')
    head = "".join(f'<th scope="col">{label}</th>' for _, label in splits)
    return (f'<div class="scroll"><table class="ref"><thead><tr><th scope="col">Case</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def inventory_numbers():
    splits = [("evolution", "Training"), ("development", "Validation"), ("inventory_test", "Sealed test"),
              ("inventory_test2", "Fresh test2")]

    def values(family, split):
        s = inv.make_spec(family, split)
        k = s["kind"]
        if k == "burgers_gauss":
            return f"D = {s['D']:g}"
        if k == "convdiff":
            return f"x₀ = {s['x0']:g}, σ₀ = {s['sigma0']:g}"
        if k in ("burgers_random", "darcy", "poisson", "helmholtz", "ns"):
            return f"DiffusionPDE instance {inv.INSTANCE[split]}"
        if k == "allen_cahn":
            return f"λ = {s['lam']:.3f}"
        if k == "burgers2d":
            return f"c_x = {s['cx']:.3f}, c_y = {s['cy']:.3f}"
        if k == "adr":
            return f"c_x = {s['cx']:.3f}, c_y = {s['cy']:.3f}, λ = {s['lam']:.3f}"
        return {"kolmogorov": "new ω₀ draw", "elliptic4d": "new â draw"}.get(k, "")
    rows = "".join(f'<tr><th scope="row">{e(name)}</th>' + "".join(f"<td>{e(values(family, s))}</td>" for s, _ in splits)
                   + "</tr>" for family, name in INV)
    head = "".join(f'<th scope="col">{label}</th>' for _, label in splits)
    return (f'<div class="scroll"><table class="ref"><thead><tr><th scope="col">Family</th>{head}</tr></thead>'
            f'<tbody>{rows}</tbody></table></div>')


replacements = {
    "{{FRESH_SUMMARY_TEX}}": summary_table(D["fresh"]["pdebench"], FRESH_CONFIGS, 20,
                                           "pde_benchmarks.tex · fresh hidden split test3 · 20 cases per run"),
    "{{FRESH_SUMMARY_INV}}": summary_table(D["fresh"]["inventory"], FRESH_CONFIGS, 12,
                                           "PDE Data Inventory · fresh hidden split inventory_test2 · 12 cases per run"),
    "{{EARLY_SUMMARY_TEX}}": summary_table(D["earlier"]["pdebench"], EARLY_CONFIGS, 37,
                                           "pde_benchmarks.tex · sealed splits test + test2 · 37 cases per run"),
    "{{EARLY_SUMMARY_INV}}": summary_table(D["earlier"]["inventory"], EARLY_CONFIGS, 12,
                                           "PDE Data Inventory · sealed split inventory_test · 12 cases per run"),
    "{{FRESH_TEX}}": per_problem_table(D["fresh"]["pdebench"], fresh_columns(), TEX_ORDER, TEX_NAMES, 3),
    "{{FRESH_INV}}": per_problem_table(D["fresh"]["inventory"], fresh_columns(), [k for k, _ in INV], dict(INV), 3),
    "{{EARLY_TEX}}": per_problem_table(D["earlier"]["pdebench"], early_columns(), TEX_ORDER, TEX_NAMES, 6),
    "{{EARLY_INV}}": per_problem_table(D["earlier"]["inventory"], early_columns(), [k for k, _ in INV], dict(INV), 3),
    "{{OTHER_TEX}}": other_tex(), "{{OTHER_STUDENT}}": other_student(), "{{OTHER_FINE}}": other_fine(),
    "{{DIAGNOSTIC}}": diagnostic(), "{{LIT_NUMBERS}}": literature_numbers(), "{{INV_NUMBERS}}": inventory_numbers(),
}
page = (HERE/"template.html").read_text()
for key, value in replacements.items():
    page = page.replace(key, value)
assert "{{" not in page, "unfilled placeholder"
(HERE/"pde-rsi-results.html").write_text(page)
print(len(page))
