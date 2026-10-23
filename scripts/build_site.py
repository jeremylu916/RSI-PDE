"""Build the RSI-PDEbench website (site/index.html) from site/template.html and the project's aggregate results.

Inputs: the bench spec and splits (counts), data/bench/calibration/summary.json (per-template probe pass rates),
reports/bench-v1/results.json (scripts/bench.py report: per-entrant aggregates). Only aggregates go into the page:
no hidden-test case, seed, input or per-case result. Publish with: python scripts/build_site.py && (gh-pages push).
"""

from datetime import datetime, timezone
import json

from pde_rsi import expanded as X
from pde_rsi.bench.spec import BUDGET, PER_TEMPLATE
from pde_rsi.bench.splits import HARDENED
from pde_rsi.storage import PROJECT_ROOT

SITE = PROJECT_ROOT/"site"
PROBES = [{"key": "v2-rsi-opus55", "label": "RSI harness v2 · Opus 5.5", "short": "RSI Opus", "slot": 1},
          {"key": "claude-code-opus55", "label": "Claude Code · Opus 5.5", "short": "Claude Code", "slot": 2},
          {"key": "codex-gpt55", "label": "Codex · GPT-5.5", "short": "Codex", "slot": 3},
          {"key": "v2-rsi-gpt55", "label": "RSI harness v2 · GPT-5.5", "short": "RSI GPT", "slot": 4}]
# results entrants: (entrant, model) -> legend entry; single runs share their model's harness run
ENTRANTS = [{"key": "pde-rsi-v2/opus55", "label": "RSI v2 · Opus", "slot": 1},
            {"key": "claude-code/opus55", "label": "Claude Code", "slot": 2},
            {"key": "codex/gpt55", "label": "Codex", "slot": 3},
            {"key": "pde-rsi-v2/gpt55", "label": "RSI v2 · GPT", "slot": 4}]
# every template by its equation (the site presents RSI-PDEbench as one benchmark)
NAMES = {"1a": "Poisson, manufactured", "1b": "Poisson, multiscale", "1c": "Poisson, interior layer",
         "2a": "Heat, manufactured", "2b": "Heat, free decay", "2c": "Heat, periodic source",
         "3a": "Conv.–diff., manufactured", "3b": "Conv.–diff., boundary layer", "3c": "Conv.–diff., layer + source",
         "3d": "Conv.–diff., Gaussian pulse", "4a": "Burgers, manufactured", "4b": "Burgers, sine data",
         "4c": "Burgers, viscous shock", "5": "Wave 2-D", "6": "Fractional ODE", "7": "Time-fractional diffusion",
         "8": "Multi-term fractional", "9": "Fractional, zero initial data", "10": "Variable order α(t)",
         "11": "Variable order α(x,t)", "burgers_gauss": "Burgers, Gaussian data", "burgers_random": "Burgers, random data",
         "convdiff": "Conv.–diff., periodic pulse", "darcy": "Darcy flow", "poisson": "Poisson, random forcing",
         "helmholtz": "Helmholtz", "ns": "Navier–Stokes vorticity", "kolmogorov": "Kolmogorov flow",
         "allen_cahn": "Allen–Cahn", "burgers2d": "Stochastic Burgers 2-D", "adr": "Adv.–diff.–reaction",
         "elliptic4d": "Elliptic equation"}
EQUATIONS = ["Poisson", "heat", "convection–diffusion", "Burgers", "2-D wave", "fractional ODE",
             "time-fractional diffusion", "multi-term fractional", "variable-order fractional", "Kuramoto–Sivashinsky",
             "Navier–Stokes", "Kolmogorov flow", "Cahn–Hilliard", "Allen–Cahn", "Maxwell TE_z", "time-harmonic EM",
             "Darcy flow", "Helmholtz", "stochastic Burgers 2-D", "advection–diffusion–reaction", "elliptic equation"]
EARLIER = [
    ["Problems 1–11 (20 cases)", "Opus 5.5", "19.7 · $1.6", "20.0 · $18.9", "19.3 · $1.9", "20.0 · $14.9", "19.7 · $6.3 (Claude Code)"],
    ["Problems 1–11 (20 cases)", "GPT-5.5", "7.7 · $0.4", "9.7 · $17.8", "3.3 · $0.4", "11.0 · $12.2", "12.0 · $14.9 (Codex)"],
    ["data-driven problems (12 cases)", "Opus 5.5", "9.0 · $2.3 *", "11.3 · $14.0", "10.7 *", "12.0 · $13.5", "12.0 · $6.0 (Claude Code)"],
    ["data-driven problems (12 cases)", "GPT-5.5", "5.3 · $0.35", "9.3 · $13.0", "5.3 · $0.4", "9.0 · $5.8", "9.7 · $9.7 (Codex)"],
]


def counts():
    """Hidden-test and validation cases per internal problem group (for case-weighted combination)."""
    n = {"tex": 20, "ext": 10, "inventory": 12}
    return ({k: v*PER_TEMPLATE["test"] for k, v in n.items()}, {k: v*PER_TEMPLATE["val"] for k, v in n.items()},
            sum(n.values()))


def calibration():
    table = json.loads((PROJECT_ROOT/"data/bench/calibration/summary.json").read_text())
    rows = [(track, name, r) for track in ("tex", "inventory") for name, r in table[track].items()]
    total = sum(r["variants"] for _, _, r in rows)
    overall = {p["key"]: sum(r["per_probe"].get(p["key"], 0)*r["variants"] for _, _, r in rows)/total for p in PROBES}
    templates = sorted(({"label": NAMES[name], "variants": r["variants"], "rates": r["per_probe"], "mean": r["mean"],
                         "hardened": name in HARDENED[track]} for track, name, r in rows), key=lambda t: t["mean"])
    return overall, total, templates


def combine(parts, hidden, val):
    """One entrant's case-weighted result over the parts it has finished; the curve holds each part's last solver."""
    n = sum(hidden[p["track"]] for p in parts)
    rate = sum(p["hidden_pass_rate"]*hidden[p["track"]] for p in parts)/n
    vparts = [p for p in parts if p["validation_pass_rate"] is not None]
    vrate = (sum(p["validation_pass_rate"]*val[p["track"]] for p in vparts)/sum(val[p["track"]] for p in vparts)
             if vparts else None)
    rounds = max(p["curve"][-1]["submissions"] for p in parts)

    def at(part, k):
        return next((c["hidden_pass_rate"] for c in reversed(part["curve"]) if c["submissions"] <= k),
                    part["curve"][0]["hidden_pass_rate"])
    curve = [{"submissions": k, "hidden_pass_rate": sum(at(p, k)*hidden[p["track"]] for p in parts)/n}
             for k in range(rounds+1)]
    calls = [(p.get("metered_usage") or {}).get("calls") for p in parts]
    costs = [p.get("cost_usd") for p in parts]
    return {"hidden_pass_rate": rate, "validation_pass_rate": vrate, "dev_test_gap": None if vrate is None else vrate-rate,
            "hidden_cases": n, "rounds": sum(p["rounds"] for p in parts), "curve": curve,
            "calls": None if None in calls else sum(calls), "cost_usd": None if None in costs else sum(costs)}


def results():
    path = PROJECT_ROOT/"reports/bench-v1/results.json"
    if not path.exists():
        return []
    hidden, val, _ = counts()
    groups = {}
    for r in json.loads(path.read_text()):
        groups.setdefault((r["entrant"], r["model"]), []).append(r)
    out = {}
    for (entrant, model), parts in groups.items():
        out[(entrant, model)] = {"entrant": {"pde-rsi-v2": "RSI harness v2", "single run": "single run",
                                             "claude-code": "Claude Code", "codex": "Codex"}[entrant],
                                 "model": {"opus55": "Opus 5.5", "gpt55": "GPT-5.5"}[model],
                                 "key": None if entrant == "single run" else f"{entrant}/{model}",
                                 "parts": sorted(p["track"] for p in parts), **combine(parts, hidden, val)}
    for (entrant, model), row in out.items():
        base = out.get(("single run", model))
        same = base is not None and base["parts"] == row["parts"]  # compare over the same templates only
        row["gain_over_single"] = (row["hidden_pass_rate"]-base["hidden_pass_rate"]
                                   if same and entrant != "single run" else None)
        if entrant == "single run":
            row["curve"] = []
    return list(out.values())


def main():
    budget = [["Improvement wall clock", f"{BUDGET['wall_seconds']//3600} h"],
              ["Model calls", f"{BUDGET['model_calls']}"],
              ["Prompt tokens (input + cache read + cache write)", f"{BUDGET['prompt_tokens']/1e6:.0f} M"],
              ["Output tokens", f"{BUDGET['output_tokens']/1e3:.0f} k"],
              ["Validation submissions", f"{BUDGET['validation_submissions']}"],
              ["Solver sandbox", "1 core, a GPU share, the case's time limit"]]
    overall, calibrated, templates = calibration()
    hidden, val, n_templates = counts()
    data = {"bench_cases": len(X.tex_cases()+X.ext_cases()+X.inventory_cases()+X.manufactured_tasks()),
            "templates_total": n_templates, "hidden_total": sum(hidden.values()), "val_total": sum(val.values()),
            "equations": EQUATIONS, "budget": budget, "probes": PROBES, "calibration": overall,
            "calibrated_cases": calibrated, "templates": templates, "entrants": ENTRANTS, "results": results(),
            "planned_entrants": 6, "earlier": EARLIER,
            "built": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    page = (SITE/"template.html").read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    (SITE/"index.html").write_text(page)
    (SITE/".nojekyll").write_text("")
    print(f"site/index.html: {len(page)/1e3:.0f} kB, {len(data['results'])} result rows")


if __name__ == "__main__":
    main()
