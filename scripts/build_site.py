"""Build the RSI-PDE Bench website (site/index.html) from site/template.html and the project's aggregate results.

Inputs: the bench spec and splits (counts), data/bench/calibration/summary.json (per-template probe pass rates),
reports/bench-v1/results.json (scripts/bench.py report: per-entrant aggregates). Only aggregates go into the page:
no hidden-test case, seed, input or per-case result. Publish with: python scripts/build_site.py && (gh-pages push).
"""

from datetime import datetime, timezone
import json

from pde_rsi.bench.spec import BUDGET, PER_TEMPLATE, TRACKS
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
PROBLEMS = {
    "tex": ["Poisson", "heat", "convection–diffusion", "Burgers", "2-D wave", "fractional ODE", "Caputo diffusion",
            "multi-term fractional", "variable-order fractional"],
    "ext": ["Kuramoto–Sivashinsky", "Navier–Stokes 2-D", "Cahn–Hilliard", "Maxwell TE_z", "time-harmonic EM"],
    "inventory": ["Burgers (2 IC families)", "convection–diffusion", "Darcy", "Poisson", "Helmholtz", "Navier–Stokes",
                  "Kolmogorov flow", "Allen–Cahn", "stochastic Burgers 2-D", "advection–diffusion–reaction",
                  "4-D elliptic"],
}
LABELS = {"burgers_gauss": "Burgers (Gauss)", "burgers_random": "Burgers (rand.)", "convdiff": "conv.–diff.",
          "darcy": "Darcy", "poisson": "Poisson", "helmholtz": "Helmholtz", "ns": "Navier–Stokes",
          "kolmogorov": "Kolmogorov", "allen_cahn": "Allen–Cahn", "burgers2d": "Burgers 2-D", "adr": "ADR",
          "elliptic4d": "elliptic 4-D"}
EARLIER = [
    ["test3 (20)", "Opus 5.5", "19.7 · $1.6", "20.0 · $18.9", "19.3 · $1.9", "20.0 · $14.9", "19.7 · $6.3 (Claude Code)"],
    ["test3 (20)", "GPT-5.5", "7.7 · $0.4", "9.7 · $17.8", "3.3 · $0.4", "11.0 · $12.2", "12.0 · $14.9 (Codex)"],
    ["inventory_test2 (12)", "Opus 5.5", "9.0 · $2.3 *", "11.3 · $14.0", "10.7 *", "12.0 · $13.5", "12.0 · $6.0 (Claude Code)"],
    ["inventory_test2 (12)", "GPT-5.5", "5.3 · $0.35", "9.3 · $13.0", "5.3 · $0.4", "9.0 · $5.8", "9.7 · $9.7 (Codex)"],
]


def tracks():
    out = {}
    counts = {"tex": 20, "ext": 10, "inventory": 12}
    for key, track in TRACKS.items():
        n = counts[key]
        out[key] = {"title": track["title"], "templates": n, "train": n*PER_TEMPLATE["train"],
                    "val": n*PER_TEMPLATE["val"], "test": n*PER_TEMPLATE["test"], "problems": PROBLEMS[key]}
    return out


def calibration():
    table = json.loads((PROJECT_ROOT/"data/bench/calibration/summary.json").read_text())
    overall, cases, templates = {}, {}, {}
    for track in ("tex", "inventory"):
        rows = table[track]
        total = sum(r["variants"] for r in rows.values())
        cases[track] = total
        overall[track] = {p["key"]: sum(r["per_probe"].get(p["key"], 0)*r["variants"] for r in rows.values())/total
                          for p in PROBES}
        templates[track] = [{"name": name, "label": LABELS.get(name, name), "variants": r["variants"], "rates": r["per_probe"], "mean": r["mean"],
                             "hardened": name in HARDENED[track]} for name, r in rows.items()]
    return overall, cases, templates


def results():
    path = PROJECT_ROOT/"reports/bench-v1/results.json"
    if not path.exists():
        return []
    rows = []
    for r in json.loads(path.read_text()):
        entrant = "pde-rsi-v2" if r["entrant"] == "pde-rsi-v2" else r["entrant"]
        key = None if entrant == "single run" else f"{entrant}/{r['model']}"
        rows.append({"track": r["track"], "model": {"opus55": "Opus 5.5", "gpt55": "GPT-5.5"}[r["model"]],
                     "entrant": {"pde-rsi-v2": "RSI harness v2", "single run": "single run"}.get(entrant, {
                         "claude-code": "Claude Code", "codex": "Codex"}.get(entrant, entrant)),
                     "key": key, "hidden_pass_rate": r["hidden_pass_rate"], "gain_over_single": r["gain_over_single"],
                     "validation_pass_rate": r["validation_pass_rate"], "dev_test_gap": r["dev_test_gap"],
                     "rounds": r["rounds"], "calls": (r.get("metered_usage") or {}).get("calls"),
                     "cost_usd": r.get("cost_usd"),
                     "curve": [{"submissions": p["submissions"], "hidden_pass_rate": p["hidden_pass_rate"]}
                               for p in r["curve"]] if key else []})
    return rows


def main():
    overall, cases, templates = calibration()
    budget = [["Improvement wall clock", f"{BUDGET['wall_seconds']//3600} h"],
              ["Model calls", f"{BUDGET['model_calls']}"],
              ["Prompt tokens (input + cache read + cache write)", f"{BUDGET['prompt_tokens']/1e6:.0f} M"],
              ["Output tokens", f"{BUDGET['output_tokens']/1e3:.0f} k"],
              ["Validation submissions", f"{BUDGET['validation_submissions']}"],
              ["Solver sandbox", "1 core, a GPU share, the case's time limit"]]
    data = {"tracks": tracks(), "budget": budget, "probes": PROBES, "calibration": overall,
            "calibration_cases": cases, "templates": templates, "entrants": ENTRANTS, "results": results(),
            "planned_runs": 18, "earlier": EARLIER,
            "results_note": "One seed per entrant (2026-10-23). Each harness run also yields the single-run entrant "
                            "for its model.",
            "built": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    page = (SITE/"template.html").read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    (SITE/"index.html").write_text(page)
    (SITE/".nojekyll").write_text("")
    print(f"site/index.html: {len(page)/1e3:.0f} kB, {len(data['results'])} result rows")


if __name__ == "__main__":
    main()
