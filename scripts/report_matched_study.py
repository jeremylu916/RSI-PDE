"""Matched-model study: single run vs RSI harness vs native coding agent (3 runs each).

The single run is one model request whose solver is frozen as returned; RSI and the native agent improve for up
to 2 h. pdebench: primary metric the fresh sealed split (test2, 20 cases, untouched before the study), secondary
the original sealed split (test, 17 cases). inventory: its one sealed split (12 cases). Also development cost,
final solver time, and failure modes.
Usage (after sourcing scripts/env.sh, once every run is scored):
    python scripts/report_matched_study.py --output runs/matched-study-results-20261018
    python scripts/report_matched_study.py --suite inventory --stamp 20261020 --output runs/matched-inv-study-results-20261020
"""

import argparse
import json
import math
import statistics
from pathlib import Path

from pde_rsi.costs import load_prices, response_cost
from pde_rsi.storage import PROJECT_ROOT, project_path

WAVES = {"opus55": ("Claude Opus 5.5", "us.anthropic.claude-opus-5-5", "claude-code", "Claude Code"),
         "gpt55": ("GPT-5.5", "us.openai.gpt-5.5", "codex", "Codex")}
CONFIGS = (("single", "Single run (one request)"), ("rsi", "Your RSI"), ("native", "Native coding agent"))
# Per suite: (folder in a harness arm, file in a native run's final/, label, cases); the first is primary.
SPLITS = {"pdebench": (("sealed_test2", "sealed_test2_results.json", "Fresh sealed", 20),
                       ("sealed_test", "sealed_test_results.json", "Original sealed", 17)),
          "inventory": (("sealed_test", "sealed_inventory_test_results.json", "Sealed", 12),)}
VALIDATION_CASES = {"pdebench": 16, "inventory": 12}


def macro(rows):
    """Macro relative L2 over classes (mean of class means), as in the gate."""
    families = {}
    for r in rows:
        families.setdefault(r.get("family") or r.get("true_family"), []).append(min(r["relative_l2"], 1e6))
    return statistics.mean(statistics.mean(v) for v in families.values())


def geometric(rows):
    ok = [r["relative_l2"] for r in rows if r["status"] == "ok" and math.isfinite(r["relative_l2"])]
    return math.exp(statistics.mean(math.log(max(v, 1e-300)) for v in ok)) if ok else None


def snapped(rows):
    """Executed cases whose reported coefficients are all 'round' (<= 2 significant digits): coefficient snapping."""
    count = 0
    for r in rows:
        values = list((r.get("identified_parameters") or {}).values())
        if values and all(float(f"{v:.2g}") == v for v in values if isinstance(v, (int, float))):
            count += 1
    return count


def harness_run(run, arm, model, prices, splits):
    manifest = json.loads((run/"manifest.json").read_text())
    report = manifest["arms"][arm]
    frozen = json.loads((run/arm/"frozen_results.json").read_text())
    sealed = [json.loads((run/arm/folder/"results.json").read_text()) for folder, *_ in splits]
    events = [json.loads(line) for line in (run/arm/"events.jsonl").open()]
    cost = sum(response_cost(p, model, prices) or 0 for p in (run/arm).glob("model-call-*/response.json"))
    gates = [e for e in events if "gate" in e]
    if report.get("one_shot"):
        return {"sealed": sealed, "public": frozen, "cost": cost,
                "solve_seconds": sum(r["runtime_seconds"] for r in sum(sealed, [])+frozen if r["status"] == "ok"),
                "submissions": 0, "accepted": 0, "invalid": int(report["one_shot_status"] != "ok"),
                "rejected_reasons": [], "stop": "one request", "wall_minutes": report["budget"]["model_wall_seconds"]/60,
                "retained_equals_last": None}
    return {"sealed": sealed, "public": frozen, "cost": cost,
            "solve_seconds": sum(r["runtime_seconds"] for r in sum(sealed, [])+frozen if r["status"] == "ok"),
            "submissions": len(gates), "accepted": sum(e["gate"]["accepted"] for e in gates),
            "invalid": sum(e.get("event") in ("invalid_proposal", "model_failure") for e in events),
            "rejected_reasons": [r.split(":")[0] for e in gates if not e["gate"]["accepted"] for r in e["gate"]["reasons"]],
            "stop": next((e["event"] for e in reversed(events) if e.get("event") in
                          ("time_budget_reached", "spending_cap_reached", "validation_budget_reached")), "rounds"),
            "wall_minutes": report["budget"]["elapsed_wall_seconds"]/60, "retained_equals_last": None}


def native_run(run, splits):
    manifest = json.loads((run/"manifest.json").read_text())
    final = run/"final"
    frozen = json.loads((final/"frozen_results.json").read_text())
    sealed = [json.loads((final/name).read_text()) for _, name, *_ in splits]
    submissions = [json.loads(line) for line in (run/"submissions.jsonl").open()] if (run/"submissions.jsonl").exists() else []
    return {"sealed": sealed, "public": frozen, "cost": manifest["model_cost_usd"],
            "solve_seconds": sum(r["runtime_seconds"] for r in sum(sealed, [])+frozen if r["status"] == "ok"),
            "submissions": len(submissions), "accepted": sum(s["accepted"] for s in submissions),
            "invalid": manifest["evaluation_counts"]["rejected_by_policy"],
            "rejected_reasons": [r.split(":")[0] for s in submissions if not s["accepted"] for r in s["reasons"]],
            "stop": manifest["stop_reason"], "wall_minutes": manifest["improvement_wall_seconds"]/60,
            "retained_equals_last": manifest["retained_equals_last_file"]}


def spread(values, fmt="{:.1f}"):
    values = [v for v in values if v is not None]
    if not values:
        return "—"
    if len(values) == 1:
        return fmt.format(values[0])
    return f"{fmt.format(statistics.mean(values))} ± {fmt.format(statistics.stdev(values))}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=tuple(SPLITS), default="pdebench")
    parser.add_argument("--stamp", default="20261018")
    args = parser.parse_args()
    splits, tag = SPLITS[args.suite], "inv-" if args.suite == "inventory" else ""
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    runs = PROJECT_ROOT/"runs"
    table, detail, records = [], [], []
    for wave, (label, model, agent, agent_label) in WAVES.items():
        for key, name in CONFIGS:
            items = []
            for r in (1, 2, 3):
                if key == "native":
                    path = runs/f"matched-{tag}{wave}-{agent}-r{r}-{args.stamp}"
                    if (path/"manifest.json").exists() and "final" in json.loads((path/"manifest.json").read_text()):
                        items.append(native_run(path, splits))
                else:
                    path = runs/f"matched-{tag}{wave}-harness-r{r}-{args.stamp}"
                    manifest = json.loads((path/"manifest.json").read_text()) if (path/"manifest.json").exists() else {}
                    one_shot = (manifest.get("arms", {}).get(key) or {}).get("one_shot") or key != "single"
                    if one_shot and all((path/key/folder/"results.json").exists() for folder, *_ in splits):
                        items.append(harness_run(path, key, model, prices, splits))
            if not items:
                continue
            harness = name if key != "native" else f"Native {agent_label}"
            passed = [[sum(r["passed"] for r in i["sealed"][k]) for i in items] for k in range(len(splits))]
            table.append(f"| {label} | {harness} | {len(items)} | {spread(passed[0])} / {splits[0][3]} | "
                         f"{spread([macro(i['sealed'][0]) for i in items], '{:.2e}')} | "
                         + "".join(f"{spread(passed[k])} / {splits[k][3]} | " for k in range(1, len(splits)))
                         + f"{spread([i['cost'] for i in items], '${:.2f}')} | "
                         f"{spread([i['solve_seconds'] for i in items], '{:.0f} s')} |")
            for n, i in enumerate(items, 1):
                reasons = {}
                for reason in i["rejected_reasons"]:
                    reasons[reason] = reasons.get(reason, 0)+1
                detail.append(f"| {label} | {harness} | {n} | "
                              + "".join(f"{sum(r['passed'] for r in rows)}/{cases} | "
                                        for rows, (*_, cases) in zip(i["sealed"], splits))
                              + f"{sum(r['passed'] for r in i['public'] if r['split'] == 'development')}/"
                              f"{VALIDATION_CASES[args.suite]} | "
                              f"{geometric(i['sealed'][0]) or float('nan'):.1e} | {i['cost']:.2f} | {i['wall_minutes']:.0f} | "
                              f"{i['stop']} | {i['accepted']}/{i['submissions']} | {i['invalid']} | "
                              f"{', '.join(f'{k} x{v}' for k, v in sorted(reasons.items())) or '—'} | "
                              f"{snapped(sum(i['sealed'], [])+i['public'])} | "
                              f"{'—' if i['retained_equals_last'] is None else 'yes' if i['retained_equals_last'] else 'no'} |")
                records.append({"model": label, "harness": harness, "run": n,
                                **{f"{folder}_passed": sum(r["passed"] for r in rows)
                                   for rows, (folder, *_) in zip(i["sealed"], splits)},
                                f"{splits[0][0]}_macro_l2": macro(i["sealed"][0]),
                                "cost_usd": i["cost"], "solve_seconds": i["solve_seconds"], "accepted": i["accepted"],
                                "submissions": i["submissions"], "invalid": i["invalid"], "stop": i["stop"]})
    cases = sum(c for *_, c in splits)+2*VALIDATION_CASES[args.suite]
    description = {"pdebench": ["Fresh sealed split: 20 cases created for this study and untouched before it; "
                                "original sealed split: 17 cases (inspected earlier)."],
                   "inventory": ["Sealed split: 12 inventory cases (one per family), never shown to any agent."]}
    lines = ["# Matched-model study: single run vs your RSI harness vs native coding agents", "",
             "Same PDE task, evaluator, starting stub, solver sandbox and hardware. The single run is one model request",
             "with the task information RSI sees in its first round; its solver is frozen as returned (no feedback,",
             "self-test, or gate). RSI and the native agent each get a 2 h budget, a $25 model-spending cap and 30 gated",
             "validation submissions. Mean ± sd over independent runs.", *description[args.suite], "",
             f"| Model | Harness | Runs | {splits[0][2]} passed | {splits[0][2]} macro L2 (lower is better) | "
             + "".join(f"{label} passed | " for _, _, label, _ in splits[1:])
             + f"Development cost | Final solver time (all {cases} cases) |",
             "| --- | --- | ---: | ---: | ---: | " + "---: | "*(len(splits)-1) + "---: | ---: |", *table, "",
             "## Per run", "",
             "| Model | Harness | Run | " + "".join(f"{label} | " for _, _, label, _ in splits)
             + f"Validation | {splits[0][2]} geo. mean L2 | Cost (USD) | Minutes | Stop | Accepted / submissions | "
             "Invalid proposals | Rejection reasons | Snapped-coefficient cases | Retained = last file |",
             "| --- | --- | ---: | " + "---: | "*len(splits)
             + "---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | :---: |", *detail]
    (output/"matched_study.md").write_text("\n".join(lines)+"\n")
    (output/"matched_study_runs.json").write_text(json.dumps(records, indent=2)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
