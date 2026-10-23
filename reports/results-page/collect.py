"""Collect every result shown on the results page into data.json.

Run from the PDE-RSI root after sourcing scripts/env.sh:
    python reports/results-page/collect.py && python reports/results-page/build.py
"""

import csv
import json
import re
from pathlib import Path

OUT = Path(__file__).parent/"data.json"


def fresh_results():
    rows = list(csv.DictReader(open("runs/v2-fresh-comparison-20261020/fresh_comparison_long.csv")))
    costs, suite = {}, None
    for line in open("runs/v2-fresh-comparison-20261020/fresh_comparison.md"):
        if line.startswith("## pdebench"):
            suite = "pdebench"
        if line.startswith("## inventory"):
            suite = "inventory"
        m = re.match(r"\| (Opus 5.5|GPT-5.5) \| ([^|]+) \| (\d) \| [^|]+ \| ([^|]+) \| \$([\d.]+) \|", line)
        if m:
            costs[(suite, m[1], m[2].strip())] = (float(m[5]), m[4].strip())
    fresh = {}
    for (suite, model, config), (cost, per_run) in costs.items():
        selected = [r for r in rows if (r["suite"], r["model"], r["configuration"]) == (suite, model, config)]
        problems = {}
        for r in selected:
            problems[r["problem"]] = problems.get(r["problem"], 0) + (r["passed"] == "True")
        fresh.setdefault(suite, []).append({"model": model, "config": config, "cost": cost, "per_run": per_run,
                                            "total": sum(r["passed"] == "True" for r in selected),
                                            "cases": len(selected), "problems": problems})
    return fresh


def earlier_results():
    earlier = {}
    for suite, path, sealed in (("pdebench", "runs/matched-study-results-20261018/per_pde_long.csv", ("test", "test2")),
                                ("inventory", "runs/matched-inv-study-results-20261020/per_family_long.csv",
                                 ("inventory_test",))):
        groups = {}
        for r in csv.DictReader(open(path)):
            if r["split"] not in sealed:
                continue
            g = groups.setdefault((r["model"], r["config"]), {"total": 0, "cases": 0, "problems": {}, "problems_n": {}})
            g["total"] += r["passed"] == "True"
            g["cases"] += 1
            g["problems"][r["problem"]] = g["problems"].get(r["problem"], 0) + (r["passed"] == "True")
            g["problems_n"][r["problem"]] = g["problems_n"].get(r["problem"], 0) + 1
        earlier[suite] = [{"model": m, "config": c, **v} for (m, c), v in groups.items()]
    for suite, path in (("pdebench", "runs/matched-study-results-20261018/matched_study.md"),
                        ("inventory", "runs/matched-inv-study-results-20261020/matched_study.md")):
        for line in open(path):
            m = re.match(r"\| (Claude Opus 5.5|GPT-5.5) \| ([^|]+) \| 3 \|.*\| \$([\d.]+) ± \$[\d.]+ \| [^|]+ \|$", line.strip())
            if not m:
                continue
            key = {"Single run (one request)": "single run", "Your RSI": "RSI"}.get(m[2].strip(), "native")
            for entry in earlier[suite]:
                native = entry["config"] in ("Claude Code", "Codex")
                if entry["model"] == m[1].replace("Claude ", "") and (entry["config"] == key or (key == "native" and native)):
                    entry["cost"] = float(m[3])
    # Single run + one rule repair: the strict single run unless its answer was rejected (only Opus on inventory).
    for suite in earlier:
        for entry in [x for x in earlier[suite] if x["config"] == "single run"]:
            repaired = dict(entry, config="single run + repair", problems=dict(entry["problems"]))
            if suite == "inventory" and entry["model"] == "Opus 5.5":
                repaired.update(total=0, problems={}, cost=None)
                for r in (1, 2, 3):
                    path = Path(f"runs/matched-inv-opus55-harness-r{r}-20261020/single_repair/sealed_test/results.json")
                    for row in json.loads(path.read_text()):
                        problem = row.get("problem") or row["task_id"].split("-")[1]
                        repaired["problems"][problem] = repaired["problems"].get(problem, 0) + row["passed"]
                        repaired["total"] += row["passed"]
            earlier[suite].append(repaired)
    return earlier


def other_results():
    other = {}
    for variant in ("pdebench", "pdebench2h", "pdebench2h-noise100", "pdebench2h-hiddentime"):
        other[variant] = json.load(open(f"runs/code-agent-results-{variant}-20261017/pdebench_overview.json"))
    other["student"] = [{k: r[k] for k in ("model", "arm", "passes", "cases", "relative_l2")}
                        for r in csv.DictReader(open("runs/code-agent-results-student-20261017/comparison.csv"))
                        if r["arm"] in ("single", "rsi")]
    other["fine"] = [{k: r[k] for k in ("families", "start", "model", "arm", "passes", "cases", "relative_l2")}
                     for r in csv.DictReader(open("runs/code-agent-results-fine-20261016/comparison.csv"))
                     if r["arm"] in ("single", "rsi")]
    return other


def diagnostic_results():
    rows = []
    for r in (1, 2, 3):
        d = json.load(open(f"runs/matched-inv-opus55-harness-r{r}-20261020/single_overlimit_v1/diagnostic_as_written/summary.json"))
        rows.append({"run": r, "characters": d["characters"], "fresh": d["inventory_test2"]["passes"],
                     "public": d["evolution"]["passes"]+d["development"]["passes"]})
    return rows


data = {"fresh": fresh_results(), "earlier": earlier_results(), "other": other_results(),
        "diagnostic": diagnostic_results()}
OUT.write_text(json.dumps(data, indent=1))
print("wrote", OUT)
