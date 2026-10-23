"""API token use and cost (tokens x price) of agent runs, from the raw provider responses.

Usage (after sourcing scripts/env.sh):
    python scripts/api_cost.py --runs runs/code-agents-* runs/aborted-* --output runs/api-cost-20261017
Prices come from configs/model_prices.json (USD per 1M tokens; null = unknown, tokens still counted).
Every model-call-*/response.json under each run counts, including failed and rejected calls, single and RSI
arms, and continuation branches. Output tokens include reasoning tokens. Calls that timed out without a
response have no usage record and are listed separately.
"""

import argparse
import csv
import json
from pathlib import Path

from pde_rsi.costs import cost, usage_tokens
from pde_rsi.storage import PROJECT_ROOT, project_path

LABELS = {"us.anthropic.claude-sonnet-4-5-20250929-v1:0": "Claude Sonnet 4.5", "openai.gpt-oss-120b-1:0": "GPT-OSS-120B",
          "us.openai.gpt-5.5": "GPT-5.5", "us.openai.gpt-6.1-sol": "GPT-6.1", "us.anthropic.claude-opus-5-5": "Claude Opus 5.5"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prices", type=Path, default=PROJECT_ROOT/"configs/model_prices.json")
    args = parser.parse_args()
    prices = json.loads(args.prices.read_text())
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in map(project_path, sorted(args.runs)):
        manifest_path = run/"manifest.json"
        if not run.is_dir() or not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("kind") == "claude_code_agent":  # usage summed from the session transcripts
            usage = manifest.get("usage_total") or {}
            tokens = [usage.get(k, 0) for k in ("input", "cache_read", "cache_write", "output")]
            value = cost(tokens, prices.get(manifest["model"], {}))
            rows.append({"run": run.name, "model": LABELS.get(manifest["model"], manifest["model"]),
                         "arm": "claude_code", "calls": len(manifest.get("sessions", [])), "calls_without_response": 0,
                         "input_tokens": tokens[0], "cache_read_tokens": tokens[1], "cache_write_tokens": tokens[2],
                         "output_tokens": tokens[3], "usd": None if value is None else round(value, 4)})
            continue
        model = manifest["model_config"]["model"]
        groups = {}
        for call in run.rglob("model-call-*"):
            arm = str(call.parent.relative_to(run))
            group = groups.setdefault(arm, {"calls": 0, "no_response": 0, "tokens": [0, 0, 0, 0]})
            group["calls"] += 1
            response = call/"response.json"
            if not response.exists():
                group["no_response"] += 1
                continue
            usage = json.loads(response.read_text()).get("usage") or {}
            group["tokens"] = [a+b for a, b in zip(group["tokens"], usage_tokens(usage))]
        for arm, group in sorted(groups.items()):
            value = cost(group["tokens"], prices.get(model, {}))
            rows.append({"run": run.name, "model": LABELS.get(model, model), "arm": arm, "calls": group["calls"],
                         "calls_without_response": group["no_response"], "input_tokens": group["tokens"][0],
                         "cache_read_tokens": group["tokens"][1], "cache_write_tokens": group["tokens"][2],
                         "output_tokens": group["tokens"][3], "usd": None if value is None else round(value, 4)})
    with (output/"api_cost_by_arm.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    runs = {}
    for row in rows:
        item = runs.setdefault(row["run"], {"model": row["model"], "calls": 0, "input": 0, "output": 0, "usd": 0.0})
        item["calls"] += row["calls"]
        item["input"] += row["input_tokens"]+row["cache_read_tokens"]+row["cache_write_tokens"]
        item["output"] += row["output_tokens"]
        item["usd"] = None if item["usd"] is None or row["usd"] is None else item["usd"]+row["usd"]
    lines = ["| Run | Model | Calls | Input tokens | Output tokens | Cost (USD) |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    total, unknown = 0.0, []
    for name, item in runs.items():
        lines.append(f"| {name} | {item['model']} | {item['calls']} | {item['input']:,} | {item['output']:,} | "
                     + ("unknown price" if item["usd"] is None else f"{item['usd']:.2f}") + " |")
        if item["usd"] is None:
            unknown.append(name)
        else:
            total += item["usd"]
    lines += ["", f"Total with known prices: ${total:.2f}" + (f"; runs with unknown price: {len(unknown)}" if unknown else ""),
              "Prices: configs/model_prices.json (not all confirmed on the AWS pricing page)."]
    (output/"api_cost.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
