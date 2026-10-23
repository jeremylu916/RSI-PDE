"""Round-by-round record of what each RSI agent proposed and inherited in pdebench runs.

Usage (after sourcing scripts/env.sh):
    python scripts/export_rsi_evolution.py runs/code-agent-results-pdebench-20261017 \
        --runs runs/code-agents-pdebench-*-20261017

Writes <results>/rsi_evolution.md: for every round, the submitted candidate's hypothesis, code change summary,
proposed next improvement workflow, and memory lesson; the self-test and validation-gate outcome; and the
workflow and memory the agent carries into the next round (inherited only on acceptance).
"""

import argparse
import json
from pathlib import Path

MODELS = {"us.anthropic.claude-sonnet-4-5-20250929-v1:0": "Claude Sonnet 4.5", "openai.gpt-oss-120b-1:0": "GPT-OSS-120B",
          "us.openai.gpt-5.5": "GPT-5.5", "us.openai.gpt-6.1-sol": "GPT-6.1", "us.anthropic.claude-opus-5-5": "Claude Opus 5.5"}
ORDER = list(MODELS.values())


def quote(text):
    return "\n".join("> "+line for line in (text or "null").strip().splitlines()) or "> null"


def arm_lines(run, arm):
    d = run/arm
    events = [json.loads(line) for line in (d/"events.jsonl").open()]
    initial = json.loads((d/"initial_agent.json").read_text())["agent"]
    rounds = sorted({e["iteration"] for e in events})
    lines, workflow, memory = [], initial["improvement_workflow"], []
    lines += ["**Starting workflow** (built in):", "", quote(workflow), ""]
    for i in rounds:
        group = [e for e in events if e["iteration"] == i]
        gate = next((e for e in group if "gate" in e), None)
        attempts = []
        for e in group:
            if e.get("event") == "self_test":
                attempts.append(f"attempt {e['attempt']}: self-test {'passed' if e['passed'] else 'broken on ' + ', '.join(c.split('-')[1] for c in e['broken_cases'])}")
            elif e.get("event") in ("invalid_proposal", "model_failure"):
                attempts.append(f"attempt {e['attempt']}: {e['event'].replace('_', ' ')} ({e['message'][:90]})")
        lines += [f"#### Round {i}", "", "Attempts: " + ("; ".join(attempts) or "none") + ".", ""]
        if gate is None:
            lines += ["**No candidate reached the validation gate; the active agent is unchanged.**", ""]
            continue
        cv = gate["candidate_validation"]
        verdict = "ACCEPTED" if gate["gate"]["accepted"] else "rejected"
        reasons = "" if gate["gate"]["accepted"] else " — " + "; ".join(gate["gate"]["reasons"][:4])
        lines += [f"**Validation gate: {verdict}** — candidate macro rel. L2 {cv['macro_relative_l2']:.3g}, "
                  f"{round(cv['pass_rate']*cv['count'])}/{cv['count']} passed{reasons}", ""]
        record = json.loads((d/f"proposal-{i:03d}-{gate['attempt']}.json").read_text())["proposal"]
        update = record["program_updates"][0]
        lines += [f"- *Hypothesis:* {record['hypothesis'].strip()}",
                  f"- *Code change* ({len(update['code']):,} characters): {update['change_summary'].strip()}",
                  f"- *Memory lesson:* {record['memory_update'].strip() or '(empty)'}", "",
                  "*Proposed next workflow:*", "", quote(record["next_improvement_policy"]), ""]
        if gate["gate"]["accepted"]:
            if record["next_improvement_policy"]:
                workflow = record["next_improvement_policy"]
            memory.append(record["memory_update"])
            lines += [f"→ Inherited: workflow {'replaced' if record['next_improvement_policy'] else 'kept'}, "
                      f"memory now {len(memory)} lesson(s).", ""]
        else:
            lines += ["→ Nothing inherited (candidate rejected).", ""]
    frozen = json.loads((d/"frozen_agent.json").read_text())["agent"]
    lines += ["**Frozen agent's workflow:**", "", quote(frozen["improvement_workflow"]), "",
              f"**Frozen agent's memory** ({len(frozen['memory'])} lessons):", ""]
    lines += [f"{m['generation']}. {m['lesson'].strip()}" for m in frozen["memory"]] or ["(none)"]
    return lines + [""]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    runs = {}
    for run in args.runs:
        manifest = json.loads((run/"manifest.json").read_text())
        runs[MODELS[manifest["model_config"]["model"]]] = run
    lines = ["Round-by-round record of each RSI agent on the pde_benchmarks.tex suite. In each round the agent may",
             "propose a full solver program, a next improvement workflow, and a memory lesson; the workflow and",
             "lesson are inherited only when the candidate passes the validation gate. Rounds where every attempt",
             "failed the self-test or the static check have no gate decision.", ""]
    for model in [m for m in ORDER if m in runs]:
        lines += [f"## {model}", "", f"Run: `{runs[model]}`", ""] + arm_lines(runs[model], "rsi")
    (args.results/"rsi_evolution.md").write_text("\n".join(lines)+"\n")
    print(args.results/"rsi_evolution.md")


if __name__ == "__main__":
    main()
