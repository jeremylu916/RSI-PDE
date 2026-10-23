"""RSI-PDEbench v1: run an entrant, score it on the hidden test, report the headline numbers.

    python scripts/bench.py run --entrant pde-rsi-v2 --model opus55 --cores 42,44,46,48,50,52 --output runs/<name>
    python scripts/bench.py run --track ext --entrant claude-code --model opus55 --cores 46,48 --output runs/<name>
    python scripts/bench.py score --runs runs/bench-v1-* --cores 42,44,46,...
    python scripts/bench.py report [--runs ...]

Entrants (src/pde_rsi/bench/spec.py has the budget; docs/rsi-pde-bench.md the rules):
    pde-rsi-v2    this project's harness v2 (modular parts, patch edits, per-part gate); its run also produces the
                  single-run entrant (one model call, frozen as returned) under its own metering lane
    claude-code   Claude Code in the native-agent sandbox (scripts/run_native_agent.py --bench)
    codex         Codex in the same sandbox
Every model call goes through the metering proxy (pde_rsi.bench.proxy), which enforces the call and token caps.
score: the retained solver after every validation submission (a round) is scored on the hidden bench-test split;
per-case hidden results stay in evaluator_private/bench/scores, the aggregate goes to <run>/bench_score.json.
Run on a GPU node (solver sandbox) after sourcing scripts/env.sh.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time

from pde_rsi.bench.proxy import Caps, Meter, MeteringProxy
from pde_rsi.bench.spec import BENCH_VERSION, BUDGET, MODELS, TRACKS
from pde_rsi.costs import cost, load_prices
from pde_rsi.modular import ROUTER, assemble
from pde_rsi.storage import PROJECT_ROOT, project_path

SECRET = PROJECT_ROOT/".runtime/secrets/bedrock_api_key"
SCORES = PROJECT_ROOT/"evaluator_private/bench/scores"
SUITE = {"tex": "pdebench", "ext": "pdebench_ext", "inventory": "inventory"}
MODEL_CONFIG = {"opus55": "claude_opus55_64k_code", "gpt55": "gpt55_code"}
NATIVE = {"claude-code": "opus55", "codex": "gpt55"}


def caps():
    return Caps(BUDGET["model_calls"], BUDGET["prompt_tokens"], BUDGET["output_tokens"])


# ---------------------------------------------------------------- run
def run_harness(args, output):
    """pde-rsi-v2: both arms (single run, RSI) concurrently, each through its own metering proxy."""
    token = SECRET.read_text().strip()
    model = MODELS[args.model]
    meter_dir = PROJECT_ROOT/".runtime/tmp"/f"bench-meter-{output.name}"
    meter_dir.mkdir(parents=True, exist_ok=False)
    meters = {arm: Meter(caps(), model, ledger=meter_dir/f"{arm}_ledger.jsonl") for arm in ("single", "rsi")}
    with MeteringProxy(meters["single"], token) as single, MeteringProxy(meters["rsi"], token) as rsi:
        command = [sys.executable, "-m", "pde_rsi", "code-agents", "--protocol", TRACKS[args.track]["protocol"], "--modular",
                   "--bench", "--start", "scratch", "--probe-rounds", "0", "--attempts-per-round", "3",
                   "--time-budget-seconds", str(BUDGET["wall_seconds"]), "--concurrent-arms", "--history-rounds", "6",
                   "--wall-seconds", str(BUDGET["wall_seconds"]+3600), "--cpu-seconds", str(BUDGET["wall_seconds"]+3600),
                   "--max-validation-submissions", str(BUDGET["validation_submissions"]),
                   "--model-config", f"configs/agent_models/{MODEL_CONFIG[args.model]}.json",
                   "--cores", args.cores, "--output", str(output),
                   "--arm-base-urls", f"single={single.url},rsi={rsi.url}"]
        status = subprocess.run(command, cwd=PROJECT_ROOT).returncode
    for arm, meter in meters.items():  # the proxies' counts are the ones that bind (the harness's own cap is higher)
        (output/arm).mkdir(parents=True, exist_ok=True)
        if meter.ledger.exists():
            shutil.copyfile(meter.ledger, output/arm/"model_ledger.jsonl")
        (output/arm/"metered_usage.json").write_text(json.dumps(meter.totals(), indent=2)+"\n")
    return status


def run_all(args, output):
    """An RSI-PDEbench run over all 42 templates: one sub-run per problem group, concurrently, two cores each."""
    cores = args.cores.split(",")
    if len(cores) < 2*len(TRACKS):
        raise SystemExit(f"--track all needs {2*len(TRACKS)} cores (two per sub-run)")
    output.mkdir(parents=True, exist_ok=False)
    procs = {part: subprocess.Popen([sys.executable, __file__, "run", "--track", part, "--entrant", args.entrant,
                                     "--model", args.model, "--cores", f"{cores[2*i]},{cores[2*i+1]}",
                                     "--output", str(output/part)], cwd=PROJECT_ROOT,
                                    stdout=open(output/f"{part}.log", "w"), stderr=subprocess.STDOUT)
             for i, part in enumerate(TRACKS)}
    status = max(proc.wait() for proc in procs.values())
    (output/"bench_entry.json").write_text(json.dumps({"bench": BENCH_VERSION, "track": "all", "parts": list(TRACKS),
                                                       "entrant": args.entrant, "model": args.model,
                                                       "exit_status": status}, indent=2)+"\n")
    return status


def run(args):
    output = project_path(args.output)
    if args.track == "all":
        return run_all(args, output)
    if args.entrant == "pde-rsi-v2":
        status = run_harness(args, output)
    else:
        if NATIVE[args.entrant] != args.model:
            raise SystemExit(f"{args.entrant} runs with {NATIVE[args.entrant]} only")
        cores = args.cores.split(",")
        status = subprocess.run([sys.executable, "scripts/run_native_agent.py", "--agent", args.entrant,
                                 "--suite", SUITE[args.track], "--model", MODELS[args.model], "--bench",
                                 "--output", str(output), "--eval-core", cores[0], "--agent-core", cores[1]],
                                cwd=PROJECT_ROOT).returncode
    (output/"bench_entry.json").write_text(json.dumps({"bench": BENCH_VERSION, "track": args.track,
                                                       "entrant": args.entrant, "model": args.model,
                                                       "exit_status": status}, indent=2)+"\n")
    return status


# ---------------------------------------------------------------- rounds of a run
def program_code(program):
    """Executable code of a harness program dict (modular parts or a single module)."""
    if set(program) == {ROUTER} and "SOLVERS" not in program[ROUTER]:
        return program[ROUTER]
    return assemble(program)


def rounds(run):
    """{arm: [(submissions used, retained code)]}, starting from the stub at 0 submissions."""
    entry = json.loads((run/"bench_entry.json").read_text())
    if entry["entrant"] != "pde-rsi-v2":
        from pde_rsi.codegen import STUB_PROGRAM
        curve, retained = [(0, STUB_PROGRAM)], STUB_PROGRAM
        path = run/"submissions.jsonl"
        for line in (path.read_text().splitlines() if path.exists() else []):
            item = json.loads(line)
            if item["accepted"]:
                retained = (run/"retained"/f"submission-{item['submission']:03d}.py").read_text()
            curve.append((item["submission"], retained))
        return {entry["entrant"]: curve}
    arms = {}
    for arm in ("rsi", "single"):
        directory = run/arm
        initial = program_code(json.loads((directory/"initial_agent.json").read_text())["agent"]["program"])
        if arm == "single":  # one model call, frozen as returned
            arms["single"] = [(0, program_code(json.loads((directory/"frozen_agent.json").read_text())["agent"]["program"]))]
            continue
        curve = [(0, initial)]
        validated = sorted(directory.glob("candidate-validation-*.json"))
        for count, path in enumerate(validated, 1):
            number = path.stem.rsplit("-", 1)[1]
            checkpoint = json.loads((directory/f"checkpoint-{number}.json").read_text())
            curve.append((count, program_code(checkpoint["agent"]["program"])))
        frozen = program_code(json.loads((directory/"frozen_agent.json").read_text())["agent"]["program"])
        if curve[-1][1] != frozen:
            raise RuntimeError(f"{run.name}: the last checkpoint differs from the frozen program")
        arms["rsi"] = curve
    return arms


# ---------------------------------------------------------------- hidden-test scoring
def hidden_scores(track, codes, cores):
    """Per-case hidden-test rows of each distinct program, cached by code hash (pde_rsi.bench.evaluate)."""
    from pde_rsi.bench.evaluate import score_programs
    results, todo = {}, []
    for code in codes:
        sha = hashlib.sha256(code.encode()).hexdigest()[:16]
        path = SCORES/f"{track}-{sha}.json"
        if path.exists():
            results[sha] = json.loads(path.read_text())
        elif code not in todo:
            todo.append(code)
    if todo:
        print(f"{track}: hidden test for {len(todo)} programs", flush=True)
        SCORES.mkdir(parents=True, exist_ok=True)
        for code, rows in score_programs(track, "bench-test", todo, cores).items():
            sha = hashlib.sha256(code.encode()).hexdigest()[:16]
            results[sha] = rows
            (SCORES/f"{track}-{sha}.json").write_text(json.dumps(rows, indent=1, default=float)+"\n")
    return results


def pass_rate(rows):
    return sum(bool(r["passed"]) for r in rows)/len(rows)


def validation_pass_rate(run, arm):
    if arm in ("claude-code", "codex"):
        rows = json.loads((run/"final/frozen_results.json").read_text())
    else:
        rows = json.loads((run/arm/"frozen_results.json").read_text())
    val = [r for r in rows if r.get("split") == "development" or str(r.get("case_id", "")).startswith("case-bv")]
    return pass_rate(val) if val else None


def usage(run, arm):
    if arm in ("claude-code", "codex"):
        manifest = json.loads((run/"manifest.json").read_text())
        return manifest.get("metered_usage"), manifest.get("improvement_wall_seconds")
    path = run/arm/"metered_usage.json"
    report = json.loads((run/arm/"report.json").read_text())
    return (json.loads(path.read_text()) if path.exists() else None), report["budget"].get("elapsed_wall_seconds")


def score(args):
    cores = [int(c) for c in args.cores.split(",")]
    prices = load_prices()
    runs = []
    for run in map(project_path, args.runs):  # a run over all templates is scored part by part
        entry = json.loads((run/"bench_entry.json").read_text())
        runs += [run/part for part in entry["parts"]] if entry.get("parts") else [run]
    for run in runs:
        entry = json.loads((run/"bench_entry.json").read_text())
        track, model = entry["track"], MODELS[entry["model"]]
        out = {"bench": BENCH_VERSION, "track": track, "entrant": entry["entrant"], "model": entry["model"], "arms": {}}
        for arm, curve in rounds(run).items():
            from pde_rsi.codegen import STUB_PROGRAM
            codes = [code for _, code in curve if code != STUB_PROGRAM]
            scores = hidden_scores(track, codes, cores)
            points = []
            for submissions, code in curve:
                sha = hashlib.sha256(code.encode()).hexdigest()[:16]
                rate = 0.0 if code == STUB_PROGRAM else pass_rate(scores[sha])
                points.append({"submissions": submissions, "program": sha, "hidden_pass_rate": rate})
            metered, wall = usage(run, arm)
            price = prices.get(model) or {}
            spent = (cost((metered["input"], metered["cache_read"], metered["cache_write"], metered["output"]), price)
                     if metered else None)
            final, val = points[-1]["hidden_pass_rate"], validation_pass_rate(run, arm)
            out["arms"][arm] = {"curve": points, "hidden_pass_rate": final, "validation_pass_rate": val,
                                "dev_test_gap": None if val is None else val-final, "metered_usage": metered,
                                "wall_seconds": wall, "cost_usd": spent, "rounds": points[-1]["submissions"]}
        (run/"bench_score.json").write_text(json.dumps(out, indent=2)+"\n")
        print(run.name, {arm: (a["hidden_pass_rate"], a["rounds"]) for arm, a in out["arms"].items()}, flush=True)


# ---------------------------------------------------------------- report
def report(args):
    runs = [project_path(r) for r in args.runs] if args.runs else sorted(PROJECT_ROOT.glob("runs/bench-v1-*"))
    rows = []
    for run in runs:
        for path in [run/"bench_score.json", *sorted(run.glob("*/bench_score.json"))]:
            if path.exists():
                rows += report_rows(path)
    single = {(r["track"], r["model"]): r["hidden_pass_rate"] for r in rows if r["entrant"] == "single run"}

    for r in rows:
        base = single.get((r["track"], r["model"]))
        r["gain_over_single"] = None if base is None or r["entrant"] == "single run" else r["hidden_pass_rate"]-base
    out = PROJECT_ROOT/"reports/bench-v1"
    out.mkdir(parents=True, exist_ok=True)
    (out/"results.json").write_text(json.dumps(rows, indent=2)+"\n")
    fmt = lambda v, d=2: "—" if v is None else f"{v:+.{d}f}" if isinstance(v, float) and v < 0 else f"{v:.{d}f}"  # noqa: E731
    lines = ["| Track | Model | Entrant | Held-out pass rate | Gain over single run | Validation pass rate | Dev–test gap "
             "| Rounds | Model calls | Cost (USD) |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    order = {"tex": 0, "ext": 1, "inventory": 2}
    for r in sorted(rows, key=lambda r: (order[r["track"]], r["model"], r["entrant"] != "single run", r["entrant"])):
        calls = (r["metered_usage"] or {}).get("calls")
        lines.append(f"| {r['track']} | {r['model']} | {r['entrant']} | {fmt(r['hidden_pass_rate'])} | "
                     f"{fmt(r['gain_over_single'])} | {fmt(r['validation_pass_rate'])} | {fmt(r['dev_test_gap'])} | "
                     f"{r['rounds']} | {calls if calls is not None else '—'} | {fmt(r['cost_usd'], 1)} |")
    (out/"results.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


def report_rows(path):
    item = json.loads(path.read_text())
    return [{"track": item["track"], "model": item["model"], "entrant": {"rsi": "pde-rsi-v2", "single": "single run"}.get(arm, arm),
             "run": str(path.parent.relative_to(PROJECT_ROOT)), **a} for arm, a in item["arms"].items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--track", choices=("all", *TRACKS), default="all",
                   help="all (default): every template, as three concurrent sub-runs; or one problem group")
    r.add_argument("--entrant", choices=("pde-rsi-v2", "claude-code", "codex"), required=True)
    r.add_argument("--model", choices=tuple(MODELS), required=True)
    r.add_argument("--cores", required=True, help="two OS cores per sub-run: the two arms (harness) or evaluation "
                                                   "and agent; six for --track all")
    r.add_argument("--output", required=True)
    s = sub.add_parser("score")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--cores", required=True)
    p = sub.add_parser("report")
    p.add_argument("--runs", nargs="*")
    args = parser.parse_args()
    started = time.time()
    status = {"run": run, "score": score, "report": report}[args.command](args)
    if args.command == "run":
        print(f"done in {time.time()-started:.0f} s, exit {status}", flush=True)
        sys.exit(status or 0)


if __name__ == "__main__":
    main()
