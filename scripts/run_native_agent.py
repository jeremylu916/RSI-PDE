"""Native coding-agent baseline (Claude Code or Codex) on the pdebench task, matched to the PDE-RSI harness.

The benchmark, evaluator, and sealed tests are those of the harness; only the system that develops the solver
changes. Matching rules (docs/harness-design-comparison.md, matched-model study of 2026-10-18):
- the same task text, equation library, 32 public case views, noisy data, and starting stub as the harness;
- training feedback: unlimited requests, the same diagnostics as the harness (labels on training, no program prints);
- validation: at most --max-validation-submissions scored submissions of the full 16 cases; each goes through the
  harness's acceptance gate against the currently retained solver, and the RETAINED solver (last accepted
  submission) is the result, as in the harness;
- 2 h wall-clock improvement budget and a --spend-cap-usd model-spending cap priced with configs/model_prices.json
  (the same rule as the harness arms), enforced live from the agent's own usage logs;
- the agent runs in bubblewrap with network for Bedrock only, no web tools, a fresh HOME, and only /work; solver
  evaluations run in the harness's GPU solver sandbox on a dedicated core.
The agent chooses its own sequence of edits and experiments. It is re-prompted (--continue / exec resume) until the
deadline, the spending cap, the validation budget, or two consecutive idle sessions.

Usage on a GPU node (after sourcing scripts/env.sh):
    python scripts/run_native_agent.py --agent codex --model us.openai.gpt-5.5 \
        --output runs/native-codex-gpt55-r1-20261018 --eval-core 29 --agent-core 31
"""

import argparse
from datetime import datetime, timezone
import glob
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import threading
import time
import types

import numpy as np

from pde_rsi.codegen import STUB_PROGRAM, ProgramRejected, check_program, gpu_env_prefix
from pde_rsi.costs import cost, load_prices
from pde_rsi.evaluator import acceptance_gate
from pde_rsi import inventory, pdebench, pdebench_ext
from pde_rsi.pdebench import PROGRAM_KEY

SUITES = {
    "pdebench": types.SimpleNamespace(contract=pdebench.CONTRACT, library=pdebench.LIBRARY, protocol=pdebench.PDEBenchProtocol,
                                      make_cases=pdebench.make_cases, run_case=pdebench.run_case, sealed=("test", "test2")),
    "inventory": types.SimpleNamespace(contract=inventory.CONTRACT, library=inventory.library(),
                                       protocol=inventory.InventoryProtocol, make_cases=inventory.make_cases,
                                       run_case=inventory.run_case, sealed=("inventory_test",)),
    "pdebench_ext": types.SimpleNamespace(contract=pdebench_ext.CONTRACT, library=pdebench_ext.LIBRARY,
                                          protocol=pdebench_ext.PDEBenchExtProtocol, make_cases=pdebench_ext.make_cases,
                                          run_case=pdebench_ext.run_case, sealed=()),
}
SUITE = SUITES["pdebench"]  # replaced from --suite in main()
from pde_rsi.runner import Budget
from pde_rsi.storage import PROJECT_ROOT, project_path

CLAUDE_BINARY = Path.home()/".local/bin/claude"
CODEX_BINARY = PROJECT_ROOT/".runtime/tools/codex/rust-v0.160.0/codex-x86_64-unknown-linux-musl"
SECRET = PROJECT_ROOT/".runtime/secrets/bedrock_api_key"
CODEX_CONFIG = """model = "{model}"
model_provider = "bedrock"
model_reasoning_effort = "medium"
web_search = "disabled"

[model_providers.bedrock]
name = "Amazon Bedrock (OpenAI-compatible Responses API)"
base_url = "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"
env_key = "BEDROCK_OPENAI_API_KEY"
wire_api = "responses"
"""
CLIENT = r'''"""Official evaluation of /work/solver.py through the host evaluation service.

    python tools/evaluate.py train                        # all training cases (labels and errors); unlimited
    python tools/evaluate.py train --cases case-t00,case-t05
    python tools/evaluate.py submit                       # one scored validation submission (all validation cases)
    python tools/evaluate.py --wait <request_id>          # collect a request that was still running
A full split can take several minutes: run this with a Bash timeout of 600000 ms.
"""
import json, os, sys, time, uuid

def wait(request_id, limit=560):
    path = f"/work/responses/{request_id}.json"
    start = time.time()
    while time.time()-start < limit:
        if os.path.exists(path):
            print(open(path).read())
            return
        time.sleep(2)
    print(json.dumps({"request_id": request_id, "status": "still running",
                      "hint": f"run: python tools/evaluate.py --wait {request_id}"}))

args = sys.argv[1:]
if args[:1] == ["--wait"]:
    wait(args[1])
    sys.exit()
kind = args[0] if args else "train"
if kind not in ("train", "submit"):
    sys.exit("usage: evaluate.py train [--cases a,b] | submit | --wait <id>")
cases = args[args.index("--cases")+1].split(",") if "--cases" in args and kind == "train" else None
request_id = uuid.uuid4().hex[:12]
request = {"request_id": request_id, "kind": kind, "cases": cases, "code": open("/work/solver.py").read()}
tmp = f"/work/requests/.{request_id}.tmp"
json.dump(request, open(tmp, "w"))
os.rename(tmp, f"/work/requests/{request_id}.json")
print(f"submitted {request_id}", flush=True)
wait(request_id)
'''


def readme(cases, deadline_text, minutes, submissions, cap, caps=None):
    views = "\n".join(f"- `cases/{c.case_id}/` ({'training' if c.split == 'evolution' else 'validation'})" for c in cases)
    n_val = sum(c.split == "development" for c in cases)
    budget = (f"a model budget of **{caps.calls} model calls, {caps.prompt_tokens:,} prompt tokens and "
              f"{caps.output_tokens:,} output tokens** (further calls are refused)" if caps is not None else
              f"a model-spending budget of **${cap:.0f}**")
    return f"""# Task: identify and solve PDE/FPDE problems from data

You have **{minutes} minutes** of wall-clock time (deadline: {deadline_text}) and {budget}.
Develop one program, `/work/solver.py`, that solves every case.

**What counts.** Your result is the *retained* solver: the last validation submission that the acceptance gate
accepted. It is scored on these public cases and on held-out test cases of the same kinds that you never see.
You have at most **{submissions} validation submissions** (`python tools/evaluate.py submit`). Each is scored on all
{n_val} validation cases and compared with the currently retained solver. It is accepted only if the macro relative L2
(mean of class means) drops by more than 0.5 %, no class mean rises more than 5 % or loses passes, no case's
combined or final-time relative L2 rises more than 25 %, and every case executes with exact IC/BC within limits.
The retained solver starts as the stub in `solver.py`. When the run ends (deadline, model budget, or you stop),
the current `solver.py` is submitted once more automatically if it differs from the retained solver and
submissions remain; that submission goes through the same gate.

## What to write

{SUITE.contract}

library = {json.dumps(SUITE.library)}

## Files and tools

- `cases/<case_id>/view.json`: the public attributes of a case (what `problem` contains besides the callables
  `forcing`, `initial`, `initial_velocity`, `boundary`, and `device`).
- `cases/<case_id>/data.npz`: `data_x`, `data_y`, `data_t`, `data_points` (only those the case has), `data_u`.
- `cases/<case_id>/inputs.npz` (when present): the case's gridded input arrays and, for 4-D cases, `points`.
- `solver.py`: your program (currently the stub).
- `tools/evaluate.py train [--cases ...]`: runs `solver.py` in the official sandbox on training cases (any subset,
  unlimited) and returns errors, pass/fail, and the true class and parameters. `tools/evaluate.py submit`: one
  scored validation submission (errors only, no labels) and the gate decision. The forcing, initial, and boundary
  data exist only inside these evaluations. Program prints are not returned.
  A full split can take several minutes: call it with a Bash timeout of 600000 ms.

Python with numpy, scipy, and torch (CUDA) is available as `python`. No internet access is available or needed.

Cases:
{views}
"""


class EvaluationService(threading.Thread):
    """Training diagnostics on demand; validation submissions through the harness gate, capped."""

    def __init__(self, run, cases, max_submissions):
        super().__init__(daemon=True)
        self.directory, self.cases, self.max_submissions = run, cases, max_submissions
        self.protocol = SUITE.protocol()
        self.protocol.cases = {c.task_id: c for c in cases}
        self.validation = [c for c in cases if c.split == "development"]
        self.stop = threading.Event()
        self.counts = {"train": 0, "submissions": 0, "accepted": 0, "rejected_by_policy": 0, "solves": 0}
        self.retained_code = STUB_PROGRAM
        self.retained_feedback = self.evaluate(self.validation, STUB_PROGRAM)
        self.last_submission = time.time()
        (run/"retained").mkdir(exist_ok=True)
        (run/"retained/solver.py").write_text(STUB_PROGRAM)

    def evaluate(self, tasks, code):
        budget = Budget(max(1, len(tasks)), 7200, max_cpu_seconds=7200)
        self.counts["solves"] = self.counts.get("solves", 0)+len(tasks)
        return self.protocol.evaluate(tasks, types.SimpleNamespace(program={PROGRAM_KEY: code}), budget)

    def answer(self, request):
        kind = request.get("kind")
        if kind == "submit" and self.counts["submissions"] >= self.max_submissions:
            return {"request_id": request["request_id"], "status": "validation_budget_exhausted",
                    "message": f"All {self.max_submissions} validation submissions are used; the retained solver is final."}
        try:
            check_program(request["code"], PROGRAM_KEY)
        except ProgramRejected as error:
            self.counts["rejected_by_policy"] += 1
            return {"request_id": request["request_id"], "status": "rejected_by_static_policy", "message": str(error)}
        if kind == "train":
            tasks = [c for c in self.cases if c.split == "evolution" and (not request.get("cases") or c.case_id in request["cases"])]
            results = self.evaluate(tasks, request["code"])
            self.counts["train"] += 1
            return {"request_id": request["request_id"], "kind": "train", "cases": len(tasks),
                    "passed": sum(r["passed"] for r in results), "diagnostics": self.protocol.diagnostics(results)}
        results = self.evaluate(self.validation, request["code"])
        self.counts["submissions"] += 1
        self.last_submission = time.time()
        gate = acceptance_gate(self.retained_feedback, results)
        number = self.counts["submissions"]
        (self.directory/"retained"/f"submission-{number:03d}.py").write_text(request["code"])
        if gate["accepted"]:
            self.counts["accepted"] += 1
            self.retained_code, self.retained_feedback = request["code"], results
            (self.directory/"retained/solver.py").write_text(request["code"])
        with (self.directory/"submissions.jsonl").open("a") as stream:
            stream.write(json.dumps({"submission": number, "accepted": gate["accepted"], "reasons": gate["reasons"],
                                     "passed": sum(r["passed"] for r in results),
                                     "macro_relative_l2": float(np.mean([np.mean([r["relative_l2"] for r in results
                                                                                  if r["family"] == f])
                                                                         for f in {r["family"] for r in results}])),
                                     "utc": datetime.now(timezone.utc).isoformat()})+"\n")
        return {"request_id": request["request_id"], "kind": "submit", "submission": number,
                "submissions_remaining": self.max_submissions-number, "accepted": gate["accepted"],
                "reasons": self.protocol.reasons(gate["reasons"], self.validation),
                "relative_gain": gate["relative_gain"], "passed": sum(r["passed"] for r in results),
                "diagnostics": self.protocol.diagnostics(results)}

    def run(self):
        requests, responses = self.directory/"workspace/requests", self.directory/"workspace/responses"
        while not self.stop.is_set():
            for path in sorted(requests.glob("*.json"), key=lambda p: p.stat().st_mtime):
                started = time.time()
                try:
                    request = json.loads(path.read_text())
                    response = self.answer(request)
                except Exception as error:  # malformed requests are answered, never crash the service
                    request, response = {"request_id": path.stem}, {"request_id": path.stem, "status": "error",
                                                                     "message": str(error)[:500]}
                tmp = responses/f".{path.stem}.tmp"
                tmp.write_text(json.dumps(response, indent=1, default=str))
                tmp.rename(responses/f"{path.stem}.json")
                shutil.move(str(path), self.directory/"evaluation_requests"/path.name)
                with (self.directory/"evaluation_log.jsonl").open("a") as stream:
                    stream.write(json.dumps({"request_id": path.stem, "kind": request.get("kind"),
                                             "cases": request.get("cases"), "status": response.get("status", "ok"),
                                             "passed": response.get("passed"), "accepted": response.get("accepted"),
                                             "seconds": time.time()-started,
                                             "utc": datetime.now(timezone.utc).isoformat()})+"\n")
            time.sleep(1)


class UsageTracker:
    """Live token totals from the agent's own logs, read incrementally (new lines only).

    Claude Code: stream-json transcripts with partial messages (message_start carries input/cache tokens, the
    last message_delta the final output tokens). Codex: rollout logs, whose token_count events carry the
    session's cumulative usage after every model call."""

    def __init__(self, agent, home):
        self.agent, self.home = agent, home
        self.offsets, self.messages, self.current, self.sessions = {}, {}, None, {}

    def files(self, transcripts):
        if self.agent == "claude-code":
            return [str(t) for t in transcripts]
        return glob.glob(str(self.home/".codex/sessions/**/rollout-*.jsonl"), recursive=True)

    def update(self, transcripts):
        for path in self.files(transcripts):
            if not os.path.exists(path):
                continue
            with open(path) as stream:
                stream.seek(self.offsets.get(path, 0))
                while True:
                    line = stream.readline()
                    if not line or not line.endswith("\n"):
                        break  # incomplete last line: read it next time
                    self.offsets[path] = stream.tell()
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    self.consume(path, event)
        return self.totals()

    def consume(self, path, event):
        if self.agent == "claude-code":
            if event.get("type") == "stream_event":
                inner = event.get("event") or {}
                if inner.get("type") == "message_start":
                    self.current = inner["message"]["id"]
                    self.messages[self.current] = dict(inner["message"].get("usage") or {})
                elif inner.get("type") == "message_delta" and self.current in self.messages:
                    self.messages[self.current].update({k: v for k, v in (inner.get("usage") or {}).items()
                                                        if v is not None})
            elif event.get("type") == "assistant":
                message = event.get("message") or {}
                self.messages.setdefault(message.get("id"), dict(message.get("usage") or {}))
            return
        payload = event.get("payload") or {}
        if payload.get("type") == "token_count" and payload.get("info"):
            item = self.sessions.setdefault(path, {"usage": {}, "calls": 0})
            item["usage"], item["calls"] = payload["info"]["total_token_usage"], item["calls"]+1

    def totals(self):
        if self.agent == "claude-code":
            pick = lambda key: sum(m.get(key) or 0 for m in self.messages.values())  # noqa: E731
            return {"input": pick("input_tokens"), "cache_read": pick("cache_read_input_tokens"),
                    "cache_write": pick("cache_creation_input_tokens"), "output": pick("output_tokens"),
                    "calls": len(self.messages)}
        totals = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0, "calls": 0}
        for item in self.sessions.values():
            usage, cached = item["usage"], item["usage"].get("cached_input_tokens") or 0
            totals["input"] += (usage.get("input_tokens") or 0)-cached
            totals["cache_read"] += cached
            totals["cache_write"] += usage.get("cache_write_input_tokens") or 0
            totals["output"] += usage.get("output_tokens") or 0
            totals["calls"] += item["calls"]
        return totals


def sandbox(run, agent, model, agent_core, token, proxy_url=None):
    gpu = [arg for device in sorted(Path("/dev").glob("nvidia*")) if device.is_char_device()
           for arg in ("--dev-bind", str(device), str(device))]
    binary = CLAUDE_BINARY.resolve() if agent == "claude-code" else CODEX_BINARY
    command = ["taskset", "-c", str(agent_core), "bwrap", "--unshare-all", "--share-net", "--die-with-parent",
               "--new-session", "--cap-drop", "ALL", "--clearenv", "--ro-bind", "/usr", "/usr",
               "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/lib", "/lib", "--symlink", "usr/bin", "/bin",
               "--ro-bind", "/etc", "/etc", "--ro-bind", str(binary), "/opt/agent/agent",
               "--ro-bind", str(gpu_env_prefix().resolve()), "/opt/gpu-env", "--ro-bind", "/sys", "/sys",
               "--bind", str(run/"workspace"), "/work", "--bind", str(run/"agent_home"), "/home/agent",
               "--proc", "/proc", "--dev", "/dev", *gpu, "--tmpfs", "/tmp", "--chdir", "/work"]
    env = {"HOME": "/home/agent", "PATH": "/opt/gpu-env/bin:/usr/bin:/bin", "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    if agent == "claude-code" and proxy_url:  # RSI-PDE Bench: the metering proxy holds the credential
        env.update({"ANTHROPIC_BEDROCK_BASE_URL": proxy_url, "CLAUDE_CODE_SKIP_BEDROCK_AUTH": "1"})
        token = "metered-by-proxy"
    if agent == "claude-code":
        env.update({"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": "us-east-1", "AWS_BEARER_TOKEN_BEDROCK": token,
                    "ANTHROPIC_MODEL": model, "ANTHROPIC_SMALL_FAST_MODEL": model, "DISABLE_AUTOUPDATER": "1",
                    "DISABLE_TELEMETRY": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000"})  # same per-call output cap as the harness's Opus arms
    else:
        env.update({"CODEX_HOME": "/home/agent/.codex",
                    "BEDROCK_OPENAI_API_KEY": "metered-by-proxy" if proxy_url else token})
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    for key, value in env.items():
        command += ["--setenv", key, value]
    return command+["/opt/agent/agent"]


def agent_command(agent, base, session, prompt):
    if agent == "claude-code":
        return base+["-p", prompt, "--dangerously-skip-permissions", "--disallowedTools", "WebSearch", "WebFetch",
                     "--output-format", "stream-json", "--verbose", "--include-partial-messages"] + \
            (["--continue"] if session > 1 else [])
    flags = ["--json", "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox"]
    return base+(["exec"]+flags+[prompt] if session == 1 else ["exec", "resume", "--last"]+flags+[prompt])


def score(code, split, target):
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in SUITE.make_cases(split):
        try:
            check_program(code, PROGRAM_KEY)
            rows.append(SUITE.run_case(case, code, Budget(1, 3600, max_cpu_seconds=3600), target/f"{case.case_id}.npz",
                                       gpu=SUITE.protocol.gpu))
        except ProgramRejected as error:
            rows.append({"case_id": case.case_id, "task_id": case.task_id, "split": case.split, "problem": case.truth.problem,
                         "status": "rejected", "passed": False, "relative_l2": 1e6, "message": str(error),
                         "family_correct": False})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("claude-code", "codex"), required=True)
    parser.add_argument("--suite", choices=tuple(SUITES), default="pdebench")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=7200)
    parser.add_argument("--spend-cap-usd", type=float, default=25.0)
    parser.add_argument("--max-validation-submissions", type=int, default=30)
    parser.add_argument("--eval-core", type=int, required=True)
    parser.add_argument("--agent-core", type=int, required=True)
    parser.add_argument("--cpu-only", action="store_true", help="testing only: solver sandbox without the GPU")
    parser.add_argument("--bench", action="store_true",
                        help="RSI-PDE Bench: bench-train/bench-val cases, the bench budget (src/pde_rsi/bench/spec.py) "
                             "metered by the proxy instead of a spending cap, and no sealed-split scoring")
    args = parser.parse_args()
    prices = load_prices()
    price = prices.get(args.model) or {}
    if price.get("output") is None:
        raise SystemExit(f"No price for {args.model}; the spending cap cannot be enforced")
    run = project_path(args.output)
    run.mkdir(parents=True, exist_ok=False)
    meter = proxy = None
    if args.bench:
        from pde_rsi.bench.proxy import Caps, Meter, MeteringProxy
        from pde_rsi.bench.spec import BUDGET
        args.budget_seconds, args.max_validation_submissions = BUDGET["wall_seconds"], BUDGET["validation_submissions"]
        meter = Meter(Caps(BUDGET["model_calls"], BUDGET["prompt_tokens"], BUDGET["output_tokens"]), args.model,
                      ledger=run/"model_ledger.jsonl")
        proxy = MeteringProxy(meter, SECRET.read_text().strip()).__enter__()
    os.sched_setaffinity(0, {args.eval_core})  # the evaluation service and final scoring run on this core
    for sub in ("workspace/cases", "workspace/tools", "workspace/requests", "workspace/responses", "agent_home",
                "evaluation_requests", "transcripts", "final"):
        (run/sub).mkdir(parents=True)
    if args.agent == "codex":
        (run/"agent_home/.codex").mkdir()
        config = CODEX_CONFIG.format(model=args.model)
        if proxy is not None:
            config = config.replace("https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1", f"{proxy.url}/openai/v1")
        (run/"agent_home/.codex/config.toml").write_text(config)
    global SUITE
    SUITE = SUITES[args.suite]
    names = ("bench-train", "bench-val") if args.bench else ("evolution", "development")
    cases = SUITE.make_cases(names[0])+SUITE.make_cases(names[1])
    SUITE.protocol.split_names = names
    for case in cases:
        folder = run/"workspace/cases"/case.case_id
        folder.mkdir()
        (folder/"view.json").write_text(json.dumps({**case.public_view(), "library": SUITE.library}, indent=1))
        data = case.data()
        np.savez(folder/"data.npz", data_u=data["u"],
                 **{f"data_{k}": data[k] for k in ("x", "y", "t", "points") if data.get(k) is not None})
        spec = case.truth.spec
        gridded = {name: np.asarray(v) for name, v in (spec.get("fields") or {}).items()} if args.suite == "inventory" else {}
        if args.suite == "inventory" and case.truth.layout == "4d_points":
            gridded["points"] = inventory.reference(case.truth)["points"]
        if gridded:
            np.savez(folder/"inputs.npz", **gridded)
    (run/"workspace/solver.py").write_text(STUB_PROGRAM)
    (run/"workspace/tools/evaluate.py").write_text(CLIENT)
    SUITE.protocol.gpu = not args.cpu_only
    service = EvaluationService(run, cases, args.max_validation_submissions)  # scores the stub before the clock
    start = time.time()
    deadline = start+args.budget_seconds
    deadline_text = datetime.fromtimestamp(deadline).astimezone().strftime("%H:%M:%S %Z")
    (run/"workspace/README.md").write_text(readme(cases, deadline_text, round(args.budget_seconds/60),
                                                  args.max_validation_submissions, args.spend_cap_usd,
                                                  meter.caps if meter is not None else None))
    version = subprocess.run([str(CLAUDE_BINARY if args.agent == "claude-code" else CODEX_BINARY), "--version"],
                             capture_output=True, text=True).stdout.strip()
    manifest = {"kind": "native_coding_agent", "suite": args.suite, "agent": args.agent, "agent_version": version,
                "bench": ({"version": __import__("pde_rsi.bench.spec", fromlist=["x"]).BENCH_VERSION,
                           "budget": __import__("pde_rsi.bench.spec", fromlist=["x"]).BUDGET, "splits": list(names),
                           "metering": "pde_rsi.bench.proxy"} if args.bench else None),
                "model": args.model,
                "budget_seconds": args.budget_seconds, "spend_cap_usd": args.spend_cap_usd,
                "max_validation_submissions": args.max_validation_submissions, "price": price,
                "start_utc": datetime.now(timezone.utc).isoformat(),
                "hardware": {"node": platform.node(), "eval_core": args.eval_core, "agent_core": args.agent_core},
                "settings": ({"web_tools": "WebSearch/WebFetch disallowed", "max_output_tokens": 64000}
                             if args.agent == "claude-code" else
                             {"web_search": "disabled", "reasoning_effort": "medium", "wire_api": "responses"}),
                "feedback": "training: unlimited, labels, no program prints; validation: gated submissions",
                "sessions": []}
    (run/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    service.start()
    token = SECRET.read_text().strip()
    base = sandbox(run, args.agent, args.model, args.agent_core, token, proxy.url if proxy is not None else None)
    transcripts, tracker = [], UsageTracker(args.agent, run/"agent_home")

    def usage():
        return tracker.update(transcripts)

    def spent():
        totals = meter.totals() if meter is not None else usage()
        return cost((totals["input"], totals["cache_read"], totals["cache_write"], totals["output"]), price) or 0.0

    def over_budget():  # bench: the proxy's caps; otherwise the spending cap
        return bool(meter.exhausted()) if meter is not None else spent() >= args.spend_cap_usd

    session, idle, stop_reason = 0, 0, None
    while True:
        remaining = deadline-time.time()
        if remaining <= 300:
            stop_reason = "deadline"
            break
        if over_budget():
            stop_reason = "model_budget" if meter is not None else "spending_cap"
            break
        if service.counts["submissions"] >= args.max_validation_submissions:
            stop_reason = "validation_budget"
            break
        if idle >= 2:
            stop_reason = "idle"
            break
        session += 1
        before = ((run/"workspace/solver.py").read_text(), service.counts["submissions"])
        prompt = ("Read /work/README.md and complete the task. Work autonomously; nobody will answer questions."
                  if session == 1 else
                  f"You have {remaining/60:.0f} minutes and {args.max_validation_submissions-service.counts['submissions']} "
                  "validation submissions left. Keep improving /work/solver.py if you can; submit only versions you "
                  "expect the gate to accept, and stop when you are confident it cannot be improved.")
        transcript = run/"transcripts"/f"session-{session:02d}.jsonl"
        transcripts.append(transcript)
        began, status = time.time(), None
        with transcript.open("w") as stream:
            proc = subprocess.Popen(agent_command(args.agent, base, session, prompt), stdout=stream,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            while proc.poll() is None:  # live limits: deadline and spending cap
                time.sleep(5)
                if time.time() >= deadline:
                    status = "killed at deadline"
                elif over_budget():
                    status = "killed at model budget" if meter is not None else "killed at spending cap"
                elif service.counts["submissions"] >= args.max_validation_submissions and \
                        time.time()-service.last_submission > 60:
                    status = "stopped: validation budget used"
                if status:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                    break
            status = status or f"exit {proc.returncode}"
        after = ((run/"workspace/solver.py").read_text(), service.counts["submissions"])
        idle = 0 if after != before or session == 1 else idle+1
        manifest["sessions"].append({"session": session, "status": status, "seconds": time.time()-began,
                                     "worked": after != before, "spent_usd_after": spent()})
        (run/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        if status.startswith("killed"):
            stop_reason = status.replace("killed at ", "")
            break
    service.stop.set()
    service.join(timeout=3600)
    # Final automatic submission of the last file through the same gate (counts against the budget), so that a
    # run stopped by the clock or the cap is judged like a harness round's last candidate.
    last = (run/"workspace/solver.py").read_text()
    manifest["final_auto_submission"] = None
    if last != service.retained_code and service.counts["submissions"] < args.max_validation_submissions:
        response = service.answer({"request_id": "final-auto", "kind": "submit", "code": last})
        manifest["final_auto_submission"] = {k: response.get(k) for k in ("status", "accepted", "reasons", "passed")}
    totals = usage()
    if meter is not None:
        manifest["metered_usage"] = meter.totals()  # the proxy's count is the one that binds
        proxy.__exit__(None, None, None)
    manifest.update({"stop_reason": stop_reason, "improvement_wall_seconds": time.time()-start,
                     "evaluation_counts": service.counts, "usage_total": totals, "model_cost_usd": spent()})
    retained = service.retained_code
    last = (run/"workspace/solver.py").read_text()
    (run/"final/retained_solver.py").write_text(retained)
    (run/"final/last_file_solver.py").write_text(last)
    manifest["retained_equals_last_file"] = retained == last
    scored = names if args.bench else ("evolution", "development", *SUITE.sealed)  # bench: hidden test by bench.py
    results = {split: score(retained, split, run/"final/solutions"/split) for split in scored}
    results.setdefault("evolution", results.get("bench-train"))
    results.setdefault("development", results.get("bench-val"))
    (run/"final/frozen_results.json").write_text(json.dumps(results["evolution"]+results["development"], indent=1, default=float)+"\n")
    for split in (() if args.bench else SUITE.sealed):
        name = {"test": "sealed_test_results.json", "test2": "sealed_test2_results.json"}.get(split, f"sealed_{split}_results.json")
        (run/"final"/name).write_text(json.dumps(results[split], indent=1, default=float)+"\n")
    manifest["final"] = {split: {"cases": len(rows), "passed": sum(r["passed"] for r in rows),
                                 "class_accuracy": sum(bool(r.get("family_correct")) for r in rows)/len(rows)}
                         for split, rows in results.items() if split in scored}
    (run/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"stop_reason": stop_reason, "cost_usd": manifest["model_cost_usd"], "final": manifest["final"]}, indent=2))


if __name__ == "__main__":
    main()
