"""Claude Code as a coding agent on the pdebench task, for comparison with the PDE-RSI harness.

The agent runs headless (`claude -p`, stream-json transcript) inside bubblewrap. It has network access (for
Bedrock only), a workspace with the public case views, noisy data, the equation library, and the solver
contract, and a fresh empty HOME. The project, references, sealed cases, and the user's home are not mounted,
and WebSearch/WebFetch are disabled. A host-side evaluation service (this process) runs the agent's
/work/solver.py on training cases (diagnostics with labels and the program's prints) or validation cases (no
labels) through the same GPU solver sandbox as the harness. The agent is re-prompted with --continue until the
wall-clock deadline, or until two continued sessions in a row leave solver.py unchanged; /work/solver.py is then
frozen and scored on all public cases and the sealed test.

Usage on a GPU node (after sourcing scripts/env.sh):
    python scripts/run_claude_code_agent.py --model us.anthropic.claude-opus-5-5 \
        --output runs/claudecode-pdebench-opus55-20261017 --budget-seconds 7200 --eval-core 90 --agent-core 91
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import threading
import time
import types
import uuid

import numpy as np

from pde_rsi.codegen import ProgramRejected, check_program, gpu_env_prefix
from pde_rsi.pdebench import CONTRACT, LIBRARY, PDEBenchProtocol, PROGRAM_KEY, make_cases, run_case
from pde_rsi.runner import Budget
from pde_rsi.storage import PROJECT_ROOT, project_path

CLAUDE_BINARY = Path.home()/".local/bin/claude"
SECRET = PROJECT_ROOT/".runtime/secrets/bedrock_api_key"
STUB = '"""No solver has been written yet."""\n\n\ndef solve(problem):\n    raise NotImplementedError("No solver yet")\n'
CLIENT = r'''"""Evaluate /work/solver.py on training or validation cases through the host evaluation service.

    python tools/evaluate.py train                 # all 16 training cases (labels, errors, your prints)
    python tools/evaluate.py validation            # all 16 validation cases (errors only, no labels)
    python tools/evaluate.py train --cases case-t00,case-t05
    python tools/evaluate.py --wait <request_id>   # collect a request that was still running
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
split = args[0] if args else "train"
if split not in ("train", "validation"):
    sys.exit("split must be train or validation")
cases = args[args.index("--cases")+1].split(",") if "--cases" in args else None
request_id = uuid.uuid4().hex[:12]
request = {"request_id": request_id, "split": split, "cases": cases, "code": open("/work/solver.py").read()}
tmp = f"/work/requests/.{request_id}.tmp"
json.dump(request, open(tmp, "w"))
os.rename(tmp, f"/work/requests/{request_id}.json")
print(f"submitted {request_id}", flush=True)
wait(request_id)
'''


def readme(cases, deadline_text, budget_minutes):
    views = "\n".join(f"- `cases/{c.case_id}/` ({'training' if c.split == 'evolution' else 'validation'})"
                      for c in cases)
    return f"""# Task: identify and solve PDE/FPDE problems from data

You have **{budget_minutes} minutes** of wall-clock time (deadline: {deadline_text}). Your result is the file
`/work/solver.py` exactly as it is at the deadline. It is then scored on these public cases and on held-out test
cases of the same kinds that you never see.

## What to write

{CONTRACT}

library = {json.dumps(LIBRARY)}

## Files

- `cases/<case_id>/view.json`: the public problem attributes of a case (what `problem` contains besides the
  callables `forcing`, `initial`, `initial_velocity`, `boundary`, and `device`).
- `cases/<case_id>/data.npz`: `data_x`, `data_y`, `data_t` (absent where the case has no such coordinate), `data_u`.
- `solver.py`: your program (currently a stub).
- `tools/evaluate.py`: runs `solver.py` in the official sandbox on training or validation cases and prints the
  diagnostics. Training diagnostics include the true class and parameters and the last 2000 characters of your
  program's prints; validation diagnostics give errors only. The forcing, initial, and boundary data exist only
  inside these evaluations (the sandbox passes them to `solve`), so test your program through this tool.
  A full split can take several minutes: call it with a Bash timeout of 600000 ms, or use `--cases` for a subset.

Python with numpy, scipy, and torch (CUDA) is available as `python`. No internet access is available or needed.

Cases:
{views}
"""


def sandbox(run, model, agent_core, token):
    gpu = []
    for device in sorted(Path("/dev").glob("nvidia*")):
        if device.is_char_device():
            gpu += ["--dev-bind", str(device), str(device)]
    command = ["taskset", "-c", str(agent_core), "bwrap", "--unshare-all", "--share-net", "--die-with-parent",
               "--new-session", "--cap-drop", "ALL", "--clearenv", "--ro-bind", "/usr", "/usr",
               "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/lib", "/lib", "--symlink", "usr/bin", "/bin",
               "--ro-bind", "/etc", "/etc", "--ro-bind", str(CLAUDE_BINARY.resolve()), "/opt/claude/claude",
               "--ro-bind", str(gpu_env_prefix().resolve()), "/opt/gpu-env", "--ro-bind", "/sys", "/sys",
               "--bind", str(run/"workspace"), "/work", "--bind", str(run/"agent_home"), "/home/agent",
               "--proc", "/proc", "--dev", "/dev", *gpu, "--tmpfs", "/tmp", "--chdir", "/work"]
    env = {"HOME": "/home/agent", "PATH": "/opt/gpu-env/bin:/usr/bin:/bin", "CLAUDE_CODE_USE_BEDROCK": "1",
           "AWS_REGION": "us-east-1", "AWS_BEARER_TOKEN_BEDROCK": token, "ANTHROPIC_MODEL": model,
           "ANTHROPIC_SMALL_FAST_MODEL": model, "DISABLE_AUTOUPDATER": "1", "DISABLE_TELEMETRY": "1",
           "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1"}
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    for key, value in env.items():
        command += ["--setenv", key, value]
    return command+["/opt/claude/claude"]


class EvaluationService(threading.Thread):
    """Answers /work/requests/*.json with harness diagnostics; logs every request."""

    def __init__(self, run, cases):
        super().__init__(daemon=True)
        self.directory, self.cases = run, cases
        self.protocol = PDEBenchProtocol()
        self.protocol.cases = {c.task_id: c for c in cases}
        self.stop = threading.Event()
        self.counts = {"train": 0, "validation": 0, "rejected": 0, "solves": 0}

    def answer(self, request):
        split = request.get("split")
        try:
            check_program(request["code"], PROGRAM_KEY)
        except ProgramRejected as error:
            self.counts["rejected"] += 1
            return {"request_id": request["request_id"], "status": "rejected_by_static_policy", "message": str(error)}
        wanted = {"train": "evolution", "validation": "development"}[split]
        tasks = [c for c in self.cases if c.split == wanted and (not request.get("cases") or c.case_id in request["cases"])]
        budget = Budget(max(1, len(tasks)), 7200, max_cpu_seconds=7200)
        results = self.protocol.evaluate(tasks, types.SimpleNamespace(program={PROGRAM_KEY: request["code"]}), budget)
        self.counts[split] += 1
        self.counts["solves"] += len(tasks)
        rows = self.protocol.diagnostics(results)
        if split == "train":
            for row, result in zip(rows, results):
                row["program_output"] = result.get("program_output")
        passes = sum(r["passed"] for r in results)
        return {"request_id": request["request_id"], "split": split, "cases": len(tasks), "passed": passes,
                "diagnostics": rows}

    def run(self):
        requests, responses = self.directory/"workspace/requests", self.directory/"workspace/responses"
        while not self.stop.is_set():
            for path in sorted(requests.glob("*.json"), key=lambda p: p.stat().st_mtime):
                started = time.time()
                try:
                    request = json.loads(path.read_text())
                    response = self.answer(request)
                except Exception as error:  # malformed requests are answered, never crash the service
                    request = {"request_id": path.stem}
                    response = {"request_id": path.stem, "status": "error", "message": str(error)[:500]}
                tmp = responses/f".{path.stem}.tmp"
                tmp.write_text(json.dumps(response, indent=1, default=str))
                tmp.rename(responses/f"{path.stem}.json")
                archive = self.directory/"evaluation_requests"/path.name
                shutil.move(str(path), archive)
                with (self.directory/"evaluation_log.jsonl").open("a") as stream:
                    stream.write(json.dumps({"request_id": path.stem, "split": request.get("split"),
                                             "cases": request.get("cases"), "status": response.get("status", "ok"),
                                             "passed": response.get("passed"), "seconds": time.time()-started,
                                             "utc": datetime.now(timezone.utc).isoformat()})+"\n")
            time.sleep(1)


def usage_from_transcript(path):
    """Token totals of a stream-json transcript, valid for sessions killed at the deadline too.

    With --include-partial-messages, message_start carries a message's input/cache tokens and the last
    message_delta its final output tokens. Assistant events alone report output tokens at message start only."""
    usage, current = {}, None
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "stream_event":
            inner = event.get("event") or {}
            if inner.get("type") == "message_start":
                current = inner["message"]["id"]
                usage[current] = dict(inner["message"].get("usage") or {})
            elif inner.get("type") == "message_delta" and current in usage:
                usage[current].update({k: v for k, v in (inner.get("usage") or {}).items() if v is not None})
        elif event.get("type") == "assistant":
            message = event.get("message") or {}
            usage.setdefault(message.get("id"), dict(message.get("usage") or {}))
    keys = {"input": "input_tokens", "output": "output_tokens", "cache_read": "cache_read_input_tokens",
            "cache_write": "cache_creation_input_tokens"}
    return {name: sum(item.get(key) or 0 for item in usage.values()) for name, key in keys.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=7200)
    parser.add_argument("--eval-core", type=int, required=True)
    parser.add_argument("--agent-core", type=int, required=True)
    args = parser.parse_args()
    run = project_path(args.output)
    run.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {args.eval_core})  # the evaluation service and final scoring run on this core
    for sub in ("workspace/cases", "workspace/tools", "workspace/requests", "workspace/responses", "agent_home",
                "evaluation_requests", "transcripts", "final"):
        (run/sub).mkdir(parents=True)
    cases = make_cases("evolution")+make_cases("development")
    for case in cases:
        folder = run/"workspace/cases"/case.case_id
        folder.mkdir()
        (folder/"view.json").write_text(json.dumps({**case.public_view(), "library": LIBRARY}, indent=1))
        data = case.data()
        np.savez(folder/"data.npz", data_u=data["u"],
                 **{f"data_{k}": data[k] for k in ("x", "y", "t") if data[k] is not None})
    (run/"workspace/solver.py").write_text(STUB)
    (run/"workspace/tools/evaluate.py").write_text(CLIENT)
    start = time.time()
    deadline = start+args.budget_seconds
    deadline_text = datetime.fromtimestamp(deadline).astimezone().strftime("%H:%M:%S %Z")
    (run/"workspace/README.md").write_text(readme(cases, deadline_text, round(args.budget_seconds/60)))
    manifest = {"kind": "claude_code_agent", "model": args.model, "claude_code_version":
                subprocess.run([str(CLAUDE_BINARY), "--version"], capture_output=True, text=True).stdout.strip(),
                "budget_seconds": args.budget_seconds, "start_utc": datetime.now(timezone.utc).isoformat(),
                "hardware": {"node": platform.node(), "eval_core": args.eval_core, "agent_core": args.agent_core},
                "disallowed_tools": ["WebSearch", "WebFetch"], "sessions": []}
    (run/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    service = EvaluationService(run, cases)
    service.start()
    token = SECRET.read_text().strip()
    base = sandbox(run, args.model, args.agent_core, token)
    session, idle = 0, 0
    # Re-prompt only while the agent still works: two continued sessions in a row that leave solver.py unchanged
    # end the run early (each --continue reloads the whole conversation, so idle re-prompts are not free).
    while deadline-time.time() > 300 and idle < 2:
        before = (run/"workspace/solver.py").read_text()
        session += 1
        remaining = deadline-time.time()
        prompt = ("Read /work/README.md and complete the task. Work autonomously; nobody will answer questions."
                  if session == 1 else
                  f"You still have {remaining/60:.0f} minutes before the deadline. Keep improving /work/solver.py "
                  "(check it with tools/evaluate.py) and stop only when you are confident it cannot be improved.")
        command = base+["-p", prompt, "--dangerously-skip-permissions", "--disallowedTools", "WebSearch", "WebFetch",
                        "--output-format", "stream-json", "--verbose", "--include-partial-messages"]+(["--continue"] if session > 1 else [])
        transcript = run/"transcripts"/f"session-{session:02d}.jsonl"
        began = time.time()
        with transcript.open("w") as stream:
            try:
                proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=remaining)
                status = f"exit {proc.returncode}"
            except subprocess.TimeoutExpired:
                status = "killed at deadline"
        changed = (run/"workspace/solver.py").read_text() != before
        idle = 0 if changed or session == 1 else idle+1
        manifest["sessions"].append({"session": session, "status": status, "seconds": time.time()-began,
                                     "solver_changed": changed, "usage": usage_from_transcript(transcript)})
        manifest["stopped_early_idle"] = idle >= 2
        (run/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    service.stop.set()
    service.join(timeout=3600)
    code = (run/"workspace/solver.py").read_text()
    (run/"final/solver.py").write_text(code)
    manifest["improvement_wall_seconds"] = time.time()-start
    manifest["evaluation_requests"] = service.counts
    results = {}
    for split in ("evolution", "development", "test"):
        target = run/"final/solutions"
        target.mkdir(exist_ok=True)
        rows = []
        for case in make_cases(split):
            try:
                check_program(code, PROGRAM_KEY)
                rows.append(run_case(case, code, Budget(1, 3600, max_cpu_seconds=3600), target/f"{case.case_id}.npz"))
            except ProgramRejected as error:
                rows.append({"case_id": case.case_id, "task_id": case.task_id, "split": case.split, "status":
                             "rejected", "passed": False, "relative_l2": 1e6, "message": str(error),
                             "family_correct": False})
        results[split] = rows
    (run/"final/frozen_results.json").write_text(json.dumps(results["evolution"]+results["development"], indent=1,
                                                            default=float)+"\n")
    (run/"final/sealed_test_results.json").write_text(json.dumps(results["test"], indent=1, default=float)+"\n")
    manifest["final"] = {split: {"cases": len(rows), "passed": sum(r["passed"] for r in rows),
                                 "class_accuracy": sum(bool(r.get("family_correct")) for r in rows)/len(rows)}
                         for split, rows in results.items()}
    manifest["usage_total"] = {k: sum(s["usage"][k] for s in manifest["sessions"])
                               for k in ("input", "output", "cache_read", "cache_write")}
    (run/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest["final"], indent=2))


if __name__ == "__main__":
    main()
