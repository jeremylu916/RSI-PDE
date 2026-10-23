"""Project-local CLI; no model API calls, network searches, or job submissions."""

import argparse
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path

from .experiment import compare, continuation_probe, finalize, selected_tasks, write_json
from .policy import AgentVersion, SolverSettings
from .runner import Budget, run_suite, run_task
from .evaluator import summarize, evaluate
from .solvers import Solution
from .storage import doctor, project_path
from .tasks import write_pilot


def root_path():
    return Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root_path())
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Verify the depot interpreter, temporary files, and cache paths")
    sub.add_parser("init", help="Generate/validate the 72-case grouped pilot")
    export = sub.add_parser("export-hf", help="Prepare a local public-only Hugging Face dataset package")
    export.add_argument("--output", type=Path, default=root_path()/"datasets/huggingface/pde-problems")
    validate = sub.add_parser("validate-hf", help="Verify all exported PDE configs through the local HF loader")
    validate.add_argument("package", type=Path)
    agents = sub.add_parser("agents", help="Run model-guided single-session and RSI agents and report public-task L2")
    agents.add_argument("--model-config", type=Path, required=True)
    agents.add_argument("--package", type=Path, default=root_path()/"datasets/huggingface/pde-problems")
    agents.add_argument("--rounds", type=int, default=3)
    agents.add_argument("--probe-rounds", type=int, default=1)
    agents.add_argument("--max-calls", type=int)
    agents.add_argument("--max-model-calls", type=int)
    agents.add_argument("--wall-seconds", type=float, default=900)
    agents.add_argument("--cpu-seconds", type=float, default=600)
    agents.add_argument("--output", type=Path)
    code = sub.add_parser("code-agents", help="Run the code-writing single run (one request) and RSI agents and report public-task L2")
    code.add_argument("--model-config", type=Path, required=True)
    code.add_argument("--package", type=Path, default=root_path()/"datasets/huggingface/pde-problems")
    code.add_argument("--rounds", type=int, default=6)
    code.add_argument("--probe-rounds", type=int, default=2)
    code.add_argument("--max-calls", type=int)
    code.add_argument("--max-model-calls", type=int)
    code.add_argument("--wall-seconds", type=float, default=5400)
    code.add_argument("--cpu-seconds", type=float, default=3600)
    code.add_argument("--max-input-chars", type=int, default=500000)
    code.add_argument("--model-timeout", type=float, default=900)
    code.add_argument("--families", default="all",
                      help="all, 1d, wave2d, group, or a comma-separated list of families")
    code.add_argument("--protocol", choices=["code", "identification", "student", "pdebench", "inventory", "pdebench-ext"],
                      default="code",
                      help="code: solve given equations; identification: choose the equation from data, then solve; "
                           "pdebench: the student protocol on the pde_benchmarks.tex cases")
    code.add_argument("--time-budget-seconds", type=float, default=None,
                      help="improve until this wall-clock budget per arm instead of a fixed number of rounds")
    code.add_argument("--concurrent-arms", action="store_true",
                      help="run the single run and RSI at the same time, each pinned to its own CPU core")
    code.add_argument("--cores", default=None, help="two comma-separated CPU cores for --concurrent-arms")
    code.add_argument("--history-rounds", type=int, default=None,
                      help="replay only the last N rounds of the conversation (both arms)")
    code.add_argument("--noise-fraction", type=float, default=None,
                      help="pdebench: data noise std as a fraction of the signal RMS (default 0.01)")
    code.add_argument("--max-validation-submissions", type=int, default=None,
                      help="cap on validation-gate decisions per arm")
    code.add_argument("--spend-cap-usd", type=float, default=None,
                      help="stop new model requests once an arm's priced model cost reaches this (USD)")
    code.add_argument("--modular", action="store_true",
                      help="pdebench/inventory, harness v2: modular program parts with patch edits, a per-part gate, "
                           "and run-time self-check guidance (src/pde_rsi/modular.py)")
    code.add_argument("--hide-time-dependence", action="store_true",
                      help="pdebench: pose steady cases in time and hide the time_dependent/coordinates fields")
    code.add_argument("--attempts-per-round", type=int, default=3,
                      help="model requests per round; each candidate is self-tested on training cases first")
    code.add_argument("--best-of", type=int, default=1,
                      help="RSI arm: proposals sampled in parallel per attempt; all are self-tested and the passing "
                           "one with the lowest training error is submitted")
    code.add_argument("--gate", choices=["standard", "pass-first"], default="standard",
                      help="RSI arm: pass-first also accepts a candidate that passes more validation cases and fails "
                           "none that passed, despite a sub-0.5%% aggregate gain")
    code.add_argument("--experiments-per-round", type=int, default=0,
                      help="RSI arm: requests per round in which the model may run a candidate on chosen training "
                           "cases before submitting (no attempt used; never validation cases)")
    code.add_argument("--bench", action="store_true",
                      help="pdebench/pdebench-ext/inventory: train and validate on the RSI-PDE Bench splits "
                           "(bench-train, bench-val) instead of evolution and development")
    code.add_argument("--arm-base-urls", default=None,
                      help="single=URL,rsi=URL: per-arm model endpoints (RSI-PDE Bench metering proxies)")
    code.add_argument("--start", choices=["classical", "scratch"], default="classical",
                      help="classical: start from working classical solvers; scratch: the agent writes every solver")
    code.add_argument("--output", type=Path)
    sub.add_parser("init-wave", help="Generate/validate the 12-case 2D wave benchmark files")
    sub.add_parser("init-group", help="Generate/validate the 20-case group FPDE benchmark files")
    baseline = sub.add_parser("baseline", help="Evaluate the initial classical solver")
    baseline.add_argument("--split", choices=["evolution", "development"], default="development")
    baseline.add_argument("--limit-per-family", type=int, default=0)
    baseline.add_argument("--output", type=Path)
    comparison = sub.add_parser("compare", help="Budgeted single/fixed/RSI development run; no final-test reads")
    comparison.add_argument("--generations", type=int, default=4)
    comparison.add_argument("--max-calls", type=int, default=240)
    comparison.add_argument("--wall-seconds", type=float, default=600)
    comparison.add_argument("--cpu-seconds", type=float, help="Aggregate parent+child CPU ceiling (default: wall ceiling)")
    comparison.add_argument("--single-mode", choices=["session", "static"], default="session")
    comparison.add_argument("--limit-per-family", type=int, default=0)
    comparison.add_argument("--output", type=Path)
    final = sub.add_parser("finalize", help="Once-per-run sealed evaluation of frozen agents")
    final.add_argument("run_directory", type=Path)
    probe = sub.add_parser("probe", help="Equal-budget continuation from identical solver/memory")
    probe.add_argument("snapshot", type=Path)
    probe.add_argument("--generations", type=int, default=2)
    probe.add_argument("--max-calls", type=int, default=120)
    probe.add_argument("--wall-seconds", type=float, default=300)
    probe.add_argument("--cpu-seconds", type=float)
    probe.add_argument("--limit-per-family", type=int, default=0)
    probe.add_argument("--output", type=Path)
    solve = sub.add_parser("solve", help="Produce solution.npz and metered metrics for one public task")
    solve.add_argument("task_id")
    solve.add_argument("--split", choices=["evolution", "development"], default="development")
    solve.add_argument("--nx", type=int, default=32)
    solve.add_argument("--nt", type=int, default=48)
    solve.add_argument("--grading", type=float, default=1)
    solve.add_argument("--advection", choices=["central", "upwind"], default="central")
    solve.add_argument("--output", type=Path)
    external = sub.add_parser("evaluate-solution", help="Accuracy and IC/BC checks for a local NPZ; resources uncertified")
    external.add_argument("task_id")
    external.add_argument("solution", type=Path)
    external.add_argument("--split", choices=["evolution", "development"], default="development")
    external.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        doctor()
        for name in ("root", "output", "run_directory", "snapshot", "solution", "package", "model_config"):
            value = getattr(args, name, None)
            if value is not None:
                setattr(args, name, project_path(value))
    except ValueError as error:
        parser.error(str(error))
    for name in ("limit_per_family", "generations"):
        if hasattr(args, name) and getattr(args, name) < 0:
            parser.error(f"{name} must be nonnegative")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if args.command == "doctor":
        result = doctor()
    elif args.command == "init":
        result = write_pilot(args.root)
    elif args.command == "export-hf":
        from .hf_dataset import export_package
        result = export_package(args.root, args.output)
    elif args.command == "validate-hf":
        from .hf_dataset import validate_package
        result = validate_package(args.package)
    elif args.command == "agents":
        from .agent_runs import run_comparison
        from .model_backend import ModelConfig
        config = ModelConfig(**json.loads(args.model_config.read_text()))
        output = args.output or args.root/"runs"/f"model-agents-{timestamp}"
        metadata = run_comparison(args.root, args.package, output, config, rounds=args.rounds,
                                  max_calls=args.max_calls, wall_seconds=args.wall_seconds,
                                  cpu_seconds=args.cpu_seconds, max_model_calls=args.max_model_calls,
                                  probe_rounds=args.probe_rounds)
        result = {"output": str(output), "comparison": metadata["comparison"],
                  "continuation_policy_advantage": metadata.get("continuation_policy_advantage"),
                  "matched_solver_calls": metadata["matched_actual_solver_calls"],
                  "matched_model_calls": metadata["matched_actual_model_calls"],
                  "all_rounds_complete": metadata["all_requested_rounds_complete"]}
    elif args.command == "init-group":
        from .tasks import write_group_benchmark
        result = write_group_benchmark(args.root)
    elif args.command == "init-wave":
        from .tasks import write_wave_benchmark
        result = write_wave_benchmark(args.root)
    elif args.command == "code-agents":
        from .code_agents import run_comparison as run_code_comparison
        from .model_backend import ModelConfig
        from .tasks import ALL_FAMILIES, FAMILIES as ONE_D, GROUP_FAMILIES
        families = {"all": ALL_FAMILIES, "1d": ONE_D, "group": GROUP_FAMILIES}.get(
            args.families, tuple(args.families.split(",")))
        protocol = None
        if args.protocol in ("identification", "student"):
            from .identification import PROGRAM_KEY, IdentificationProtocol, StudentProtocol
            protocol = StudentProtocol() if args.protocol == "student" else IdentificationProtocol()
            families = (PROGRAM_KEY,)
        elif args.protocol == "pdebench-ext":
            from .pdebench_ext import PROGRAM_KEY, PDEBenchExtProtocol
            protocol, families = PDEBenchExtProtocol(modular=args.modular), (PROGRAM_KEY,)
        elif args.protocol == "inventory":
            from .inventory import PROGRAM_KEY, InventoryProtocol
            protocol, families = InventoryProtocol(modular=args.modular), (PROGRAM_KEY,)
        elif args.protocol == "pdebench":
            from .pdebench import PROGRAM_KEY, PDEBenchProtocol
            from .pdebench import NOISE_FRACTION
            protocol = PDEBenchProtocol(NOISE_FRACTION if args.noise_fraction is None else args.noise_fraction,
                                        hide_time_dependence=args.hide_time_dependence, modular=args.modular)
            families = (PROGRAM_KEY,)
        if args.bench:
            if args.protocol not in ("pdebench", "pdebench-ext", "inventory"):
                raise SystemExit("--bench needs --protocol pdebench, pdebench-ext or inventory")
            protocol.split_names = ("bench-train", "bench-val")
        config = ModelConfig(**json.loads(args.model_config.read_text()))
        output = args.output or args.root/"runs"/f"code-agents-{timestamp}"
        metadata = run_code_comparison(args.root, args.package, output, config, rounds=args.rounds,
                                       max_calls=args.max_calls, wall_seconds=args.wall_seconds,
                                       cpu_seconds=args.cpu_seconds, max_model_calls=args.max_model_calls,
                                       probe_rounds=args.probe_rounds, max_input_chars=args.max_input_chars,
                                       model_timeout=args.model_timeout, start=args.start,
                                       attempts_per_round=args.attempts_per_round, families=families,
                                       protocol=protocol, time_budget_seconds=args.time_budget_seconds,
                                       concurrent_arms=args.concurrent_arms,
                                       cores=[int(c) for c in args.cores.split(",")] if args.cores else None,
                                       history_rounds=args.history_rounds,
                                       max_validation_submissions=args.max_validation_submissions,
                                       spend_cap_usd=args.spend_cap_usd, best_of=args.best_of, gate=args.gate,
                                       experiments_per_round=args.experiments_per_round,
                                       arm_base_urls=dict(item.split("=", 1) for item in args.arm_base_urls.split(","))
                                       if args.arm_base_urls else None)
        result = {"output": str(output), "comparison": metadata["comparison"],
                  "continuation_policy_advantage": metadata.get("continuation_policy_advantage"),
                  "matched_solver_calls": metadata["matched_actual_solver_calls"],
                  "matched_model_calls": metadata["matched_actual_model_calls"],
                  "all_rounds_complete": metadata["all_requested_rounds_complete"]}
    elif args.command == "baseline":
        tasks = selected_tasks(args.root, args.split, args.limit_per_family)
        results = run_suite(tasks, AgentVersion(), Budget(len(tasks), len(tasks)*15+10))
        output = args.output or args.root/"runs"/f"baseline-{timestamp}"
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/"results.json", results)
        result = summarize(results)
        write_json(output/"summary.json", result)
        result = {"output": str(output), **result}
    elif args.command == "compare":
        output = args.output or args.root/"runs"/f"compare-{timestamp}"
        metadata = compare(args.root, output, args.generations, args.max_calls, args.wall_seconds, args.limit_per_family,
                           single_mode=args.single_mode, cpu_seconds=args.cpu_seconds)
        result = {"output": str(output), "arms": metadata["arms"]}
    elif args.command == "finalize":
        metadata = finalize(args.root, args.run_directory)
        result = {"output": str(args.run_directory), "final_evaluated": True,
                  "arms": {arm: report["final"] for arm, report in metadata["arms"].items()}}
    elif args.command == "probe":
        output = args.output or args.root/"runs"/f"probe-{timestamp}"
        metadata = continuation_probe(args.root, args.snapshot, output, args.generations, args.max_calls,
                                      args.wall_seconds, args.limit_per_family, cpu_seconds=args.cpu_seconds)
        result = {"output": str(output), "matched_actual_solve_calls": metadata["matched_actual_solve_calls"],
                  "complete_paired_continuations": metadata["complete_paired_continuations"],
                  "evolved_policy_advantage": metadata["evolved_policy_advantage"]}
    else:
        matches = [t for t in selected_tasks(args.root, args.split) if t.task_id == args.task_id]
        if len(matches) != 1:
            parser.error(f"Unknown public task ID in {args.split}: {args.task_id}")
        task = matches[0]
        output = args.output or args.root/"runs"/f"{args.command}-{timestamp}"
        if args.command == "solve":
            settings = SolverSettings(args.nx, args.nt, args.grading, args.advection)
            try:
                settings.validate()
            except ValueError as error:
                parser.error(str(error))
            output.mkdir(parents=True, exist_ok=False)
            write_json(output/"task.json", task.public_record())
            write_json(output/"settings.json", vars(settings))
            budget = Budget(1, task.max_seconds, ledger_path=output/"solve_calls.jsonl", phase="solve")
            metrics = run_task(task, settings, budget, solution_path=output/"solution.npz")
            write_json(output/"metrics.json", metrics)
            source = args.root/"src/pde_rsi/solvers.py"
            result = {"output": str(output), "task_id": task.task_id, "metrics": metrics,
                      "budget": budget.report(), "solver_source": str(source),
                      "solver_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        else:
            metrics = evaluate(task, Solution.load_npz(args.solution))
            # A submitted NPZ cannot prove its runtime, memory, or compute use.
            metrics["numerical_pass"] = metrics["accuracy_pass"] and metrics["constraints_pass"]
            metrics["resources_pass"] = metrics["passed"] = None
            for field in ("runtime_seconds", "wall_seconds", "peak_memory_mb"):
                metrics[field] = None
            result = {"evaluation_scope": "accuracy_and_constraints_only", "solution": str(args.solution),
                      "metrics": metrics}
            output.mkdir(parents=True, exist_ok=False)
            result["output"] = str(output)
        write_json(output/"report.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
