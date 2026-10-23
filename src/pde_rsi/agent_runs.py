"""Model-guided single-session and recursive policy experiments on public PDE tasks."""

import copy
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .evaluator import EVALUATOR_VERSION, acceptance_gate, failed_result, summarize
from .experiment import append_event, selected_tasks, write_json
from .model_backend import ModelBackend, ModelCallError, ModelConfig
from .policy import SolverSettings
from .runner import Budget, BudgetExceeded, run_task
from .storage import PROJECT_ROOT, project_path
from .tasks import FAMILIES, Task

INITIAL_WORKFLOW = (
    "Inspect errors by family and instance. Identify spatial, temporal, transport, or initial-time "
    "regularity limitations. Propose bounded numerical settings, favoring transferable improvements "
    "while preserving accuracy and resource constraints. Use only the supplied task and diagnostic data."
)
METRICS = ("absolute_l2", "relative_l2", "final_absolute_l2", "final_relative_l2")


@dataclass
class LoopVersion:
    model: str
    solver: dict = field(default_factory=lambda: {family: SolverSettings() for family in FAMILIES})
    improvement_workflow: str = INITIAL_WORKFLOW
    memory: list = field(default_factory=list)
    generation: int = 0
    parent_id: str | None = None

    def to_dict(self):
        return asdict(self)

    @property
    def version_id(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:20]

    def snapshot(self):
        return {"version_id": self.version_id, "agent": self.to_dict()}


def proposal_schema():
    settings = {
        "type": "object", "additionalProperties": False,
        "properties": {"family": {"type": "string", "enum": list(FAMILIES)},
                       "nx": {"type": "integer", "minimum": 8, "maximum": 128},
                       "nt": {"type": "integer", "minimum": 8, "maximum": 384},
                       "grading": {"type": "number", "minimum": 1, "maximum": 4},
                       "advection": {"type": "string", "enum": ["central", "upwind"]}},
        "required": ["family", "nx", "nt", "grading", "advection"],
    }
    return {"type": "object", "additionalProperties": False,
            "properties": {"hypothesis": {"type": "string", "minLength": 1, "maxLength": 3000},
                           "solver": {"type": "array", "items": settings, "minItems": 6, "maxItems": 6},
                           "next_improvement_policy": {"type": ["string", "null"], "minLength": 1, "maxLength": 2048},
                           "memory_update": {"type": "string", "maxLength": 1200}},
            "required": ["hypothesis", "solver", "next_improvement_policy", "memory_update"]}


def apply_proposal(version, record, recursive):
    if set(record) != {"hypothesis", "solver", "next_improvement_policy", "memory_update"}:
        raise ValueError("Proposal requires exactly hypothesis, solver, next_improvement_policy, memory_update")
    if not isinstance(record["hypothesis"], str) or not 1 <= len(record["hypothesis"]) <= 3000:
        raise ValueError("Invalid proposal hypothesis")
    if not isinstance(record["memory_update"], str) or len(record["memory_update"]) > 1200:
        raise ValueError("Invalid bounded memory update")
    policy = record["next_improvement_policy"]
    if policy is not None and (not isinstance(policy, str) or not 1 <= len(policy) <= 2048):
        raise ValueError("Improvement workflow must be null or a bounded instruction string")
    if not recursive and policy is not None:
        raise ValueError("A fixed-controller single-session run cannot update its improvement policy")
    rows = record["solver"]
    if not isinstance(rows, list) or len(rows) != len(FAMILIES):
        raise ValueError("Proposal must contain one solver setting for each PDE family")
    solver = {}
    for row in rows:
        if set(row) != {"family", "nx", "nt", "grading", "advection"}:
            raise ValueError("Unknown numerical setting or missing solver field")
        family = row["family"]
        if family not in FAMILIES or family in solver:
            raise ValueError("Solver families must be unique and cover the public benchmark")
        try:
            settings = SolverSettings(**{k: row[k] for k in ("nx", "nt", "grading", "advection")}).validate()
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid settings for {family}: {error}") from error
        if settings.nx > 128 or settings.nt > 384:
            raise ValueError("This protocol caps the grids at nx=128 and nt=384")
        solver[family] = settings
    candidate = copy.deepcopy(version)
    candidate.solver = solver
    if recursive:
        candidate.parent_id = version.version_id
        candidate.generation += 1
        candidate.improvement_workflow = policy if policy is not None else version.improvement_workflow
        candidate.memory.append({"generation": candidate.generation, "hypothesis": record["hypothesis"],
                                 "lesson": record["memory_update"]})
    return candidate


def metric_summary(results):
    base = summarize(results)
    for metric in METRICS:
        valid = [row[metric] for row in results if row.get(metric) is not None and row["status"] == "ok"]
        base["mean_"+metric] = sum(valid)/len(valid) if valid else None
    base["executed_cases"] = sum(row["status"] != "budget_exhausted" for row in results)
    base["successful_cases"] = sum(row["status"] == "ok" for row in results)
    base["failed_cases"] = sum(row["status"] != "ok" for row in results)
    for family in base["families"]:
        items = [row for row in results if row["family"] == family]
        for metric in METRICS:
            valid = [row[metric] for row in items if row.get(metric) is not None and row["status"] == "ok"]
            base["families"][family]["mean_"+metric] = sum(valid)/len(valid) if valid else None
    return base


def evaluate_suite(tasks, version, budget, solution_directory=None):
    results = []
    if solution_directory is not None:
        project_path(solution_directory).mkdir(parents=True, exist_ok=True)
    for index, task in enumerate(tasks):
        try:
            path = solution_directory/f"{task.task_id}.npz" if solution_directory is not None else None
            results.append(run_task(task, version.solver[task.family], budget, solution_path=path))
        except BudgetExceeded:
            results.extend(failed_result(t, "budget_exhausted", "Task was not executed; experiment allowance exhausted", 0)
                           for t in tasks[index:])
            break
    return results


def released_diagnostics(results):
    fields = ("task_id", "family", "status", "relative_l2", "final_relative_l2", "absolute_l2",
              "final_absolute_l2", "initial_error", "boundary_error", "physical_residual_relative",
              "runtime_seconds", "wall_seconds", "peak_memory_mb", "accuracy_pass", "constraints_pass",
              "resources_pass", "passed", "nx", "nt", "message")
    return [{key: row.get(key) for key in fields} for row in results]


def public_tasks(root, package):
    # These are the exact native records in the uploaded dataset, checked against
    # the independently evaluated local public benchmark before using its reference.
    from .hf_dataset import validate_package
    validate_package(package)
    tasks = {}
    for source_split, hf_split in (("evolution", "train"), ("development", "validation")):
        expected = {t.task_id: t for t in selected_tasks(root, source_split)}
        records = []
        for family in FAMILIES:
            path = package/f"records/{family}/{hf_split}.jsonl"
            for line in path.read_text().splitlines():
                row = json.loads(line)
                task = Task.from_dict(row["task_record"])
                if task.task_id not in expected or task.to_dict() != expected[task.task_id].to_dict():
                    raise ValueError("Dataset record has no matching independently evaluated public task")
                records.append(task)
        if len(records) != len(expected):
            raise ValueError("Public dataset coverage differs from the benchmark")
        tasks[source_split] = records
    return tasks


def run_arm(directory, starting, evolution, development, rounds, recursive, budget, backend,
            max_model_calls, max_input_chars=180000, model_timeout=180):
    directory = project_path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    version = copy.deepcopy(starting)
    budget.ledger_path = directory/"solve_calls.jsonl"
    budget.phase = "initial_validation"
    feedback = evaluate_suite(development, version, budget)
    write_json(directory/"initial_validation.json", feedback)
    write_json(directory/"initial_agent.json", version.snapshot())
    initial = metric_summary(feedback)
    all_tasks = evolution+development
    system = (
        "You are a numerical PDE/FPDE agent selecting settings for trusted classical solvers. "
        "Return only one JSON object matching the supplied schema, without explanatory prose or Markdown. "
        "Respect every field type, lower/upper bound, and string-length limit. You cannot execute code or tools. "
        "Use only these public task specifications and released numerical diagnostics. "
        "Never request references, sealed cases, external search, evaluator changes, or guessed exact solutions. "
        "The allowed solvers are finite differences, backward Euler, IMEX Burgers, and graded Caputo L1. "
        "Minimize macro relative space-time L2 error while preserving each family's pass rate and "
        "avoiding instance/final-time regressions. Do not increase every setting blindly. "
        "All six families must appear exactly once. Every nx and nt must be an integer: "
        "8<=nx<=128 and 8<=nt<=384, grading in [1,4]. "
        "Poisson still requires nt>=8 in the common settings schema (use nt=48); "
        "its stationary solver ignores that field. Never return nt=0. "
        "Use the supplied active_improvement_policy as the search procedure for the current round, "
        "and use inherited memory to interpret diagnostics and select numerical changes. "
        "A candidate must gain at least 0.5% macro accuracy and satisfy resource/IC/BC checks. "
        + ("You may also propose a bounded next improvement-policy instruction and reusable memory lesson. "
           "These changes are inherited only if the numerical candidate is accepted, and then drive later proposals."
           if recursive else
           "The improvement controller remains fixed throughout this session. next_improvement_policy MUST be null. "
           "Your solver revisions and prior feedback may be used within this session; no new persistent policy or memory is inherited.")
    )
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps({
                    "public_problems": [task.public_record() for task in all_tasks],
                    "proposal_schema": proposal_schema()}, allow_nan=False)}]
    accepted = completed = requests = 0
    model_seconds = 0.0
    usage_records, resolved_models = [], []
    for iteration in range(1, rounds+1):
        if requests >= max_model_calls:
            append_event(directory/"events.jsonl", {"iteration": iteration, "event": "model_call_budget_exhausted"})
            break
        try:
            budget.reserve(len(evolution)+len(development)+len(all_tasks))
        except BudgetExceeded:
            append_event(directory/"events.jsonl", {"iteration": iteration, "event": "solver_budget_exhausted"})
            break
        budget.phase = f"evolution_{iteration:03d}"
        evolution_feedback = evaluate_suite(evolution, version, budget)
        write_json(directory/f"evolution-{iteration:03d}.json", evolution_feedback)
        context = {"iteration": iteration, "active_agent": version.snapshot(),
                   "active_improvement_policy": version.improvement_workflow,
                   "evolution_diagnostics": released_diagnostics(evolution_feedback),
                   "validation_diagnostics": released_diagnostics(feedback),
                   "remaining_solver_calls": budget.max_solver_calls-budget.solver_calls-len(all_tasks),
                   "remaining_wall_seconds": budget.remaining_seconds,
                   "remaining_cpu_seconds": budget.remaining_cpu_seconds}
        messages.append({"role": "user", "content": json.dumps(context, allow_nan=False)})
        if sum(len(message["content"]) for message in messages) > max_input_chars:
            append_event(directory/"events.jsonl", {"iteration": iteration, "event": "input_character_budget_exhausted"})
            break
        model_directory = directory/f"model-call-{iteration:03d}"
        if budget.remaining_seconds <= 0 or budget.remaining_cpu_seconds <= 0:
            break
        try:
            result = backend.complete(messages, proposal_schema(), model_directory,
                                      timeout=min(model_timeout, budget.remaining_seconds))
            requests += 1
            model_seconds += result["wall_seconds"]
            usage_records.append(result["usage"])
            resolved = result["resolved_model"]
            if resolved:
                resolved_models.append(resolved)
                if len(set(resolved_models)) != 1:
                    append_event(directory/"events.jsonl", {"iteration": iteration, "event": "model_identity_changed",
                                 "resolved_models": resolved_models})
                    break
            proposal = result["proposal"]
            messages.append({"role": "assistant", "content": json.dumps(proposal, allow_nan=False)})
            candidate = apply_proposal(version, proposal, recursive)
            write_json(directory/f"proposal-{iteration:03d}.json", {"proposal": proposal, **candidate.snapshot()})
        except ModelCallError as error:
            requests += int(error.call_attempted)
            model_seconds += error.wall_seconds
            usage_records.append(error.usage)
            append_event(directory/"events.jsonl", {"iteration": iteration, "event": "model_failure",
                         "message": str(error), "usage": error.usage, "request_attempted": error.call_attempted})
            messages.append({"role": "user", "content": "Response rejected: "+str(error)[:1000]+
                             ". On the next allotted round return exactly one complete JSON object matching the schema."})
            continue
        except (ValueError, TypeError, KeyError) as error:
            append_event(directory/"events.jsonl", {"iteration": iteration, "event": "invalid_proposal",
                         "message": str(error)[:1000]})
            messages.append({"role": "user", "content": "Proposal rejected structurally: "+str(error)[:1000]})
            continue
        budget.phase = f"candidate_validation_{iteration:03d}"
        candidate_feedback = evaluate_suite(development, candidate, budget)
        write_json(directory/f"candidate-validation-{iteration:03d}.json", candidate_feedback)
        gate = acceptance_gate(feedback, candidate_feedback)
        completed += 1
        event = {"iteration": iteration, "parent_id": version.version_id,
                 "candidate_id": candidate.version_id, "gate": gate,
                 "candidate_validation": metric_summary(candidate_feedback), "model_usage": result["usage"],
                 "resolved_model": resolved}
        if gate["accepted"]:
            version, feedback = candidate, candidate_feedback
            accepted += 1
        event["active_version_id"] = version.version_id
        append_event(directory/"events.jsonl", event)
        messages.append({"role": "user", "content": json.dumps({"candidate_decision": event}, allow_nan=False)})
        write_json(directory/f"checkpoint-{iteration:03d}.json", version.snapshot())
    write_json(directory/"frozen_agent.json", version.snapshot())
    budget.phase = "frozen_public_evaluation"
    frozen = evaluate_suite(all_tasks, version, budget, solution_directory=directory/"solutions")
    write_json(directory/"frozen_results.json", frozen)
    report = {"model": starting.model, "resolved_models": sorted(set(resolved_models)),
              "recursive_policy_updates_enabled": recursive, "accepted_candidates": accepted,
              "accepted_generation_updates": version.generation-starting.generation,
              "requested_rounds": rounds, "completed_rounds": completed,
              "initial_validation": initial, "selected_validation": metric_summary(feedback),
              "frozen_public": metric_summary(frozen),
              "frozen_by_split": {split: metric_summary([row for row in frozen if row["split"] == split])
                                  for split in ("evolution", "development")},
              "frozen_version_id": version.version_id,
              "budget": {**budget.report(), "max_model_calls": max_model_calls, "model_calls": requests,
                         "max_input_characters_per_request": max_input_chars,
                         "max_output_tokens_per_request": backend.config.max_output_tokens,
                         "model_wall_seconds": model_seconds, "model_usage": usage_records},
              "complete_frozen_evaluation": all(row["status"] == "ok" for row in frozen)}
    write_json(directory/"report.json", report)
    return report, version, frozen


def write_tables(directory, arms):
    rows = []
    families = []
    summary = []
    for arm, (report, version, results) in arms.items():
        for result in results:
            rows.append({"arm": arm, "task_id": result["task_id"], "family": result["family"],
                         "split": result["split"], "status": result["status"],
                         **{key: result.get(key) for key in METRICS}, "passed": result["passed"],
                         "wall_seconds": result.get("wall_seconds"), "cpu_seconds": result.get("cpu_seconds"),
                         "peak_memory_mb": result.get("peak_memory_mb"),
                         **{key: getattr(version.solver[result["family"]], key) for key in ("nx", "nt", "grading", "advection")}})
        for split, split_report in report["frozen_by_split"].items():
            for family, value in split_report["families"].items():
                families.append({"arm": arm, "split": split, "family": family, "cases": value["count"],
                                 "pass_rate": value["pass_rate"],
                                 **{"mean_"+key: value["mean_"+key] for key in METRICS}})
        value = report["frozen_public"]
        summary.append({"arm": arm, "cases": value["count"], "pass_rate": value["pass_rate"],
                        **{"mean_"+key: value["mean_"+key] for key in METRICS},
                        "solver_calls": report["budget"]["solver_calls"], "model_calls": report["budget"]["model_calls"],
                        "cpu_seconds": report["budget"]["total_cpu_seconds"],
                        "wall_seconds": report["budget"]["elapsed_wall_seconds"]})
    for filename, records in (("per_case_l2.csv", rows), ("per_family_l2.csv", families), ("comparison.csv", summary)):
        with project_path(directory/filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return summary


def run_comparison(root, package, output, model_config, rounds=3, max_calls=None, wall_seconds=900,
                   cpu_seconds=600, max_model_calls=None, probe_rounds=1, backend_factory=ModelBackend):
    root, package, output = (project_path(path) for path in (root, package, output))
    if rounds < 1 or probe_rounds < 0:
        raise ValueError("At least one main round and nonnegative continuation rounds are required")
    tasks = public_tasks(root, package)
    evolution, development = tasks["evolution"], tasks["development"]
    required = len(development)+(len(evolution)+len(development))*rounds+len(evolution)+len(development)
    if max_calls is None:
        max_calls = required
    if max_calls < required:
        raise ValueError(f"Each arm needs {required} calls including frozen evaluation")
    if max_model_calls is None:
        max_model_calls = rounds
    if max_model_calls < rounds:
        raise ValueError("The model allowance must cover all requested rounds")
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"kind": "model_guided_single_vs_rsi", "created_utc": datetime.now(timezone.utc).isoformat(),
                "model_config": model_config.to_dict(), "evaluator_version": EVALUATOR_VERSION,
                "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((root/"src").rglob("*.py"))},
                "dataset_manifest_sha256": hashlib.sha256((package/"manifest.json").read_bytes()).hexdigest(),
                "task_ids": {split: [t.task_id for t in items] for split, items in tasks.items()},
                "configuration": {"rounds": rounds, "max_solver_calls_per_arm": max_calls,
                                  "max_model_calls_per_arm": max_model_calls, "max_wall_seconds_per_arm": wall_seconds,
                                  "max_cpu_seconds_per_arm": cpu_seconds, "cpu_threads": 1, "gpus": 0,
                                  "probe_rounds": probe_rounds},
                "interpretation": "model-guided settings agent; public development/selection results, not sealed-test RSI evidence",
                "arms": {}, "sealed_cases_used": False}
    write_json(output/"manifest.json", metadata)
    starting = LoopVersion(model=model_config.model)
    arms = {}
    for name in ("single", "rsi"):
        budget = Budget(max_calls, wall_seconds, max_cpu_seconds=cpu_seconds)
        arms[name] = run_arm(output/name, starting, evolution, development, rounds, name == "rsi", budget,
                             backend_factory(model_config), max_model_calls)
        metadata["arms"][name] = arms[name][0]
        write_json(output/"manifest.json", metadata)
    metadata["comparison"] = write_tables(output, arms)
    metadata["matched_actual_solver_calls"] = len({a[0]["budget"]["solver_calls"] for a in arms.values()}) == 1
    metadata["matched_actual_model_calls"] = len({a[0]["budget"]["model_calls"] for a in arms.values()}) == 1
    resolved = {m for a in arms.values() for m in a[0]["resolved_models"]}
    metadata["resolved_model_identity_consistent"] = len(resolved) <= 1
    metadata["all_requested_rounds_complete"] = all(a[0]["completed_rounds"] == rounds for a in arms.values())
    metadata["all_within_resource_ceilings"] = all(a[0]["budget"]["within_resource_ceilings"] for a in arms.values())
    if probe_rounds:
        continuations = {}
        source = arms["rsi"][1]
        calls = len(development)+(len(evolution)+len(development))*probe_rounds+len(evolution)+len(development)
        for name in ("initial_policy", "evolved_policy"):
            start = copy.deepcopy(source)
            if name == "initial_policy":
                start.improvement_workflow = INITIAL_WORKFLOW
            continuations[name] = run_arm(output/"continuation"/name, start, evolution, development,
                                         probe_rounds, False, Budget(calls, wall_seconds, max_cpu_seconds=cpu_seconds),
                                         backend_factory(model_config), probe_rounds)
        metadata["continuation"] = {name: item[0] for name, item in continuations.items()}
        write_tables(output/"continuation", continuations)
        gains = {name: item[0]["initial_validation"]["macro_relative_l2"]-
                      item[0]["selected_validation"]["macro_relative_l2"]
                 for name, item in continuations.items()}
        metadata["continuation_policy_advantage"] = gains["evolved_policy"]-gains["initial_policy"]
        metadata["continuation_matched_actual_calls"] = len({item[0]["budget"]["solver_calls"]
                                                              for item in continuations.values()}) == 1
    write_json(output/"manifest.json", metadata)
    return metadata
