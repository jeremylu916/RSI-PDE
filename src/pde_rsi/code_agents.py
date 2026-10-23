"""Code-writing single-run and recursive improvement agents on public PDE tasks.

The frozen model writes complete Python solver modules, one per PDE family. The
harness checks each module against a static policy, executes it in a sandbox,
evaluates it with the unchanged independent evaluator, and applies the same
acceptance gate used by the settings agents. The single run is one model request
whose programs are frozen as returned; the RSI arm iterates with the gate.
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import copy
import dataclasses
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
import platform
import time

from .agent_runs import METRICS, metric_summary, public_tasks
from .agent_runs import released_diagnostics as released_1d
from .codegen import (
    ALLOWED_MODULES, MAX_CODE_CHARS, MAX_OUTPUT_VALUES, SANDBOX_VERSION, agent_problem_record,
    STARTS, check_program, code_sha256, program_digest, run_program_task, sandbox_self_test, classical_program,
    starting_program,
)
from .evaluator import EVALUATOR_VERSION, acceptance_gate, failed_result
from .experiment import append_event, write_json
from .model_backend import ModelBackend, ModelCallError
from .runner import Budget, BudgetExceeded
from .storage import project_path
from .tasks import ALL_FAMILIES, FAMILIES, GROUP_FAMILIES, WAVE_FAMILIES, load_group_tasks, load_wave_tasks

PROTOCOL_VERSION = "code-agents-v2-self-test"
BROKEN_RELATIVE_L2 = 1.0
INITIAL_WORKFLOW = (
    "Inspect errors by family and instance. Identify the dominant error source: spatial or temporal "
    "truncation, initial-time singularity, transport, nonlinear coupling, interpolation onto evaluator "
    "nodes, or execution failure. Rewrite the solver programs of the families where a targeted numerical "
    "change should help, preserving exact IC/BC and resource limits. Use only the supplied task and "
    "diagnostic data."
)
SOLVER_CONTRACT_1D = f"""Each PDE family has one standalone Python module defining solve(problem). It returns
{{"x": x, "t": t, "u": u}} (or a tuple x, t, u):
- x: strictly increasing UNIFORM grid with x[0]=0 and x[-1]=problem.length, at least 3 points.
- t: Poisson returns t=[0.0]. Time-dependent families return strictly increasing t with t[0]=0 and
  t[-1]=problem.final_time, at least 3 points; nonuniform time grids are allowed.
- u: finite real array of shape (len(t), len(x)). u[0] must equal problem.initial(x) and
  u[:, 0] = u[:, -1] = 0 to within 1e-10 (Poisson: boundaries only).
problem attributes: family, equation, length, final_time, diffusivity, velocity, orders, weights,
left_boundary, right_boundary (both 0.0), relative_l2_tolerance, boundary_tolerance, initial_tolerance,
max_seconds, memory_mb, output_space_points, output_time_points.
problem.forcing(x, t) returns the source term at a NumPy array x and scalar time t. Time-dependent
families require t > 0, and fractional sources may be singular as t -> 0. Poisson ignores t.
problem.initial(x) returns u(x, 0). Forcing and initial data are black-box functions.
Caputo(alpha, u)(t) = 1/Gamma(1-alpha) * integral_0^t (t-s)^(-alpha) u_s(s) ds, history from t=0.
The evaluator linearly interpolates the returned solution onto output_space_points uniform x nodes
and output_time_points uniform t nodes (Poisson: x nodes only), then integrates squared error with
the trapezoidal rule. A returned grid containing those nodes avoids interpolation error.
Allowed imports: {", ".join(sorted(ALLOWED_MODULES))}.
Forbidden: file, network, or process access; eval, exec, compile, open, getattr, setattr, type, dir;
any name or attribute beginning with "_"; frame or code introspection. Programs run single-threaded
without a GPU in an isolated sandbox. Each solve is limited to max_seconds of wall and CPU time
(including about 0.5 s interpreter start-up), memory_mb of address space, and {MAX_OUTPUT_VALUES:,}
returned values. Each module is at most {MAX_CODE_CHARS:,} characters.
Write genuine numerical discretizations of the stated equation. Never infer, fit, or hard-code
closed-form solutions, and never branch on individual task identities or coefficients to special-case
a benchmark instance."""
SOLVER_CONTRACT_2D = """Two-dimensional families (spatial_dimension = 2, currently wave2d) differ as follows.
The domain is [0, length] x [0, length] with u = 0 on all four edges. solve(problem) returns
{"x": x, "y": y, "t": t, "u": u} (or a tuple x, y, t, u) where x and y are uniform grids from 0 to length
and u has shape (len(t), len(y), len(x)), i.e. u[n, j, i] = u(x[i], y[j], t[n]).
problem.forcing(x, y, t) and problem.initial(x, y) accept broadcastable arrays such as
numpy.meshgrid(x, y) outputs; problem.initial_velocity(x, y) returns u_t(x, y, 0).
wave2d solves u_tt - diffusivity*(u_xx + u_yy) = f with u(x,y,0) = initial and u_t(x,y,0) = initial_velocity;
wave_speed = sqrt(diffusivity). The source may be singular as t -> 0 but is integrable.
The evaluator linearly interpolates onto output_space_points uniform nodes in each of x and y and
output_time_points uniform times, so return only the time levels you need: the full array must stay
within the returned-value limit."""
SOLVER_CONTRACT_GROUP = """Group fractional families (fode, fpde_caputo, fpde_variable_order, fpde_variable_order_xt):
- All start from rest: u = 0 at t = 0, and t in [0, final_time].
- fode is a scalar problem (spatial_dimension = 0): problem.forcing(t) returns f(t), problem.initial()
  returns u(0) = 0, and solve(problem) returns {"t": t, "u": u} (or a tuple t, u) with u of shape (len(t),).
  The evaluator compares u at output_time_points uniform times.
- The fpde families live on x in [0, length] with u(0,t) = u(length,t) = 0. problem.forcing(x, t) accepts an
  array x and scalar t >= 0, problem.initial(x) returns the initial data, and solve(problem) returns
  {"x", "t", "u"} with u[t, x] as for the 1D families.
- The Caputo order: problem.alpha(t) returns orders[0] (constant order) or orders[0]*sin(t)**2
  (fpde_variable_order); for fpde_variable_order_xt, problem.alpha(x, t) returns orders[0]*sin(x)*t**2 for an
  array x. A variable order is evaluated at the current point:
  Caputo(alpha, u)(x, t) = 1/Gamma(1-alpha(x,t)) * integral_0^t (t-s)^(-alpha(x,t)) u_s(x, s) ds."""
SOLVER_CONTRACT = SOLVER_CONTRACT_1D


def solver_contract(families):
    contract = SOLVER_CONTRACT_1D
    if any(family in WAVE_FAMILIES for family in families):
        contract += "\n"+SOLVER_CONTRACT_2D
    if any(family in GROUP_FAMILIES for family in families):
        contract += "\n"+SOLVER_CONTRACT_GROUP
    return contract


def released_diagnostics(results):
    rows = released_1d(results)
    for row, result in zip(rows, results):
        if result.get("ny") is not None:
            row["ny"] = result["ny"]
    return rows


@dataclass
class CodeVersion:
    model: str
    program: dict = field(default_factory=classical_program)
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

    def summary(self):
        """Snapshot with program sources replaced by hashes, for model context."""
        agent = self.to_dict()
        agent["program"] = program_digest(self.program)
        return {"version_id": self.version_id, "agent": agent}


def proposal_schema(families=FAMILIES, classes=None, experiment_cases=None):
    """classes: the library classes of a modular (harness v2) protocol; parts may then be replaced or patched.
    experiment_cases: the training case count when the experiment tool is enabled; adds an optional experiment key."""
    schema = _proposal_schema(families, classes)
    if experiment_cases:
        schema["properties"]["experiment"] = {
            "type": "object", "additionalProperties": False,
            "properties": {"cases": {"type": "array", "items": {"type": "string"}, "maxItems": experiment_cases},
                           "question": {"type": "string", "minLength": 1, "maxLength": 600}},
            "required": ["cases", "question"]}
    return schema


def _proposal_schema(families, classes):
    if classes is not None:
        from .modular import part_names, update_schema
        update, families = update_schema(classes), part_names(classes)
        return {"type": "object", "additionalProperties": False,
                "properties": {"hypothesis": {"type": "string", "minLength": 1, "maxLength": 3000},
                               "program_updates": {"type": "array", "items": update, "minItems": 1,
                                                   "maxItems": len(families)},
                               "next_improvement_policy": {"type": ["string", "null"], "minLength": 1, "maxLength": 2048},
                               "memory_update": {"type": "string", "maxLength": 1200}},
                "required": ["hypothesis", "program_updates", "next_improvement_policy", "memory_update"]}
    update = {"type": "object", "additionalProperties": False,
              "properties": {"family": {"type": "string", "enum": list(families)},
                             "change_summary": {"type": "string", "minLength": 1, "maxLength": 600},
                             "code": {"type": "string", "minLength": 1, "maxLength": MAX_CODE_CHARS}},
              "required": ["family", "change_summary", "code"]}
    return {"type": "object", "additionalProperties": False,
            "properties": {"hypothesis": {"type": "string", "minLength": 1, "maxLength": 3000},
                           "program_updates": {"type": "array", "items": update, "minItems": 1,
                                               "maxItems": len(families)},
                           "next_improvement_policy": {"type": ["string", "null"], "minLength": 1, "maxLength": 2048},
                           "memory_update": {"type": "string", "maxLength": 1200}},
            "required": ["hypothesis", "program_updates", "next_improvement_policy", "memory_update"]}


def apply_proposal(version, record, recursive, classes=None):
    """Validate a proposal and return (candidate, changed families) without executing code. With classes (a
    modular protocol), program_updates replace or patch parts (modular.apply_updates)."""
    if set(record) != {"hypothesis", "program_updates", "next_improvement_policy", "memory_update"}:
        raise ValueError("Proposal requires exactly hypothesis, program_updates, next_improvement_policy, memory_update")
    if not isinstance(record["hypothesis"], str) or not 1 <= len(record["hypothesis"]) <= 3000:
        raise ValueError("Invalid proposal hypothesis")
    if not isinstance(record["memory_update"], str) or len(record["memory_update"]) > 1200:
        raise ValueError("Invalid bounded memory update")
    policy = record["next_improvement_policy"]
    if policy is not None and (not isinstance(policy, str) or not 1 <= len(policy) <= 2048):
        raise ValueError("Improvement workflow must be null or a bounded instruction string")
    if not recursive and policy is not None:
        raise ValueError("A non-recursive arm cannot update its improvement policy")
    updates = record["program_updates"]
    if classes is not None:
        from .modular import apply_updates
        program, changed = apply_updates(version.program, updates, classes)
        updates = []
    families = list(version.program)
    if classes is None and (not isinstance(updates, list) or not 1 <= len(updates) <= len(families)):
        raise ValueError(f"program_updates must list one to {len(families)} family programs")
    if classes is None:
        program = dict(version.program)
        changed = []
    for update in updates:
        if not isinstance(update, dict) or set(update) != {"family", "change_summary", "code"}:
            raise ValueError("Each program update requires exactly family, change_summary, code")
        family = update["family"]
        if family not in families or family in changed:
            raise ValueError("Program update families must be unique benchmark families")
        if not isinstance(update["change_summary"], str) or not 1 <= len(update["change_summary"]) <= 600:
            raise ValueError(f"Invalid change_summary for {family}")
        check_program(update["code"], family)
        program[family] = update["code"]
        changed.append(family)
    if program == version.program:
        raise ValueError("Proposal leaves every solver program unchanged")
    candidate = copy.deepcopy(version)
    candidate.program = program
    if recursive:
        candidate.parent_id = version.version_id
        candidate.generation += 1
        candidate.improvement_workflow = policy if policy is not None else version.improvement_workflow
        candidate.memory.append({"generation": candidate.generation, "hypothesis": record["hypothesis"],
                                 "changed_families": changed, "lesson": record["memory_update"]})
    return candidate, changed


def evaluate_suite(tasks, version, budget, solution_directory=None):
    results = []
    if solution_directory is not None:
        project_path(solution_directory).mkdir(parents=True, exist_ok=True)
    for index, task in enumerate(tasks):
        try:
            path = solution_directory/f"{task.task_id}.npz" if solution_directory is not None else None
            results.append(run_program_task(task, version.program[task.family], budget, solution_path=path))
        except BudgetExceeded:
            results.extend(failed_result(t, "budget_exhausted", "Task was not executed; experiment allowance exhausted", 0)
                           for t in tasks[index:])
            break
    return results


def write_program(directory, program):
    directory = project_path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for family, code in program.items():
        (directory/f"{family}.py").write_text(code)


def _updates_sentence(modular):
    if modular:
        from .modular import SYSTEM_PROGRAM_SENTENCE
        return SYSTEM_PROGRAM_SENTENCE
    return ("Each program_updates item gives a family, a change_summary, and code: the COMPLETE replacement module "
            "for that family as a JSON string with escaped newlines and quotes. Families you omit keep their active program. ")


def option_sentences(best_of=1, gate="standard", experiments_per_round=0):
    """Prompt sentences for the opt-in RSI options; empty for the defaults."""
    sentences = ""
    if gate == "pass-first":
        sentences += (" Gate variant (pass-first): a candidate that makes more validation cases pass and makes no "
                      "passing case fail is accepted even if macro relative L2 drops by less than 0.5% or a family "
                      "mean rises, provided every case executes within IC/BC and resource limits and no case that "
                      "passes in both versions has its combined or final-time relative L2 rise more than 25%.")
    if best_of > 1:
        sentences += (f" Best-of-{best_of}: each attempt draws {best_of} independent proposals from this same "
                      "context. The harness self-tests all of them and submits the one with the lowest training "
                      "macro relative L2; later context shows only the selected proposal.")
    if experiments_per_round:
        sentences += (" Experiment tool: a proposal may add the key experiment = {\"cases\": [...], \"question\": "
                      "\"...\"}. The harness then applies program_updates to the ACTIVE program, runs that candidate "
                      "on the listed TRAINING cases (ids as in evolution_diagnostics; an empty list means all "
                      "training cases), and returns its diagnostics next to the active program's on the same cases. "
                      "An experiment is not submitted to the validation gate and uses no attempt. "
                      f"Up to {experiments_per_round} experiment(s) per round; validation cases cannot be "
                      "run. Without an experiment key, or once the round's experiments are used, the proposal is "
                      "self-tested and submitted as usual.")
    return sentences


def system_prompt(recursive, modular=False, options=None):
    """options: keyword arguments of option_sentences (opt-in RSI options); None keeps the default prompt."""
    return (
        "You are a numerical PDE/FPDE agent that writes Python solver programs. "
        "Return only one JSON object with exactly the keys hypothesis, program_updates, "
        "next_improvement_policy, and memory_update, matching the supplied schema, without prose or Markdown. "
        + _updates_sentence(modular) +
        ("You cannot execute code or tools; the harness runs your programs and returns diagnostics. "
         if not (options or {}).get("experiments_per_round") else
         "You cannot execute code yourself; the harness runs your programs and returns diagnostics, including "
         "the experiments described below. ") +
        "Use only these public task specifications, the solver contract, and released numerical diagnostics. "
        "Never request references, sealed cases, external search, or evaluator changes. "
        "Minimize macro relative space-time L2 error (the mean of family means) while every case passes. "
        "The candidate is gated as a whole on the validation problems: it is rejected if any case fails to "
        "execute or violates IC/BC or resource limits, if macro relative L2 does not drop by more than 0.5%, "
        "if any family mean rises more than 5% or its pass rate drops, or if any case's combined or final-time "
        "relative L2 rises more than 25%. Prefer correct, robust code over risky rewrites of many families at once. "
        "Each round allows several attempts. The harness first runs each candidate on the training (evolution) "
        "problems. If any training case fails to execute, violates IC/BC or resource limits, or has relative L2 >= 1, "
        "you receive the training diagnostics and may revise within the same round; every revision is a new request. "
        "The first candidate that passes this self-test is submitted to the validation gate. "
        "Use the supplied active_improvement_policy as the procedure for the current round, and use inherited "
        "memory to interpret diagnostics. "
        + ("You may also propose a bounded next improvement-policy instruction and reusable memory lesson. "
           "They are inherited only if the candidate is accepted, and then drive later proposals."
           if recursive else
           "The improvement controller remains fixed throughout this session. next_improvement_policy MUST be null. "
           "Your program revisions and prior feedback may be used within this session; no new persistent policy "
           "or memory is inherited.")
        + option_sentences(**(options or {}))
    )


# v1 (2026-10-20 morning): the program-length limit was only in the response schema.
# v2: the limit is stated in the prompt (all three Opus inventory answers in v1 exceeded it).
ONE_SHOT_PROMPT_VERSION = "v2-stated-code-limit"


def one_shot_system_prompt(modular=False):
    return (
        "You are a numerical PDE/FPDE agent that writes Python solver programs. "
        "Return only one JSON object with exactly the keys hypothesis, program_updates, "
        "next_improvement_policy, and memory_update, matching the supplied schema, without prose or Markdown. "
        + _updates_sentence(modular) +
        "You cannot execute code or tools. "
        "Use only these public task specifications, the solver contract, and released numerical diagnostics. "
        "Never request references, sealed cases, external search, or evaluator changes. "
        "Minimize macro relative space-time L2 error (the mean of family means) while every case passes. "
        "This is a single run: you get exactly one request and no feedback. The programs you return are frozen "
        "and scored as they are, with no self-test, revision, or validation gate, so return your best complete, "
        "robust solver now. Use the supplied active_improvement_policy as the procedure. "
        f"Each program is rejected without being run if its code exceeds {MAX_CODE_CHARS:,} characters, so keep "
        "every module well under that limit. "
        "next_improvement_policy MUST be null; memory_update is not used."
    )


def compact_diagnostics(rows):
    """Short per-case rows for older rounds; failures keep a truncated message."""
    def identifier(row):  # identification diagnostics use anonymized case IDs
        return {"case_id": row["case_id"]} if "case_id" in row else {"task_id": row["task_id"]}
    return [{**identifier(row), "status": row["status"], "relative_l2": row["relative_l2"],
             "passed": row["passed"], **({"message": (row.get("message") or "")[:300]}
                                         if row["status"] != "ok" else {})} for row in rows]


def build_messages(transcript):
    """Compact earlier programs and per-case diagnostics; the latest round stays complete."""
    contexts = [i for i, (role, value) in enumerate(transcript)
                if role == "user" and isinstance(value, dict) and "active_program" in value]
    proposals = [i for i, (role, value) in enumerate(transcript) if role == "assistant" and isinstance(value, dict)]
    decisions = [i for i, (role, value) in enumerate(transcript)
                 if role == "user" and isinstance(value, dict) and "candidate_decision" in value]
    tests = [i for i, (role, value) in enumerate(transcript)
             if role == "user" and isinstance(value, dict) and "self_test" in value]
    experiments = [i for i, (role, value) in enumerate(transcript)
                   if role == "user" and isinstance(value, dict) and "experiment_result" in value]
    messages = []
    for index, (role, value) in enumerate(transcript):
        if role == "system":
            messages.append({"role": role, "content": value})
            continue
        if role == "user" and index in contexts[:-1]:
            value = {**value, "active_program": "omitted; see the latest context",
                     "active_program_digest": program_digest(value["active_program"]),
                     "evolution_diagnostics": compact_diagnostics(value["evolution_diagnostics"]),
                     "validation_diagnostics": compact_diagnostics(value["validation_diagnostics"])}
        if role == "user" and index in tests[:-1]:
            value = {**value, "self_test": {**value["self_test"], "train_diagnostics":
                                            compact_diagnostics(value["self_test"]["train_diagnostics"])}}
        if role == "user" and index in experiments[:-1]:
            result = value["experiment_result"]
            value = {**value, "experiment_result": {
                **result, **{key: compact_diagnostics(result[key])
                             for key in ("candidate_diagnostics", "active_diagnostics") if key in result}}}
        if role == "user" and index in decisions[:-1]:
            decision = value["candidate_decision"]
            value = {"candidate_decision": {**decision, "candidate_validation_diagnostics":
                                            compact_diagnostics(decision["candidate_validation_diagnostics"])}}
        if role == "assistant" and index in proposals[:-1]:
            from .modular import compact_update
            value = {**value, "program_updates": [compact_update(u, code_sha256) for u in value["program_updates"]]}
        messages.append({"role": role, "content": value if isinstance(value, str) else json.dumps(value, allow_nan=False)})
    return messages


def self_test_failures(results):
    """Training cases that show a candidate is broken rather than merely inaccurate."""
    return [row["task_id"] for row in results
            if row["status"] != "ok" or not row["constraints_pass"] or not row["resources_pass"]
            or row["relative_l2"] >= BROKEN_RELATIVE_L2]


class CodeProtocol:
    """Task-specific parts of the improvement loop. Methods look up module functions at call time."""
    name = "code"

    def problem_records(self, tasks):
        return [agent_problem_record(task) for task in tasks]

    def contract(self, families):
        return solver_contract(families)

    def evaluate(self, tasks, version, budget, solution_directory=None):
        return evaluate_suite(tasks, version, budget, solution_directory)

    def diagnostics(self, results):
        return released_diagnostics(results)

    def reasons(self, reasons, tasks):
        return reasons

    def load_tasks(self, root, package, families):
        return load_public_tasks(root, package, families)

    def self_test_tasks(self, evolution):
        return evolution

    def starting_program(self, start, families):
        return starting_program(start, families)

    def metadata(self, tasks):
        return {}


def history_window(transcript, round_starts, history_rounds):
    """The system and task messages plus the last history_rounds rounds (all rounds when None)."""
    if not history_rounds or len(round_starts) <= history_rounds:
        return transcript
    return transcript[:2]+transcript[round_starts[-history_rounds]:]


GATES = ("standard", "pass-first")
RSI_OPTION_DEFAULTS = {"best_of": 1, "gate": "standard", "experiments_per_round": 0}


def rsi_options(best_of=1, gate="standard", experiments_per_round=0):
    """Validated opt-in RSI options that differ from the defaults ({} for a default run)."""
    if not isinstance(best_of, int) or best_of < 1:
        raise ValueError("best_of must be a positive integer")
    if gate not in GATES:
        raise ValueError(f"gate must be one of {GATES}")
    if not isinstance(experiments_per_round, int) or experiments_per_round < 0:
        raise ValueError("experiments_per_round must be a nonnegative integer")
    if best_of > 1 and experiments_per_round:
        raise ValueError("best_of and experiments_per_round cannot be combined")
    options = {"best_of": best_of, "gate": gate, "experiments_per_round": experiments_per_round}
    return {key: value for key, value in options.items() if value != RSI_OPTION_DEFAULTS[key]}


def case_key(task):
    """The case identifier the model sees in diagnostics: case_id where the protocol anonymizes, else task_id."""
    return getattr(task, "case_id", None) or task.task_id


def check_experiment(experiment):
    if not (isinstance(experiment, dict) and set(experiment) == {"cases", "question"}
            and isinstance(experiment["cases"], list) and all(isinstance(c, str) for c in experiment["cases"])
            and isinstance(experiment["question"], str) and 1 <= len(experiment["question"]) <= 600):
        raise ValueError("experiment must be an object with exactly cases (a list of training case ids) and a "
                         "question (1-600 characters)")


def run_arm(directory, starting, evolution, development, rounds, recursive, budget, backend,
            max_model_calls, max_input_chars=500000, model_timeout=900, attempts_per_round=3, protocol=None,
            deadline=None, history_rounds=None, max_validation_submissions=None, spend_cap_usd=None,
            best_of=1, gate="standard", experiments_per_round=0):
    """One arm. With a deadline (time.time() value), rounds continue until it passes: no new round or model
    request starts after it, and a request already running may finish its round. Frozen evaluation follows.
    max_validation_submissions caps validation-gate decisions; spend_cap_usd stops new model requests once the
    priced cost of the arm's responses reaches it (configs/model_prices.json).

    Opt-in options (defaults reproduce the earlier harness exactly):
    best_of: each attempt draws this many independent proposals in parallel; all are self-tested and the passing
      one with the lowest training macro relative L2 is submitted.
    gate: "pass-first" accepts a candidate that passes more validation cases and fails none that passed
      (evaluator.acceptance_gate pass_first).
    experiments_per_round: requests per round in which the model may run a candidate on chosen training cases
      before submitting; experiments use no attempt and never touch validation cases."""
    from .costs import load_prices, response_cost
    options = rsi_options(best_of, gate, experiments_per_round)
    gate_options = {"pass_first": True} if gate == "pass-first" else {}
    protocol = protocol or CodeProtocol()
    prices = load_prices() if spend_cap_usd is not None else None
    if prices is not None and (prices.get(starting.model) or {}).get("output") is None:
        raise ValueError(f"No price for {starting.model} in configs/model_prices.json; cannot enforce a spending cap")
    spent = 0.0
    directory = project_path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    version = copy.deepcopy(starting)
    budget.ledger_path = directory/"solve_calls.jsonl"
    budget.phase = "initial_validation"
    feedback = protocol.evaluate(development, version, budget)
    write_json(directory/"initial_validation.json", feedback)
    write_json(directory/"initial_agent.json", version.snapshot())
    write_program(directory/"programs/initial", version.program)
    initial = metric_summary(feedback)
    all_tasks = evolution+development
    families = list(version.program)
    classes = protocol.classes() if getattr(protocol, "modular", False) else None
    schema = proposal_schema(families, classes, len(evolution) if experiments_per_round else None)
    transcript = [("system", system_prompt(recursive, classes is not None, options or None)),
                  ("user", {"public_problems": protocol.problem_records(all_tasks),
                            "solver_contract": protocol.contract(families), "proposal_schema": schema})]
    accepted = completed = requests = 0
    model_seconds = 0.0
    usage_records, resolved_models = [], []
    stop = False
    round_starts = []
    experiment_cases = {case_key(task): task for task in evolution}

    def precheck(iteration, attempt):
        """Messages for the next request, or "break" (model calls used) / "stop" (another limit reached)."""
        if requests >= max_model_calls:
            return "break"
        if deadline is not None and time.time() >= deadline:
            append_event(directory/"events.jsonl", {"iteration": iteration, "attempt": attempt,
                         "event": "time_budget_reached"})
            return "stop"
        if spend_cap_usd is not None and spent >= spend_cap_usd:
            append_event(directory/"events.jsonl", {"iteration": iteration, "attempt": attempt,
                         "event": "spending_cap_reached", "spent_usd": spent})
            return "stop"
        messages = build_messages(history_window(transcript, round_starts, history_rounds))
        if sum(len(message["content"]) for message in messages) > max_input_chars:
            append_event(directory/"events.jsonl", {"iteration": iteration, "attempt": attempt,
                         "event": "input_character_budget_exhausted"})
            return "stop"
        if budget.remaining_seconds <= 0 or budget.remaining_cpu_seconds <= 0:
            return "stop"
        return messages

    def append_proposal(proposal):
        if isinstance(proposal.get("program_updates"), list) and all(
                isinstance(u, dict) and (isinstance(u.get("code"), str) or isinstance(u.get("edits"), list))
                and isinstance(u.get("change_summary"), str)
                and isinstance(u.get("family"), str) for u in proposal["program_updates"]):
            transcript.append(("assistant", proposal))
        else:
            transcript.append(("assistant", json.dumps(proposal, allow_nan=False)))

    def record_response(result, iteration, attempt, sample=None):
        """Count a model response; False (and stop) when the resolved model identity changed."""
        nonlocal requests, model_seconds, stop
        requests += 1
        model_seconds += result["wall_seconds"]
        usage_records.append(result["usage"])
        resolved = result["resolved_model"]
        if resolved:
            resolved_models.append(resolved)
            if len(set(resolved_models)) != 1:
                append_event(directory/"events.jsonl", {"iteration": iteration, "attempt": attempt,
                             **({} if sample is None else {"sample": sample}),
                             "event": "model_identity_changed", "resolved_models": resolved_models})
                stop = True
                return False
        return True

    def record_failure(error, iteration, attempt, sample=None):
        nonlocal requests, model_seconds
        requests += int(error.call_attempted)
        model_seconds += error.wall_seconds
        usage_records.append(error.usage)
        append_event(directory/"events.jsonl", {"iteration": iteration, "attempt": attempt,
                     **({} if sample is None else {"sample": sample}),
                     "event": "model_failure", "message": str(error), "usage": error.usage,
                     "request_attempted": error.call_attempted})

    def charge(call_directory):
        nonlocal spent
        if prices is not None:
            spent += response_cost(call_directory/"response.json", starting.model, prices) or 0.0

    def propose(iteration, attempt, tag, remaining, messages):
        """One request applied to the active program: a dict, or None after a rejected response."""
        try:
            try:
                result = backend.complete(messages, schema, directory/f"model-call-{tag}",
                                          timeout=min(model_timeout, budget.remaining_seconds))
            finally:  # failed calls are billed too
                charge(directory/f"model-call-{tag}")
            if not record_response(result, iteration, attempt):
                return None
            proposal = result["proposal"]
            write_json(directory/f"proposal-{tag}.json", {"proposal": proposal})
            append_proposal(proposal)
            experiment = None
            if experiments_per_round and isinstance(proposal, dict) and "experiment" in proposal:
                experiment = proposal["experiment"]
                proposal = {key: value for key, value in proposal.items() if key != "experiment"}
                check_experiment(experiment)
            candidate, changed = apply_proposal(version, proposal, recursive, classes)
            write_json(directory/f"proposal-{tag}.json",
                       {"proposal": result["proposal"], "changed_families": changed, **candidate.snapshot()})
            write_program(directory/f"programs/candidate-{tag}", candidate.program)
        except ModelCallError as error:
            record_failure(error, iteration, attempt)
            transcript.append(("user", "Response rejected: "+str(error)[:1000]+
                               f". Attempts remaining this round: {remaining}. "
                               "Return exactly one complete JSON object matching the schema."))
            return None
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            append_event(directory/"events.jsonl", {"iteration": iteration, "attempt": attempt,
                         "event": "invalid_proposal", "message": str(error)[:1000]})
            transcript.append(("user", "Proposal rejected before execution: "+str(error)[:1000]+
                               f". The active programs are unchanged. Attempts remaining this round: {remaining}."))
            return None
        return {"candidate": candidate, "changed": changed, "result": result, "resolved": result["resolved_model"],
                "experiment": experiment}

    def self_test(candidate, tag):
        budget.phase = f"self_test_{tag}"
        test = protocol.evaluate(protocol.self_test_tasks(evolution), candidate, budget)
        write_json(directory/f"self-test-{tag}.json", test)
        return test, self_test_failures(test), metric_summary(test)["macro_relative_l2"]

    def self_test_feedback(iteration, attempt, broken, test_macro, active_train_macro, test, remaining, extra=None):
        transcript.append(("user", {"self_test": {
            "iteration": iteration, "attempt": attempt, "submitted": False, "broken_cases": broken,
            "candidate_train_macro_relative_l2": test_macro, "active_train_macro_relative_l2": active_train_macro,
            "train_diagnostics": protocol.diagnostics(test), "attempts_remaining": remaining, **(extra or {})},
            "instruction": ("The candidate was not submitted because training cases failed to execute, violated "
                            "IC/BC or resource limits, or had relative L2 >= 1. "
                            + ("Return a revised complete proposal. program_updates apply to the ACTIVE program, "
                               "so include every family you want changed." if remaining else
                               "No attempts remain this round; the active programs are unchanged."))}))

    def run_experiment(iteration, attempt, tag, number, outcome, evolution_feedback, left):
        request = outcome["experiment"]
        unknown = [case for case in request["cases"] if case not in experiment_cases]
        result = {"iteration": iteration, "attempt": attempt, "experiment": number, "question": request["question"]}
        if unknown:
            result.update(error="Unknown training case ids: "+", ".join(unknown[:20]), cases=request["cases"])
        else:
            chosen = list(dict.fromkeys(request["cases"])) or list(experiment_cases)
            tasks = [experiment_cases[case] for case in chosen]
            budget.phase = f"experiment_{tag}"
            rows = protocol.evaluate(tasks, outcome["candidate"], budget)
            write_json(directory/f"experiment-{tag}.json", rows)
            ids = {task.task_id for task in tasks}
            active_rows = [row for row in evolution_feedback if row["task_id"] in ids]
            result.update(cases=chosen, candidate_macro_relative_l2=metric_summary(rows)["macro_relative_l2"],
                          active_macro_relative_l2=metric_summary(active_rows)["macro_relative_l2"],
                          candidate_diagnostics=protocol.diagnostics(rows),
                          active_diagnostics=protocol.diagnostics(active_rows))
        result["experiments_remaining_this_round"] = left
        append_event(directory/"events.jsonl", {
            "event": "experiment", **{key: value for key, value in result.items()
                                      if key not in ("candidate_diagnostics", "active_diagnostics")},
            "changed_families": outcome["changed"], "model_usage": outcome["result"]["usage"]})
        transcript.append(("user", {"experiment_result": result, "instruction": (
            "This experiment was not submitted and used no attempt. Return your next proposal. program_updates "
            "apply to the ACTIVE program, so include every change you want; add an experiment key to run another "
            "experiment" + (" (none remain this round)." if not left else "."))}))

    def sample_best_of(iteration, attempt, tag, remaining, messages, active_train_macro):
        """Best-of-K attempt: (candidate, changed, result, resolved, test_macro, attempt) or None."""
        count = min(best_of, max_model_calls-requests)
        timeout = min(model_timeout, budget.remaining_seconds)

        def call(sample):
            note = {"best_of_sample": sample, "samples_this_attempt": count}
            try:
                return backend.complete(messages+[{"role": "user", "content": json.dumps(note)}], schema,
                                        directory/f"model-call-{tag}-s{sample}", timeout=timeout)
            except ModelCallError as error:
                return error
        with ThreadPoolExecutor(count) as pool:
            responses = list(pool.map(call, range(1, count+1)))
        samples, answered = [], []
        for sample, response in enumerate(responses, 1):  # bill and count every sample before using any
            charge(directory/f"model-call-{tag}-s{sample}")
            if isinstance(response, ModelCallError):
                record_failure(response, iteration, attempt, sample)
                samples.append({"sample": sample, "status": "model_failure", "message": str(response)[:300]})
            elif record_response(response, iteration, attempt, sample):
                answered.append((sample, response))
        if stop:
            return None
        for sample, response in answered:
            proposal = response["proposal"]
            write_json(directory/f"proposal-{tag}-s{sample}.json", {"proposal": proposal})
            try:
                candidate, changed = apply_proposal(version, proposal, recursive, classes)
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                append_event(directory/"events.jsonl", {"iteration": iteration, "attempt": attempt, "sample": sample,
                             "event": "invalid_proposal", "message": str(error)[:1000]})
                samples.append({"sample": sample, "status": "invalid_proposal", "message": str(error)[:300],
                                "proposal": proposal})
                continue
            write_json(directory/f"proposal-{tag}-s{sample}.json",
                       {"proposal": proposal, "changed_families": changed, **candidate.snapshot()})
            write_program(directory/f"programs/candidate-{tag}-s{sample}", candidate.program)
            test, broken, test_macro = self_test(candidate, f"{tag}-s{sample}")
            append_event(directory/"events.jsonl", {"iteration": iteration, "attempt": attempt, "sample": sample,
                         "event": "self_test", "passed": not broken, "broken_cases": broken,
                         "changed_families": changed, "candidate_train_macro_relative_l2": test_macro,
                         "active_train_macro_relative_l2": active_train_macro, "model_usage": response["usage"]})
            samples.append({"sample": sample, "status": "self_tested", "broken_cases": broken,
                            "candidate_train_macro_relative_l2": test_macro, "proposal": proposal,
                            "candidate": candidate, "changed": changed, "result": response, "test": test})
        samples.sort(key=lambda s: s["sample"])
        tested = [s for s in samples if s["status"] == "self_tested"]
        passing = [s for s in tested if not s["broken_cases"]]
        chosen = min(passing or tested, key=lambda s: (s["candidate_train_macro_relative_l2"], s["sample"]),
                     default=None)
        overview = [{key: s[key] for key in ("sample", "status", "message", "broken_cases",
                                             "candidate_train_macro_relative_l2") if key in s} for s in samples]
        append_event(directory/"events.jsonl", {"iteration": iteration, "attempt": attempt, "event": "best_of",
                     "samples": overview, "selected_sample": chosen["sample"] if passing else None})
        if passing:
            append_proposal(chosen["proposal"])
            return (chosen["candidate"], chosen["changed"], chosen["result"], chosen["result"]["resolved_model"],
                    chosen["candidate_train_macro_relative_l2"], attempt)
        if chosen is not None:  # every self-tested sample was broken: show the least broken one
            append_proposal(chosen["proposal"])
            self_test_feedback(iteration, attempt, chosen["broken_cases"], chosen["candidate_train_macro_relative_l2"],
                               active_train_macro, chosen["test"], remaining, {"best_of_samples": overview})
        else:
            transcript.append(("user", f"All {count} sampled responses were rejected: "
                               + "; ".join(f"sample {s['sample']}: {s['message']}" for s in samples)[:2000]
                               + f". Attempts remaining this round: {remaining}. "
                               "Return exactly one complete JSON object matching the schema."))
        return None

    for iteration in range(1, rounds+1):
        if deadline is not None and time.time() >= deadline:
            append_event(directory/"events.jsonl", {"iteration": iteration, "event": "time_budget_reached"})
            break
        if max_validation_submissions is not None and completed >= max_validation_submissions:
            append_event(directory/"events.jsonl", {"iteration": iteration, "event": "validation_budget_reached"})
            break
        if requests >= max_model_calls:
            append_event(directory/"events.jsonl", {"iteration": iteration, "event": "model_call_budget_exhausted"})
            break
        try:
            budget.reserve(len(evolution)*(1+attempts_per_round*best_of+experiments_per_round)
                           +len(development)+len(all_tasks))
        except BudgetExceeded:
            append_event(directory/"events.jsonl", {"iteration": iteration, "event": "solver_budget_exhausted"})
            break
        budget.phase = f"evolution_{iteration:03d}"
        evolution_feedback = protocol.evaluate(evolution, version, budget)
        write_json(directory/f"evolution-{iteration:03d}.json", evolution_feedback)
        active_train_macro = metric_summary(evolution_feedback)["macro_relative_l2"]
        round_starts.append(len(transcript))
        transcript.append(("user", {
            "iteration": iteration, "active_agent": version.summary(),
            "active_improvement_policy": version.improvement_workflow,
            "active_program": version.program,
            "evolution_diagnostics": protocol.diagnostics(evolution_feedback),
            "validation_diagnostics": protocol.diagnostics(feedback),
            "remaining_rounds_including_this_one": rounds-iteration+1 if deadline is None else None,
            **({} if deadline is None else {"seconds_until_improvement_deadline": max(0.0, deadline-time.time())}),
            "attempts_this_round": attempts_per_round,
            **({"experiments_this_round": experiments_per_round} if experiments_per_round else {}),
            "remaining_solver_calls": budget.max_solver_calls-budget.solver_calls-len(all_tasks),
            "remaining_wall_seconds": budget.remaining_seconds if deadline is None else max(0.0, deadline-time.time()),
            "remaining_cpu_seconds": budget.remaining_cpu_seconds}))
        submitted = None
        experiments_left = experiments_per_round
        for attempt in range(1, attempts_per_round+1):
            tag = f"{iteration:03d}-{attempt}"
            remaining = attempts_per_round-attempt
            messages = precheck(iteration, attempt)
            if isinstance(messages, str):
                stop = messages == "stop"
                break
            if best_of > 1:
                submitted = sample_best_of(iteration, attempt, tag, remaining, messages, active_train_macro)
                if stop or submitted is not None:
                    break
                continue
            outcome = propose(iteration, attempt, tag, remaining, messages)
            experiment_number = 0
            while outcome is not None and outcome["experiment"] is not None and experiments_left:
                experiments_left -= 1
                experiment_number += 1
                run_experiment(iteration, attempt, f"{tag}x{experiment_number}", experiment_number, outcome,
                               evolution_feedback, experiments_left)
                messages = precheck(iteration, attempt)
                if isinstance(messages, str):
                    stop = messages == "stop"
                    break
                outcome = propose(iteration, attempt, f"{tag}x{experiment_number}", remaining, messages)
            if stop or isinstance(messages, str):
                break
            if outcome is None:
                continue
            candidate, changed, result = outcome["candidate"], outcome["changed"], outcome["result"]
            resolved = outcome["resolved"]
            test_tag = f"{tag}x{experiment_number}" if experiment_number else tag
            test, broken, test_macro = self_test(candidate, test_tag)
            append_event(directory/"events.jsonl", {"iteration": iteration, "attempt": attempt, "event": "self_test",
                         "passed": not broken, "broken_cases": broken, "changed_families": changed,
                         "candidate_train_macro_relative_l2": test_macro,
                         "active_train_macro_relative_l2": active_train_macro, "model_usage": result["usage"],
                         **({"experiment_ignored": True} if outcome["experiment"] is not None else {})})
            if not broken:
                submitted = (candidate, changed, result, resolved, test_macro, attempt)
                break
            self_test_feedback(iteration, attempt, broken, test_macro, active_train_macro, test, remaining)
        if stop:
            break
        if submitted is None:
            append_event(directory/"events.jsonl", {"iteration": iteration, "event": "no_submission",
                         "message": "No candidate passed the training self-test this round"})
            continue
        candidate, changed, result, resolved, test_macro, attempt = submitted
        budget.phase = f"candidate_validation_{iteration:03d}"
        candidate_feedback = protocol.evaluate(development, candidate, budget)
        write_json(directory/f"candidate-validation-{iteration:03d}.json", candidate_feedback)
        merged = None
        if classes is not None:  # harness v2: class-only changes are gated per part and merged
            from .modular import part_gate
            decision, merged_program, merged_feedback = part_gate(version.program, candidate.program, changed,
                                                                  feedback, candidate_feedback, **gate_options)
            if decision["accepted"] and merged_program != candidate.program:
                merged = copy.deepcopy(candidate)
                merged.program = merged_program
        else:
            decision = acceptance_gate(feedback, candidate_feedback, **gate_options)
        completed += 1
        event = {"iteration": iteration, "attempt": attempt, "parent_id": version.version_id,
                 "candidate_id": candidate.version_id, "changed_families": changed, "gate": decision,
                 "candidate_validation": metric_summary(candidate_feedback), "model_usage": result["usage"],
                 "candidate_train_macro_relative_l2": test_macro, "resolved_model": resolved}
        if decision["accepted"]:
            version, feedback = (merged, merged_feedback) if merged is not None else (candidate, candidate_feedback)
            accepted += 1
        event["active_version_id"] = version.version_id
        append_event(directory/"events.jsonl", event)
        transcript.append(("user", {"candidate_decision": {
            "iteration": iteration, "attempt": attempt, "accepted": decision["accepted"], "reasons": protocol.reasons(decision["reasons"], development),
            "relative_gain": decision["relative_gain"], "changed_families": changed,
            "candidate_train_macro_relative_l2": test_macro, "active_train_macro_relative_l2": active_train_macro,
            "candidate_macro_relative_l2": event["candidate_validation"]["macro_relative_l2"],
            "candidate_validation_diagnostics": protocol.diagnostics(candidate_feedback),
            **({"part_decisions": {part: {**d, "reasons": protocol.reasons(d["reasons"], development)}
                                   for part, d in decision["parts"].items()},
                "accepted_parts": decision.get("accepted_parts", [])} if "parts" in decision else {}),
            **({"pass_first_waived": protocol.reasons(decision["pass_first"]["waived"], development)}
               if "pass_first" in decision else {}),
            "active_version_id": version.version_id}}))
        write_json(directory/f"checkpoint-{iteration:03d}.json", version.snapshot())
    write_json(directory/"frozen_agent.json", version.snapshot())
    write_program(directory/"programs/frozen", version.program)
    budget.phase = "frozen_public_evaluation"
    frozen = protocol.evaluate(all_tasks, version, budget, solution_directory=directory/"solutions")
    write_json(directory/"frozen_results.json", frozen)
    report = {"model": starting.model, "resolved_models": sorted(set(resolved_models)),
              "recursive_policy_updates_enabled": recursive, "accepted_candidates": accepted,
              "accepted_generation_updates": version.generation-starting.generation,
              "requested_rounds": rounds, "completed_rounds": completed, "attempts_per_round": attempts_per_round,
              "initial_validation": initial, "selected_validation": metric_summary(feedback),
              "frozen_public": metric_summary(frozen),
              "frozen_by_split": {split: metric_summary([row for row in frozen if row["split"] == split])
                                  for split in ("evolution", "development")},
              "frozen_version_id": version.version_id, "frozen_program": program_digest(version.program),
              "changed_from_start": sorted(f for f in version.program if version.program[f] != starting.program.get(f)),
              "budget": {**budget.report(), "max_model_calls": max_model_calls, "model_calls": requests,
                         "max_input_characters_per_request": max_input_chars,
                         "max_output_tokens_per_request": backend.config.max_output_tokens,
                         "model_cost_usd": spent if prices is not None else None, "spend_cap_usd": spend_cap_usd,
                         "max_validation_submissions": max_validation_submissions,
                         "model_wall_seconds": model_seconds, "model_usage": usage_records},
              "complete_frozen_evaluation": all(row["status"] == "ok" for row in frozen),
              **({"rsi_options": options} if options else {})}
    write_json(directory/"report.json", report)
    return report, version, frozen


RETRYABLE_FAILURES = ("transport_error", "HTTP 429", "HTTP 5")


def one_shot_transcript(starting, evolution, development, protocol, evolution_feedback, validation_feedback):
    """The single run's messages: what RSI sees in its first round. Returns (transcript, schema, classes)."""
    families = list(starting.program)
    classes = protocol.classes() if getattr(protocol, "modular", False) else None
    schema = proposal_schema(families, classes)
    transcript = [("system", one_shot_system_prompt(classes is not None)),
                  ("user", {"public_problems": protocol.problem_records(evolution+development),
                            "solver_contract": protocol.contract(families), "proposal_schema": schema}),
                  ("user", {"iteration": 1, "active_agent": starting.summary(),
                            "active_improvement_policy": starting.improvement_workflow,
                            "active_program": starting.program,
                            "evolution_diagnostics": protocol.diagnostics(evolution_feedback),
                            "validation_diagnostics": protocol.diagnostics(validation_feedback),
                            "single_run": True, "attempts_this_round": 1})]
    return transcript, schema, classes


def repair_one_shot(strict_directory, directory, starting, evolution, development, backend, protocol=None,
                    model_timeout=900, max_requests=3):
    """Single run with one rule repair. When the single run's one answer was rejected before execution (static
    policy, schema, program length) or cut off at the output cap, send that one rejection message back and accept
    one corrected answer. No solver has run, so this carries no information about results; it only lets the
    model meet the code rules that RSI and the native agents can meet by retrying. The strict single run in
    strict_directory is left unchanged. Writes the same files as request_one_shot."""
    protocol = protocol or CodeProtocol()
    strict_directory, directory = project_path(strict_directory), project_path(directory)
    strict = json.loads((strict_directory/"one_shot.json").read_text())
    if strict["status"] == "ok":
        raise ValueError("the strict single run's answer was valid; nothing to repair")
    directory.mkdir(parents=True, exist_ok=False)
    evolution_feedback = json.loads((strict_directory/"evolution-001.json").read_text())
    validation_feedback = json.loads((strict_directory/"initial_validation.json").read_text())
    for name in ("evolution-001.json", "initial_validation.json"):
        (directory/name).write_text((strict_directory/name).read_text())
    write_json(directory/"initial_agent.json", starting.snapshot())
    write_program(directory/"programs/initial", starting.program)
    transcript, schema, classes = one_shot_transcript(starting, evolution, development, protocol,
                                                      evolution_feedback, validation_feedback)
    first = strict_directory/"proposal-001-1.json"
    if strict["status"] == "invalid_proposal" and first.exists():
        transcript.append(("assistant", json.loads(first.read_text())["proposal"]))
        notice = (f"Your answer was rejected before execution: {strict['message']}. No solver has been run. "
                  "This is your one repair: return a corrected complete answer that meets the rules.")
    else:
        notice = (f"Your answer was cut off at the {strict.get('max_output_tokens')}-token output limit before it was "
                  "complete, so nothing could be used. No solver has been run. This is your one repair: return a "
                  f"complete answer that fits, with less reasoning, keeping every program well under "
                  f"{MAX_CODE_CHARS:,} characters.")
    transcript.append(("user", notice))
    messages = build_messages(transcript)
    record = {"one_shot": True, "repair": True, "repair_of": {"status": strict["status"], "message": strict["message"]},
              "repair_notice": notice, "requests": [], "status": None, "message": None,
              "max_output_tokens": backend.config.max_output_tokens, "model_timeout_seconds": model_timeout,
              "prompt_version": strict.get("prompt_version", ONE_SHOT_PROMPT_VERSION)}
    version = copy.deepcopy(starting)
    for request in range(1, max_requests+1):
        call = directory/("model-call-001-2" if request == 1 else f"model-call-001-2-resend{request}")
        try:
            result = backend.complete(messages, schema, call, timeout=model_timeout)
        except ModelCallError as error:
            record["requests"].append({"directory": call.name, "error": str(error), "usage": error.usage,
                                       "wall_seconds": error.wall_seconds})
            if any(code in str(error) for code in RETRYABLE_FAILURES) and request < max_requests:
                continue
            record.update(status="model_failure", message=str(error))
            break
        record["requests"].append({"directory": call.name, "usage": result["usage"],
                                   "wall_seconds": result["wall_seconds"], "resolved_model": result["resolved_model"]})
        proposal = result["proposal"]
        write_json(directory/"proposal-001-2.json", {"proposal": proposal})
        try:
            version, changed = apply_proposal(starting, proposal, False, classes)
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            record.update(status="invalid_proposal", message=str(error)[:1000])
            break
        write_json(directory/"proposal-001-2.json", {"proposal": proposal, "changed_families": changed,
                                                     **version.snapshot()})
        record.update(status="ok", changed_families=changed)
        break
    write_json(directory/"one_shot.json", record)
    append_event(directory/"events.jsonl", {"iteration": 1, "attempt": 2, "event": "one_shot_repair", **record})
    write_json(directory/"frozen_agent.json", version.snapshot())
    write_program(directory/"programs/frozen", version.program)
    return version


def request_one_shot(directory, starting, evolution, development, backend, protocol=None,
                     evolution_feedback=None, validation_feedback=None, budget=None, model_timeout=900,
                     max_input_chars=500000, max_requests=3):
    """The single-run baseline's request: one model call carrying what RSI sees in its first round (task, contract,
    starting programs and their diagnostics), with no feedback, revision, self-test, or gate. Provider failures
    (transport, HTTP 429/5xx) are re-sent up to max_requests; any answer the model gives is final. Writes
    one_shot.json and programs/frozen (the returned programs, or the starting ones when the answer is invalid).
    Starting-program feedback may be passed in (it is deterministic); otherwise it is computed with budget."""
    protocol = protocol or CodeProtocol()
    directory = project_path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    if evolution_feedback is None:
        budget.phase = "initial_evolution"
        evolution_feedback = protocol.evaluate(evolution, starting, budget)
    if validation_feedback is None:
        budget.phase = "initial_validation"
        validation_feedback = protocol.evaluate(development, starting, budget)
    write_json(directory/"evolution-001.json", evolution_feedback)
    write_json(directory/"initial_validation.json", validation_feedback)
    write_json(directory/"initial_agent.json", starting.snapshot())
    write_program(directory/"programs/initial", starting.program)
    transcript, schema, classes = one_shot_transcript(starting, evolution, development, protocol,
                                                      evolution_feedback, validation_feedback)
    messages = build_messages(transcript)
    record = {"one_shot": True, "requests": [], "status": None, "message": None,
              "max_output_tokens": backend.config.max_output_tokens, "model_timeout_seconds": model_timeout,
              "prompt_version": ONE_SHOT_PROMPT_VERSION}
    version = copy.deepcopy(starting)
    if sum(len(message["content"]) for message in messages) > max_input_chars:
        record.update(status="input_character_budget_exhausted")
    for request in range(1, max_requests+1):
        if record["status"] is not None:
            break
        call = directory/("model-call-001-1" if request == 1 else f"model-call-001-1-resend{request}")
        try:
            result = backend.complete(messages, schema, call, timeout=model_timeout)
        except ModelCallError as error:
            record["requests"].append({"directory": call.name, "error": str(error), "usage": error.usage,
                                       "wall_seconds": error.wall_seconds})
            if any(code in str(error) for code in RETRYABLE_FAILURES) and request < max_requests:
                continue
            record.update(status="model_failure", message=str(error))
            break
        record["requests"].append({"directory": call.name, "usage": result["usage"],
                                   "wall_seconds": result["wall_seconds"], "resolved_model": result["resolved_model"]})
        proposal = result["proposal"]
        write_json(directory/"proposal-001-1.json", {"proposal": proposal})
        try:
            version, changed = apply_proposal(starting, proposal, False, classes)
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            record.update(status="invalid_proposal", message=str(error)[:1000])
            break
        write_json(directory/"proposal-001-1.json", {"proposal": proposal, "changed_families": changed,
                                                     **version.snapshot()})
        record.update(status="ok", changed_families=changed)
    write_json(directory/"one_shot.json", record)
    append_event(directory/"events.jsonl", {"iteration": 1, "attempt": 1, "event": "one_shot", **record})
    write_json(directory/"frozen_agent.json", version.snapshot())
    write_program(directory/"programs/frozen", version.program)
    return version


def finish_one_shot(directory, starting, evolution, development, budget, protocol=None, max_output_tokens=None):
    """Score the frozen single-run programs on all public cases and write report.json."""
    from .costs import load_prices, response_cost
    protocol = protocol or CodeProtocol()
    directory = project_path(directory)
    record = json.loads((directory/"one_shot.json").read_text())
    version = copy.deepcopy(starting)
    version.program = {path.stem: path.read_text() for path in sorted((directory/"programs/frozen").glob("*.py"))}
    budget.ledger_path = directory/"solve_calls.jsonl"
    budget.phase = "frozen_public_evaluation"
    frozen = protocol.evaluate(evolution+development, version, budget, solution_directory=directory/"solutions")
    write_json(directory/"frozen_results.json", frozen)
    prices = load_prices()
    costs = [response_cost(directory/r["directory"]/"response.json", starting.model, prices) for r in record["requests"]]
    resolved = sorted({r["resolved_model"] for r in record["requests"] if r.get("resolved_model")})
    report = {"model": starting.model, "resolved_models": resolved, "one_shot": True,
              "one_shot_status": record["status"], "one_shot_message": record["message"],
              "recursive_policy_updates_enabled": False, "accepted_candidates": int(record["status"] == "ok"),
              "accepted_generation_updates": 0, "requested_rounds": 1, "completed_rounds": 1, "attempts_per_round": 1,
              "initial_validation": metric_summary(json.loads((directory/"initial_validation.json").read_text())),
              "selected_validation": metric_summary([row for row in frozen if row["split"] == "development"]),
              "frozen_public": metric_summary(frozen),
              "frozen_by_split": {split: metric_summary([row for row in frozen if row["split"] == split])
                                  for split in ("evolution", "development")},
              "frozen_version_id": version.version_id, "frozen_program": program_digest(version.program),
              "changed_from_start": sorted(f for f in version.program if version.program[f] != starting.program.get(f)),
              "budget": {**budget.report(), "max_model_calls": 1, "model_calls": len(record["requests"]),
                         "model_requests_resent": max(0, len(record["requests"])-1),
                         "max_output_tokens_per_request": max_output_tokens,
                         "model_cost_usd": sum(costs) if all(c is not None for c in costs) else None,
                         "model_wall_seconds": sum(r["wall_seconds"] for r in record["requests"]),
                         "model_usage": [r["usage"] for r in record["requests"]]},
              "complete_frozen_evaluation": all(row["status"] == "ok" for row in frozen)}
    write_json(directory/"report.json", report)
    return report, version, frozen


def run_one_shot(directory, starting, evolution, development, budget, backend, protocol=None, model_timeout=900,
                 max_input_chars=500000, **_):
    """Single run: one request, then the frozen programs are scored (see request_one_shot)."""
    request_one_shot(directory, starting, evolution, development, backend, protocol, budget=budget,
                     model_timeout=model_timeout, max_input_chars=max_input_chars)
    return finish_one_shot(directory, starting, evolution, development, budget, protocol,
                           backend.config.max_output_tokens)


def write_tables(directory, arms):
    rows, families, summary = [], [], []
    for arm, (report, version, results) in arms.items():
        for result in results:
            rows.append({"arm": arm, "task_id": result["task_id"], "family": result["family"],
                         "split": result["split"], "status": result["status"],
                         **{key: result.get(key) for key in METRICS}, "passed": result["passed"],
                         "wall_seconds": result.get("wall_seconds"), "cpu_seconds": result.get("cpu_seconds"),
                         "peak_memory_mb": result.get("peak_memory_mb"), "returned_nx": result.get("nx"),
                         "returned_ny": result.get("ny"), "returned_nt": result.get("nt"),
                         "program_sha256": result.get("program_sha256"),
                         **({"case_id": result["case_id"]} if "case_id" in result else {}),
                         **({"family_correct": result.get("family_correct")} if "family_correct" in result else {})})
        for split, split_report in report["frozen_by_split"].items():
            for family, value in split_report["families"].items():
                families.append({"arm": arm, "split": split, "family": family, "cases": value["count"],
                                 "pass_rate": value["pass_rate"],
                                 **{"mean_"+key: value["mean_"+key] for key in METRICS}})
        value = report["frozen_public"]
        summary.append({"arm": arm, "cases": value["count"], "pass_rate": value["pass_rate"],
                        **{"mean_"+key: value["mean_"+key] for key in METRICS},
                        "accepted_candidates": report["accepted_candidates"],
                        "completed_rounds": report["completed_rounds"],
                        "solver_calls": report["budget"]["solver_calls"], "model_calls": report["budget"]["model_calls"],
                        "cpu_seconds": report["budget"]["total_cpu_seconds"],
                        "wall_seconds": report["budget"]["elapsed_wall_seconds"]})
    for filename, records in (("per_case_l2.csv", rows), ("per_family_l2.csv", families), ("comparison.csv", summary)):
        with project_path(directory/filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return summary


def _format(value):
    return "n/a" if value is None else f"{value:.6g}"


def write_summary(output, metadata):
    lines = [f"Code-writing single-run and RSI agents on the 48 public manufactured problems ({metadata['start']} start).",
             f"Model: {metadata['model_config']['model']}. Evaluator: {metadata['evaluator_version']}. "
             f"Sandbox: {metadata['sandbox']['sandbox_version']}.", "",
             "| Arm | Passes | Absolute L2 | Relative L2 | Final absolute L2 | Final relative L2 | Accepted / rounds | Solver calls | Model calls |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    reference = metadata["starting_reference"]
    lines.append(f"| starting programs ({metadata['start']}) | "
                 f"{round(reference['pass_rate']*reference['count'])}/{reference['count']} | "
                 + " | ".join(_format(reference["mean_"+key]) for key in METRICS) + " | — | — | — |")
    for name, report in metadata["arms"].items():
        value = report["frozen_public"]
        lines.append(f"| {name} | {round(value['pass_rate']*value['count'])}/{value['count']} | "
                     + " | ".join(_format(value["mean_"+key]) for key in METRICS)
                     + (" | one request | " if report.get("one_shot") else
                        f" | {report['accepted_candidates']}/{report['requested_rounds']} | ")
                     + f"{report['budget']['solver_calls']} | {report['budget']['model_calls']} |")
    lines += ["", "Means are arithmetic over all 48 public cases. Validation results were queried repeatedly during "
              "selection; these are development measurements, not sealed-test evidence."]
    if "continuation" in metadata:
        lines += ["", "| Continuation branch | Initial validation macro rel. L2 | Selected | Future gain | Accepted |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for name, report in metadata["continuation"].items():
            before = report["initial_validation"]["macro_relative_l2"]
            after = report["selected_validation"]["macro_relative_l2"]
            lines.append(f"| {name} | {_format(before)} | {_format(after)} | {_format(before-after)} | "
                         f"{report['accepted_candidates']}/{report['requested_rounds']} |")
        lines += ["", f"Evolved-minus-initial workflow advantage: {_format(metadata['continuation_policy_advantage'])}."]
    project_path(output/"summary.md").write_text("\n".join(lines)+"\n")


def load_public_tasks(root, package, families):
    """Public evolution/development tasks for the requested families, in family order."""
    tasks = {"evolution": [], "development": []}
    if any(family in FAMILIES for family in families):
        published = public_tasks(root, package)
        for split in tasks:
            tasks[split] += [task for task in published[split] if task.family in families]
    if any(family in WAVE_FAMILIES for family in families):
        for split in tasks:
            tasks[split] += [task for task in load_wave_tasks(root, split) if task.family in families]
    if any(family in GROUP_FAMILIES for family in families):
        for split in tasks:
            tasks[split] += [task for task in load_group_tasks(root, split) if task.family in families]
    for split, items in tasks.items():
        items.sort(key=lambda task: (families.index(task.family), task.task_id))
        if {task.family for task in items} != set(families):
            raise ValueError(f"Missing public {split} tasks for some requested families")
    return tasks


def cpu_model():
    with open("/proc/cpuinfo") as stream:
        for line in stream:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def _arm_worker(core, backend_factory, model_config, limits, kwargs, base_url=None):
    """Run one arm in its own process, pinned to one CPU core (sandboxed solves inherit the pinning).
    The budget is created here so its CPU meter starts from this process's own counters. base_url replaces the
    model endpoint for this arm (RSI-PDE Bench: each arm calls the model through its own metering proxy)."""
    if core is not None:
        os.sched_setaffinity(0, {core})
    if base_url is not None:
        model_config = dataclasses.replace(model_config, base_url=base_url)
    max_calls, wall_seconds, cpu_seconds = limits
    arm = run_one_shot if kwargs.pop("one_shot", False) else run_arm
    return arm(backend=backend_factory(model_config),
               budget=Budget(max_calls, wall_seconds, max_cpu_seconds=cpu_seconds), **kwargs)


def run_comparison(root, package, output, model_config, rounds=6, max_calls=None, wall_seconds=5400,
                   cpu_seconds=3600, max_model_calls=None, probe_rounds=1, backend_factory=ModelBackend,
                   max_input_chars=500000, model_timeout=900, start="classical", attempts_per_round=3,
                   families=FAMILIES, protocol=None, time_budget_seconds=None, concurrent_arms=False, cores=None,
                   history_rounds=None, max_rounds=200, max_validation_submissions=None, spend_cap_usd=None,
                   best_of=1, gate="standard", experiments_per_round=0, arm_base_urls=None):
    """Single run (one model request, frozen as returned) vs RSI. With time_budget_seconds, each arm improves until a shared wall-clock deadline
    (rounds are capped only by max_rounds); wall_seconds/cpu_seconds stay hard ceilings that must also cover
    the frozen evaluation. concurrent_arms runs both arms at once, each pinned to its own core from cores."""
    root, output = project_path(root), project_path(output)
    if time_budget_seconds is not None:
        if time_budget_seconds <= 0 or probe_rounds:
            raise ValueError("A time budget must be positive and is run without continuation probes")
        if wall_seconds < time_budget_seconds+1800 or cpu_seconds < time_budget_seconds+1800:
            raise ValueError("Hard wall/CPU ceilings must exceed the time budget by at least 30 minutes")
        rounds = max_rounds
    if concurrent_arms:
        cores = list(cores or sorted(os.sched_getaffinity(0))[:2])
        if len(cores) != 2 or len(set(cores)) != 2 or not set(cores) <= os.sched_getaffinity(0):
            raise ValueError("Concurrent arms need two distinct allowed CPU cores")
    package = project_path(package) if package is not None else None  # only the published 1D set needs it
    if rounds < 1 or probe_rounds < 0:
        raise ValueError("At least one main round and nonnegative continuation rounds are required")
    if start not in STARTS:
        raise ValueError(f"start must be one of {STARTS}")
    if attempts_per_round < 1:
        raise ValueError("At least one attempt per round is required")
    options = rsi_options(best_of, gate, experiments_per_round)  # RSI arm only; the single run stays one request
    protocol = protocol or CodeProtocol()
    families = tuple(families)
    if protocol.name == "code" and (not families or len(set(families)) != len(families)
                                    or any(f not in ALL_FAMILIES for f in families)):
        raise ValueError(f"families must be distinct members of {ALL_FAMILIES}")
    tasks = protocol.load_tasks(root, package, families)
    evolution, development = tasks["evolution"], tasks["development"]

    def solver_calls(count):
        per_round = len(evolution)*(1+attempts_per_round*best_of+experiments_per_round)+len(development)
        return len(development)+per_round*count+len(evolution)+len(development)
    required = solver_calls(rounds)
    if max_calls is None:
        max_calls = required
    if max_calls < required:
        raise ValueError(f"Each arm needs {required} calls including frozen evaluation")
    if max_model_calls is None:
        max_model_calls = rounds*(attempts_per_round*best_of+experiments_per_round)
    if max_model_calls < rounds:
        raise ValueError("The model allowance must cover all requested rounds")
    sandbox = sandbox_self_test()
    output.mkdir(parents=True, exist_ok=False)
    starting = CodeVersion(model=model_config.model, program=protocol.starting_program(start, families),
                           improvement_workflow=getattr(protocol, "initial_workflow", INITIAL_WORKFLOW))
    metadata = {"kind": "code_writing_single_vs_rsi", "protocol_version": PROTOCOL_VERSION,
                "task_protocol": protocol.name, "task_protocol_metadata": protocol.metadata(tasks), "created_utc": datetime.now(timezone.utc).isoformat(),
                "model_config": model_config.to_dict(), "evaluator_version": EVALUATOR_VERSION, "sandbox": sandbox,
                "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((root/"src").rglob("*.py"))},
                "families": list(families),
                "dataset_manifest_sha256": (hashlib.sha256((package/"manifest.json").read_bytes()).hexdigest()
                                            if any(f in FAMILIES for f in families) else None),
                "wave_manifest_sha256": (hashlib.sha256((root/"benchmarks/wave2d/manifest.json").read_bytes()).hexdigest()
                                         if any(f in WAVE_FAMILIES for f in families) else None),
                "group_manifest_sha256": (
                    hashlib.sha256((root/"benchmarks/group_fpde/manifest.json").read_bytes()).hexdigest()
                    if any(f in GROUP_FAMILIES for f in families) else None),
                "task_ids": {split: [t.task_id for t in items] for split, items in tasks.items()},
                "start": start, "starting_program": program_digest(starting.program),
                "configuration": {"rounds": rounds, "max_solver_calls_per_arm": max_calls,
                                  "max_model_calls_per_arm": max_model_calls, "max_wall_seconds_per_arm": wall_seconds,
                                  "max_cpu_seconds_per_arm": cpu_seconds, "cpu_threads": 1, "gpus": 0,
                                  "max_input_characters_per_request": max_input_chars,
                                  "model_timeout_seconds": model_timeout, "probe_rounds": probe_rounds,
                                  "start": start, "attempts_per_round": attempts_per_round,
                                  "self_test_broken_relative_l2": BROKEN_RELATIVE_L2,
                                  "time_budget_seconds_per_arm": time_budget_seconds,
                                  "concurrent_arms": concurrent_arms, "history_rounds": history_rounds,
                                  "max_validation_submissions": max_validation_submissions,
                                  "spend_cap_usd_per_arm": spend_cap_usd, "arm_base_urls": arm_base_urls,
                                  **({"rsi_options": options} if options else {})},
                "hardware": {"node": platform.node(), "cpu_model": cpu_model(), "logical_cpus": os.cpu_count(),
                             "allowed_cores": sorted(os.sched_getaffinity(0)),
                             "arm_cores": dict(zip(("single", "rsi"), cores)) if concurrent_arms else None,
                             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
                "interpretation": "code-writing agent; public development/selection results, not sealed-test RSI evidence",
                "single_arm": "one_shot", "arms": {}, "sealed_cases_used": False}
    write_json(output/"manifest.json", metadata)
    # Reference only: the starting programs on all public cases, outside every arm budget.
    reference_budget = Budget(len(evolution)+len(development), wall_seconds, max_cpu_seconds=cpu_seconds,
                              ledger_path=None)
    reference_results = protocol.evaluate(evolution+development, starting, reference_budget)
    write_json(output/"starting_reference_results.json", reference_results)
    metadata["starting_reference"] = metric_summary(reference_results)
    arms = {}

    def arm_kwargs(name, deadline):
        return dict(directory=output/name, starting=starting, evolution=evolution, development=development,
                    rounds=rounds, recursive=name == "rsi", max_model_calls=max_model_calls, max_input_chars=max_input_chars, model_timeout=model_timeout,
                    attempts_per_round=attempts_per_round, protocol=protocol, deadline=deadline,
                    history_rounds=history_rounds, max_validation_submissions=max_validation_submissions,
                    spend_cap_usd=spend_cap_usd, one_shot=name == "single",
                    **(options if name == "rsi" else {}))
    if concurrent_arms:
        # One shared deadline: both arms improve in the same wall-clock window on the same node.
        deadline = None if time_budget_seconds is None else time.time()+time_budget_seconds
        metadata["arm_start_utc"] = datetime.now(timezone.utc).isoformat()
        with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("fork")) as pool:
            futures = {name: pool.submit(_arm_worker, core, backend_factory, model_config,
                                         (max_calls, wall_seconds, cpu_seconds), arm_kwargs(name, deadline),
                                         (arm_base_urls or {}).get(name))
                       for name, core in zip(("single", "rsi"), cores)}
            arms = {name: future.result() for name, future in futures.items()}
        for name in ("single", "rsi"):
            metadata["arms"][name] = arms[name][0]
        write_json(output/"manifest.json", metadata)
    else:
        for name in ("single", "rsi"):
            deadline = None if time_budget_seconds is None else time.time()+time_budget_seconds
            arms[name] = _arm_worker(None, backend_factory, model_config, (max_calls, wall_seconds, cpu_seconds),
                                     arm_kwargs(name, deadline), (arm_base_urls or {}).get(name))
            metadata["arms"][name] = arms[name][0]
            write_json(output/"manifest.json", metadata)
    metadata["comparison"] = write_tables(output, arms)
    metadata["matched_actual_solver_calls"] = len({a[0]["budget"]["solver_calls"] for a in arms.values()}) == 1
    metadata["matched_actual_model_calls"] = len({a[0]["budget"]["model_calls"] for a in arms.values()}) == 1
    resolved = {m for a in arms.values() for m in a[0]["resolved_models"]}
    metadata["resolved_model_identity_consistent"] = len(resolved) <= 1
    metadata["all_requested_rounds_complete"] = all(a[0]["completed_rounds"] == a[0]["requested_rounds"]
                                                    for a in arms.values())
    metadata["all_within_resource_ceilings"] = all(a[0]["budget"]["within_resource_ceilings"] for a in arms.values())
    if probe_rounds:
        continuations = {}
        source = arms["rsi"][1]
        calls = solver_calls(probe_rounds)
        for name in ("initial_policy", "evolved_policy"):
            start = copy.deepcopy(source)
            if name == "initial_policy":
                start.improvement_workflow = INITIAL_WORKFLOW
            continuations[name] = run_arm(output/"continuation"/name, start, evolution, development,
                                         probe_rounds, False, Budget(calls, wall_seconds, max_cpu_seconds=cpu_seconds),
                                         backend_factory(model_config),
                                         probe_rounds*(attempts_per_round*best_of+experiments_per_round),
                                         max_input_chars, model_timeout, attempts_per_round, protocol, **options)
        metadata["continuation"] = {name: item[0] for name, item in continuations.items()}
        write_tables(output/"continuation", continuations)
        gains = {name: item[0]["initial_validation"]["macro_relative_l2"]-
                      item[0]["selected_validation"]["macro_relative_l2"]
                 for name, item in continuations.items()}
        metadata["continuation_policy_advantage"] = gains["evolved_policy"]-gains["initial_policy"]
        metadata["continuation_distinct_workflows"] = (source.improvement_workflow != INITIAL_WORKFLOW)
        metadata["continuation_matched_actual_calls"] = len({item[0]["budget"]["solver_calls"]
                                                              for item in continuations.values()}) == 1
    write_json(output/"manifest.json", metadata)
    write_summary(output, metadata)
    return metadata
