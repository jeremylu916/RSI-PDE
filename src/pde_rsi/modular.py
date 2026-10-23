"""Harness v2 (2026-10-20): modular solver programs, patch edits, a per-part gate, and run-time self-checks.

A program is a dict of parts. The router "identify" defines solve(problem): it identifies the equation class
and coefficients, calls SOLVERS[class](problem, model), and returns the solution with "model". An optional
"common" part holds shared helpers. Each class part is named after a library class and registers
solve_<class>(problem, model) in SOLVERS. Parts are checked separately (each within MAX_CODE_CHARS) and
assembled into one module behind a trusted prelude. A proposal replaces a part ("code") or patches it ("edits":
exact old/new pairs). A candidate that changes only class parts leaves routing unchanged. Each changed part is
then gated on the validation cases routed to it (by the class the router reports), and accepted parts are merged.
"""

import ast
import copy

from .codegen import MAX_CODE_CHARS, ProgramRejected, check_program
from .evaluator import acceptance_gate

ROUTER, COMMON = "identify", "common"
PRELUDE = '"""Harness prelude: class parts register their solvers here."""\n\nSOLVERS = {}\n'
MAX_EDITS = 20
INITIAL_WORKFLOW = (
    "Inspect errors by class and case. Identify the dominant error source: wrong class or coefficients (check the "
    "data misfit), spatial or temporal truncation, initial-time singularity, transport, nonlinear coupling, "
    "interpolation onto evaluator nodes, or execution failure. Fix it with a targeted patch to the part that causes "
    "it; replace a whole part only when it must be rewritten. Keep a safety margin: aim for errors well below the "
    "tolerance, with a convergence check that uses the remaining time, because hidden cases differ in "
    "coefficients. Preserve exact IC/BC and resource limits. Use only the supplied task and diagnostic data."
)

CONTRACT_ADDENDUM = f"""
Program structure (modular). The program is a set of parts assembled into one module in this order: a harness
prelude that defines SOLVERS = {{}}, then "{COMMON}" (optional shared helpers), then one part per library class you
support, named exactly as the class (for example "burgers"), then "{ROUTER}".
- Each class part defines solve_<class>(problem, model) that returns the solution dict (without "model"), and ends
  with SOLVERS["<class>"] = solve_<class>. It must not define solve.
- "{ROUTER}" defines solve(problem): it identifies the class and coefficients, calls SOLVERS[cls](problem, model),
  and returns {{**solution, "model": model}}. Keep "{ROUTER}" to identification and dispatch (the starting router
  shows the pattern) and put all solving code in class parts: only changes confined to class parts get the
  per-part gate, while any change to "{ROUTER}" is gated on every validation case at once.
- Every part is at most {MAX_CODE_CHARS:,} characters. Top-level names must be unique across parts: prefix
  helpers with the part name. Imports are per part (repeat them where needed).
- A program_updates item either replaces or creates a part ("code": the complete part source) or patches it
  ("edits": up to {MAX_EDITS} items {{"old": exact text that occurs exactly once in the active part, "new": its
  replacement}}). Parts you omit stay unchanged. Prefer small patches.
- Gate: a candidate that changes only class parts is judged per part, on the validation cases the router sends to
  that part (by the class it reports). Improving parts are kept even when another part of the same candidate is
  rejected. A change to "{ROUTER}" or "{COMMON}" is judged on all validation cases.

Self-checks at run time (recommended). The data are available for every case, including hidden test cases whose
coefficients you have never seen.
1. Choose the class and coefficients by fitting. For each plausible class, solve with fitted coefficients and
   compute the data misfit m = sqrt(mean(((u at the data points) - data_u) / noise_std)**2). For the true
   equation m is about 1; otherwise it is much larger.
2. Refine until converged. Compare solutions at successive resolutions and refine until the estimated error is
   below relative_l2_tolerance / 5, or until about half of max_seconds is used. problem.elapsed_seconds() gives
   the wall seconds since the solve process started; max_seconds includes start-up.
Hidden cases differ in coefficients, so aim for errors well below the tolerance, not just under it."""

SYSTEM_PROGRAM_SENTENCE = (
    "Each program_updates item gives a family (a part name: \"identify\", \"common\", or a library class), a "
    "change_summary, and either code (the COMPLETE source of that part as a JSON string with escaped newlines and "
    "quotes) or edits (a list of exact old/new text replacements in the active part). Parts you omit keep their "
    "active source. "
)


def part_names(classes):
    return [ROUTER, COMMON, *classes]


ROUTER_SCAFFOLD = '''"""Router: identify the class and coefficients, then dispatch to that class's part."""


def identify_case(problem):
    raise NotImplementedError("No identification has been written yet")


def solve(problem):
    model = identify_case(problem)
    solver = SOLVERS.get(model["family"])
    if solver is None:
        raise NotImplementedError("No part solves this class yet")
    return {**solver(problem, model), "model": model}
'''


def starting_program():
    """The router scaffold: identification and dispatch only; solvers go into class parts."""
    return {ROUTER: ROUTER_SCAFFOLD}


def order(program):
    return [name for name in (COMMON, *sorted(p for p in program if p not in (ROUTER, COMMON)), ROUTER) if name in program]


def assemble(program):
    """One executable module: the prelude, then common, the class parts, and the router."""
    return PRELUDE+"".join(f"\n\n# ---- part: {name}\n{program[name]}" for name in order(program))


def _top_level_names(tree):
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names += [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return names


def check_part(name, code, classes):
    """Static policy for one part plus the structural rules of its role."""
    label = f"{name} part"
    if name == ROUTER:
        check_program(code, name)
        return
    if name != COMMON and name not in classes:
        raise ProgramRejected(f"unknown part {name!r}: use \"{ROUTER}\", \"{COMMON}\", or a library class name")
    check_program(code, name, require_solve=False)
    tree = ast.parse(code)
    top = _top_level_names(tree)
    if "solve" in top:
        raise ProgramRejected(f"{label} must not define solve; only \"{ROUTER}\" does")
    if "SOLVERS" in top:
        raise ProgramRejected(f"{label} must not rebind SOLVERS")
    if name in classes:
        function = f"solve_{name}"
        if function not in top:
            raise ProgramRejected(f"{label} must define {function}(problem, model)")
        registered = any(isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == "SOLVERS"
            and isinstance(t.slice, ast.Constant) and t.slice.value == name for t in node.targets)
            for node in tree.body)
        if not registered:
            raise ProgramRejected(f'{label} must register SOLVERS["{name}"] = {function}')


def check_modular(program, classes):
    for name, code in program.items():
        check_part(name, code, classes)
    if ROUTER not in program:
        raise ProgramRejected(f'the program needs a "{ROUTER}" part')
    seen = {}
    for name in order(program):
        for top in _top_level_names(ast.parse(program[name])):
            if top in seen and top not in ("SOLVERS",):
                raise ProgramRejected(f"top-level name {top!r} is defined in both {seen[top]!r} and {name!r}")
            seen.setdefault(top, name)


def apply_edits(source, edits, name):
    if not isinstance(edits, list) or not 1 <= len(edits) <= MAX_EDITS:
        raise ValueError(f"edits for {name} must list 1 to {MAX_EDITS} replacements")
    for index, edit in enumerate(edits, 1):
        if not isinstance(edit, dict) or set(edit) != {"old", "new"} or not all(
                isinstance(edit[k], str) for k in ("old", "new")) or not edit["old"]:
            raise ValueError(f"edit {index} for {name} needs nonempty string old and string new")
        count = source.count(edit["old"])
        if count != 1:
            raise ValueError(f"edit {index} for {name}: old text occurs {count} times in the active part (must be once)")
        source = source.replace(edit["old"], edit["new"], 1)
    return source


def update_schema(classes):
    return {"type": "object", "additionalProperties": False,
            "properties": {"family": {"type": "string", "enum": part_names(classes)},
                           "change_summary": {"type": "string", "minLength": 1, "maxLength": 600},
                           "code": {"type": "string", "minLength": 1, "maxLength": MAX_CODE_CHARS},
                           "edits": {"type": "array", "minItems": 1, "maxItems": MAX_EDITS,
                                     "items": {"type": "object", "additionalProperties": False,
                                               "properties": {"old": {"type": "string", "minLength": 1},
                                                              "new": {"type": "string"}},
                                               "required": ["old", "new"]}}},
            "required": ["family", "change_summary"]}


def apply_updates(program, updates, classes):
    """New program dict and the changed part names; raises ValueError/ProgramRejected on an invalid proposal."""
    names = part_names(classes)
    if not isinstance(updates, list) or not 1 <= len(updates) <= len(names):
        raise ValueError(f"program_updates must list one to {len(names)} parts")
    program, changed = dict(program), []
    for update in updates:
        if not isinstance(update, dict) or "family" not in update or "change_summary" not in update:
            raise ValueError("Each program update requires family and change_summary, plus code or edits")
        name = update["family"]
        if name not in names or name in changed:
            raise ValueError(f"Program update parts must be unique members of {names}")
        if set(update) - {"family", "change_summary", "code", "edits"} or ("code" in update) == ("edits" in update):
            raise ValueError(f"The update for {name} needs exactly one of code or edits")
        if not isinstance(update["change_summary"], str) or not 1 <= len(update["change_summary"]) <= 600:
            raise ValueError(f"Invalid change_summary for {name}")
        if "code" in update:
            if not isinstance(update["code"], str):
                raise ValueError(f"code for {name} must be a string")
            program[name] = update["code"]
        else:
            if name not in program:
                raise ValueError(f"edits need an existing part; {name} does not exist yet (send code)")
            program[name] = apply_edits(program[name], update["edits"], name)
        changed.append(name)
    check_modular(program, classes)
    return program, changed


def compact_update(update, digest):
    """History form of an update: code or edit text replaced by a digest."""
    if "code" in update:
        return {"family": update["family"], "change_summary": update["change_summary"],
                "code": f"omitted; sha256={digest(update['code'])}, {len(update['code'])} characters"}
    return {"family": update["family"], "change_summary": update["change_summary"],
            "edits": f"omitted; {len(update.get('edits') or [])} edits"}


def _routes(rows):
    return {r["task_id"]: r.get("identified_family") for r in rows}


def part_gate(active, candidate, changed, active_rows, candidate_rows, **gate_options):
    """Per-part decisions for a candidate that changes only class parts.

    Returns (decision, merged_program, merged_rows). decision has the overall gate fields plus "parts": per changed
    part {accepted, reasons, cases}. merged_program is None when nothing is accepted. When the router or common
    part changed, or a case is routed to a changed part in one version but a different changed part in the other,
    the whole candidate is gated as one."""
    def whole():
        gate = acceptance_gate(active_rows, candidate_rows, **gate_options)
        return {**gate, "mode": "whole"}, (candidate if gate["accepted"] else None), candidate_rows
    if not changed or ROUTER in changed or COMMON in changed:
        return whole()
    before, after = _routes(active_rows), _routes(candidate_rows)
    affected = {}
    for task_id in before:
        parts = {p for p in (before[task_id], after[task_id]) if p in changed}
        if len(parts) > 1:
            return whole()
        if parts:
            affected.setdefault(parts.pop(), []).append(task_id)
    decisions, accepted = {}, []
    for part in changed:
        ids = set(affected.get(part, []))
        if not ids:
            decisions[part] = {"accepted": False, "reasons": ["no validation case is routed to this part"], "cases": 0}
            continue
        gate = acceptance_gate([r for r in active_rows if r["task_id"] in ids],
                               [r for r in candidate_rows if r["task_id"] in ids], **gate_options)
        decisions[part] = {"accepted": gate["accepted"], "reasons": gate["reasons"], "cases": len(ids),
                           "relative_gain": gate["relative_gain"]}
        if gate["accepted"]:
            accepted.append(part)
    if not accepted:
        reasons = [reason for d in decisions.values() for reason in d["reasons"]]
        return {"accepted": False, "reasons": reasons, "relative_gain": 0.0, "mode": "per_part",
                "parts": decisions}, None, candidate_rows
    merged = copy.deepcopy(active)
    for part in accepted:
        merged[part] = candidate[part]
    moved = {task_id for part in accepted for task_id in affected[part]}
    by_id = {r["task_id"]: r for r in candidate_rows}
    merged_rows = [by_id[r["task_id"]] if r["task_id"] in moved else r for r in active_rows]
    overall = acceptance_gate(active_rows, merged_rows, **gate_options)
    return {"accepted": True, "reasons": [r for d in decisions.values() if not d["accepted"] for r in d["reasons"]],
            "relative_gain": overall["relative_gain"], "mode": "per_part", "parts": decisions,
            "accepted_parts": accepted}, merged, merged_rows


def load_program(directory):
    """Executable code of a written program directory (programs/frozen or similar). A single-module program of
    the earlier harness is returned unchanged; a modular one is assembled."""
    parts = {path.stem: path.read_text() for path in sorted(directory.glob("*.py"))}
    if set(parts) == {ROUTER} and "SOLVERS" not in parts[ROUTER]:
        return parts[ROUTER]
    return assemble(parts)
