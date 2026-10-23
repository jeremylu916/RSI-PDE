# PDE-RSI RSI harness vs. Claude Code: a design comparison

The earlier comparison (`runs/code-agent-results-pdebench2h-20261017/claude_code_comparison.md`) compared outcomes:
one 2-hour run of each, same model, same cases. This document compares the designs, component by component. For
each component it gives what each design does, what that implies, and what it actually did in the October 17–18
pdebench runs (Opus 5.5 and Sonnet 4.5, 2 h, 1 % noise).

## 1. The two loops

**PDE-RSI harness (RSI arm)**

```
system prompt + solver contract + 32 public case views
repeat until the 2 h deadline:                                   ← harness decides when
    run active program on 16 training cases                      ← harness runs code
    send: active program, training diagnostics (with labels),
          validation diagnostics (no labels), active workflow, memory
    up to 3 attempts:
        model → ONE JSON proposal: full program + next workflow + memory lesson
        static check → self-test on 4 training cases
    first passing candidate → validation gate (16 cases)          ← fixed acceptance rule
    accepted → program, workflow, memory become active             ← gated self-modification
freeze the active program → score public + sealed cases
```

**Claude Code**

```
system prompt (Claude Code's own) + README.md (same contract + library) + files
repeat until the agent stops:                                    ← agent decides when
    model → tool call (Read / Write / Edit / Bash ...)            ← agent runs code itself
    tool result appended to the conversation
    (agent may call tools/evaluate.py: training or validation diagnostics, any time)
the file solver.py as it is at the end → score public + sealed cases   ← no acceptance rule
```

## 2. Component by component

| Component | RSI harness | Claude Code | What happened in the runs |
| --- | --- | --- | --- |
| **Action unit** | One structured proposal (whole program, ≤ 30k chars) per model call | Many small tool calls (edit a line, run a script) | Opus: 19–23 model calls in 2 h (harness) vs 34 turns, 39 tool calls in 26 min (Claude Code). Sonnet: 216 tool calls (148 Bash, 49 Edit) |
| **Who executes code** | Only the harness, in the solver sandbox | The agent, in its own shell; official scoring still in the solver sandbox | Claude Code ran its own quick checks (data loading, fits) between official evaluations |
| **Feedback schedule** | Fixed: training every round, self-test per attempt, one validation gate per round | On demand: the agent chooses which split, which cases, and when | Opus asked for 20 evaluations: 11 training and 9 validation, often 1–4 cases at a time. Sonnet asked for 39 |
| **Feedback content** | Errors, pass/fail, identified class; labels on training only | The same, plus the program's own prints on training cases | Prints let Claude Code debug inside `solve()`; the harness model never sees them |
| **Validation access** | Only through the gate, once per round, as a decision input | Unlimited and free-form, so it can be used for selection and tuning | Opus checked full validation 4 times (14 → 15 → 16 → 16 passes) before stopping |
| **Acceptance / selection** | Monotone gate: a candidate replaces the active program only if validation improves with no regressions | None: the last file state is the result | Sonnet's full-validation checks read 3, 2, 2 passes, but its final `solver.py` passed 0: a late edit made it worse and nothing stopped it. The harness can never end below its best accepted program |
| **Memory and context** | History replayed with code removed and only the last 6 rounds kept. RSI adds an explicit workflow and memory, carried only on acceptance | One growing conversation with automatic compaction. Files on disk act as memory | Harness requests 57–246 KB per call. Claude Code context peaked at 109k tokens (Opus) and 192k (Sonnet, compacted once) |
| **Self-improvement** | Built in and gated: the agent rewrites its own improvement workflow and lessons; a fixed rule decides inheritance | None built in. Its strategy lives in the model and the transcript; it could write notes or instructions to itself, but nothing evaluates or carries them | Opus RSI changed its workflow 6 times (e.g. "fit grid must equal solve grid"). Claude Code wrote no notes |
| **Stopping** | Deadline-driven: works until the budget ends | Agent-driven: stops when it judges the work done | Opus Claude Code stopped at 26 min. The harness kept going for 2 h, with small gains after ~25 min |
| **Isolation** | The model never touches a shell; the solver sandbox has no network and no project files; static code policy | The agent has a shell, so it must itself be sandboxed (here bubblewrap, network for Bedrock only, web tools off) | Both kept truth out of reach. Claude Code needs the outer sandbox; without it, it could read the project, including references |
| **Cost structure** | Few, large, output-heavy calls (median 22–28k output tokens per Opus call) | Many short turns, each re-reading the cached conversation | Opus harness: $17–19 per arm. Claude Code: ~$2.45 for the real work, but $89 in total because re-prompts reloaded the full context 503 times |
| **Reproducibility** | Fixed schedule; every request, proposal, solve, and gate decision logged | Transcript logged, but the path depends on the agent's choices | Harness arms are directly comparable round by round; Claude Code runs are comparable only by outcome |

## 3. What each design contributes

The RSI harness provides four things Claude Code does not:

- **A guarantee against regression.** The gate keeps the best validated program. Sonnet under Claude Code shows the
  failure it prevents.
- **Explicit, inspectable self-modification.** The workflow and memory are objects that can be measured, for
  example by the continuation test, and compared against a single run (one request, no iteration).
- **Clean separation of selection from evaluation.** Validation enters only as the gate's input, and the sealed
  test only after freezing.
- **Controlled budgets and a fixed schedule**, which make single run vs RSI a fair comparison.

Claude Code provides four things the harness does not:

- **Cheap, fine-grained experimentation.** It tests on 1–4 cases, inspects data, and edits one function.
- **Debugging visibility**, through the program's own prints.
- **Its own judgement of when it is done**, which saves time: Opus reached the final score in 26 minutes.
- **A general tool set** (file edits, shell) without a task-specific protocol.

## 4. How to compare the designs experimentally

The single Claude Code run differs from the harness in about six design factors at once, so its outcome cannot be
attributed to any one of them. A design comparison varies one factor at a time with the same model, cases, budget,
and solver sandbox:

| Arm | Agent runs code | Validation access | Acceptance gate | Inherited workflow and memory | Status |
| --- | :---: | --- | :---: | :---: | --- |
| Single run (one request, frozen as returned) | no | none | no | no | run (redone 2026-10-20) |
| Fixed workflow (iterated, no inherited memory; replaced by the single run) | no | gate only | yes | no | archived |
| Harness RSI | no | gate only | yes | yes | run |
| Claude Code | yes | unlimited | no | no | run (1 repeat) |
| Claude Code + gate | yes | gate only (one validation per session) | yes | no | not run |
| Claude Code + gate + RSI | yes | gate only | yes | yes: the next fresh session gets the accepted solver, workflow, and memory | not run |
| Harness + training tool | partly: may request extra training-case runs inside a round | gate only | yes | yes | not run |

How the factors are isolated:

- **Gate:** Claude Code vs Claude Code + gate.
- **Self-improvement inside an agentic harness:** Claude Code + gate vs Claude Code + gate + RSI.
- **Agentic tool use:** harness RSI vs Claude Code + gate + RSI.

Each arm should run at least 3 times. The sealed test should be the primary metric, together with pass rate,
median L2, time to first 16/16 on validation, and cost.
