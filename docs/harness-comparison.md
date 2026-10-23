# PDE-RSI harness vs. general coding-agent harnesses (Claude Code, Codex CLI)

This compares how the PDE-RSI improvement harness (`src/pde_rsi/code_agents.py`) drives a model with how general
coding-agent harnesses such as Claude Code and OpenAI Codex CLI drive one.

## Summary

The PDE-RSI harness is a **controlled experiment loop**:

- The model never runs code itself.
- Each round it returns one JSON proposal (a full solver program, plus a workflow and memory for RSI).
- The harness executes the proposal in a locked-down sandbox and returns fixed diagnostics.
- A fixed gate decides whether the proposal is accepted.

Claude Code and Codex CLI are **open-ended tool-use loops**. The model reads and writes files, runs commands and
tests, and decides itself when to evaluate and when it is done.

## Dimension by dimension

| Dimension | PDE-RSI harness (single run / RSI) | Claude Code | Codex CLI |
| --- | --- | --- | --- |
| Interaction unit | One structured JSON proposal per attempt (≤ 3 attempts per round) | Many tool calls per turn: Read/Edit/Write files, Bash, search; subagents | Many tool calls per turn: shell commands, file patches |
| Who runs code | Only the harness, in a bubblewrap sandbox; the model sees results | The agent runs code itself (Bash) | The agent runs code itself (sandboxed shell) |
| Feedback | Fixed per round: training diagnostics with labels, validation diagnostics without; self-test and gate results | Whatever the agent's own commands print; it chooses what to test | Same as Claude Code |
| When evaluation happens | Fixed: training every round, self-test per attempt, one validation gate per round | Whenever the agent decides | Whenever the agent decides |
| Acceptance | Validation gate (> 0.5 % macro-L2 gain, no class/case regression, all cases execute) | None built in: the last state of the files is the result | None built in |
| Memory across steps | Replayed conversation (older code removed), plus explicit workflow and memory inherited on acceptance (RSI only) | Full conversation with automatic compaction; project memory files (`CLAUDE.md`) | Conversation context; `AGENTS.md` project instructions |
| Self-modification | RSI may rewrite its own improvement workflow and memory, gated | None: workflow is the model's own behavior plus user instructions | None |
| Stop rule | Fixed rounds or wall-clock deadline (time-budgeted mode); budget and safety stops | Agent ends its turn; user or caller may continue | Same |
| Isolation of the solver | No network or project files; static policy (imports, no I/O or introspection); rlimits | Permission modes and an optional sandbox; network and filesystem allowed unless restricted | OS sandbox (Landlock/seccomp on Linux, Seatbelt on macOS) with approval modes |
| Ground-truth exposure | Truth never inside the sandbox; forcing / initial / boundary data only as black-box closures | Must be enforced by the caller (workspace contents, network) | Must be enforced by the caller |
| Budgets | Solver calls, model calls, wall and CPU per arm, 90 s and 16 GB per solve, input-size cap | Turns, tokens, and time are set by the caller | Same |
| Reproducibility | Every request, response, proposal, solve, and gate decision logged; pinned cores; same deadline for both arms | Transcript (stream-json) can be logged; tool use is not fixed in advance | Session logs |

## What this means for comparing results

1. **Coding-agent harnesses give the model far more feedback per hour.** It can run quick experiments, print
   intermediate values, and test on sub-problems. Our harness allows one gated proposal per round, so an agent like
   Claude Code should be expected to do better on the same budget.
2. **Our harness separates selection from evaluation.** Validation is used only by the gate, and the sealed test
   only after freezing. A coding agent that can query validation as often as it likes can overfit it, so its
   comparison needs either rate-limited validation access or reliance on the sealed test.
3. **RSI is defined only in our harness.** "Workflow" and "memory" are explicit objects there, inherited on
   acceptance. Claude Code and Codex have no such object, so the comparison is between whole harnesses (ours,
   single vs RSI, against a coding agent), not a third RSI arm.

## Claude Code arm (run October 17–18, 2026)

Results: `runs/code-agent-results-pdebench2h-20261017/claude_code_comparison.md`. Driver: `scripts/run_claude_code_agent.py`.

### Original plan

It was approved and run as planned:

- **Agent:** Claude Code CLI 2.1.288 in headless mode (`-p`, stream-json transcript) with the same Bedrock model as
  our arms (Opus 5.5; Sonnet 4.5 as a second run).
- **Budget:** a 2 h wall-clock deadline, re-prompted with `--continue` if it stops early.
- **Sandbox:** bubblewrap with network access, only a workspace that holds the public case views, noisy data, the
  equation library, and the solver contract. Fresh empty HOME; no project files, references, or home directory.
  `WebSearch` and `WebFetch` disabled.
- **Evaluation tool:** a host-side service that runs `solver.py` on training cases (diagnostics with labels) or
  validation cases (no labels) through our GPU sandbox. Its solves are pinned to a dedicated core; every request
  is logged.
- **Final score:** `solver.py` at the deadline, scored by the same evaluator on all 32 public cases and the 17
  sealed test cases. Tokens and cost are logged.
