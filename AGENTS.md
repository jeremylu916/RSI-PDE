# Project instructions

- Current phase: the user explicitly resumed single-agent and RSI implementation
  and L2 evaluation on October 14, 2026. Build a fixed-foundation-model single-session
  agent and a recursive improvement-policy arm with equal declared budgets. Preserve
  historical runs, the independent evaluator, and the public/final information boundary.
  Use only the model/backend selected by the user. Since October 14, 2026 the agents
  write solver code (`code-agents`, docs/code-agents.md); generated code runs only
  through the bubblewrap sandbox in `codegen.py`, never in the harness process.
- Create all project files, environments, package/model caches, temporary files,
  and experiment outputs under `/depot/lin491/data/Binghang_project/PDE-RSI`.
  Source `scripts/env.sh` before installing; use `scripts/run.sh` and `scripts/test.sh`.
  Never install into the home directory or change global shell/conda configuration.
- Work within this project. Agent experiments must use supplied task files and local tools;
  do not search the web or other project directories for solutions.
- Prioritize classical Python solvers. Keep the foundation model frozen when an LLM is added.
- Treat `evaluator_private/` and final-test outputs as evaluator-only. Never use them for
  candidate creation, selection, memory updates, or improvement-policy training.
- Preserve task IDs, provenance, generation lineage, rejected proposals, and budget accounting.
- Do not change evaluators or tolerances as an agent improvement. Version evaluator changes
  separately and invalidate comparisons made with different evaluator versions.
- Do not claim recursive improvement from a better solver alone. Use equal-budget
  continuation comparisons with the same starting solver and memory.
- Do not submit cluster jobs or call paid model APIs unless requested.
- No figure, subplot, panel, or exported-table titles. Use captions outside figures.
