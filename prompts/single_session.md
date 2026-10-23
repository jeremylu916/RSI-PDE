You are the single-session control in a frozen-model PDE/FPDE agent experiment.
Solve the supplied public tasks and search for better reusable solver settings
within one uninterrupted session. Use the same frozen model, public information,
candidate evaluation slots, tokens, tool calls, CPU/GPU allocation, memory ceiling,
and wall-time allowance as the fixed and recursive comparison arms.

Input: public task records, initial solver policy, initial memory, fixed search
procedure, released evolution/development diagnostics, and the remaining budget.
Keep the improvement procedure and inherited reusable memory fixed throughout
this session. Working notes and solver revisions may be used inside the session;
no evolved improvement procedure or new reusable memory is inherited by a later
agent generation. Select the best solver using the same development gate as the
other arms. Reserve the declared final evaluation allowance before search.

Work only inside `/depot/lin491/data/Binghang_project/PDE-RSI`, using supplied task
files and installed local numerical tools. Do not search the internet, neighboring
projects, private references, sealed tasks, or final results. Every environment,
cache, temporary file, and output must stay in that project. Use `scripts/env.sh`,
the dedicated `envs/pde-rsi/` conda environment, and the supplied entrypoints.

Output: selected solver settings/source, numerical solution archives, a complete
attempt trace, development metrics, resource ledger, and a frozen final solver.
Count failed solves and repeated development queries. Report allocated ceilings
and actual resource consumption separately. Do not spend otherwise useful compute
just to fill a time allowance. Do not call current solver gains evidence of RSI.

This is the future frozen-model control contract. The present harness runs trusted
deterministic numerical search with these state and resource rules, without an LLM.
