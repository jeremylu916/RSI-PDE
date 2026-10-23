# Recursive Self Improvement for PDE and FPDE Agents

Research project plan supplied by the user, with an implementation roadmap appended.
The foundation model stays frozen initially so agent-system changes can be studied.

## Current working scope — October 14, 2026

The user resumed single-agent and RSI implementation and requested L2 evaluation
on the collected problems. The current model-backed experiment uses the 48 public
Hugging Face problems across six families, trusted classical Python solvers, and
fixed foundation models through native Amazon Bedrock Converse.

1. Run a single-session model agent with a fixed improvement workflow. It proposes
   bounded spatial/time settings, receives independent diagnostics and candidate
   decisions, and retains numerical revisions within the session.
2. Run a recursive arm with the same starting settings, model, public information,
   and declared resources. Accepted numerical candidates may also inherit an
   updated improvement workflow and a persistent memory lesson.
3. Freeze each selected agent and calculate absolute/relative combined and
   final-time L2 for all 48 public cases. These remain development/selection
   measurements; the reserved final cases are excluded.
4. Compare future continuations from the same frozen RSI solver and memory,
   changing only the initial versus evolved workflow under equal allowances.

The implemented model configurations are `openai.gpt-oss-120b-1:0` and
`us.anthropic.claude-sonnet-4-5-20250929-v1:0`; model identity remains fixed within
each comparison. Models choose settings for existing solvers, rather than
executing tools or writing arbitrary solver programs. The historical deterministic
scaffold and its evidence are preserved separately.

Problem collection, published-benchmark imports, independently validated numerical
references, and the user's FPDE examples continue as the benchmark roadmap.
Arbitrary generated code and PINN/FPINN are future extensions. See the
[problem catalog](problem-catalog.md) for equation locations and the
[agent experiment protocol](agent-experiments.md) for current inputs, outputs,
budgets, gates, and L2 definitions.

## Research objective

Build an AI agent that recursively improves its ability to solve PDEs and fractional
PDEs. The central question is whether successive agent generations improve the
process that produces future improvements. A benchmark provides numerical tasks,
feedback, and independent evidence of progress.

## Operational definition

Each agent version contains a solver policy, improvement policy, and persistent
memory. The improvement policy can propose changes to all three components,
including its own search and experiment procedures. After validation, the accepted
improvement policy drives the next generation.

| Level | What changes | Evidence required |
| --- | --- | --- |
| Solver improvement | Numerical solver, MATLAB program, or PINN program | Better accuracy or efficiency on a task |
| Agent improvement | Reusable skills, memory, or solving workflow | Better performance on new tasks |
| Recursive improvement | Procedure that creates subsequent agent versions | Better future improvement under an equal budget |

The supplied plan names existing PDE/PINN agents [7,8,10] as baselines and AIDE
squared [9] as a harness-RSI methodological comparator. Full bibliographic
identifiers are still needed; no unsupported comparisons are made in v0.1.

## Research questions

- Does an evolved improvement policy produce stronger subsequent agent versions
  than the initial or fixed policy under the same compute budget?
- Do improvements transfer to new parameters, boundary conditions, time horizons,
  and local or nonlocal operator families?
- Which changes sustain progress, and how effectively do independent numerical
  checks detect regressions or misleading proxy gains?

## Milestone 1: benchmark construction

Unify selected benchmark problems and group FPDE examples under one task/evaluation
interface. The planned pilot contains roughly 60–80 cases across 6–8 families.
Expand to approximately 200–300 cases after evaluator validation and the first
RSI experiment. These counts are planning estimates.

| Source | Role |
| --- | --- |
| CFDLLMBench [1] | Python solver tasks first; OpenFOAM in a separate transfer track |
| PDEAgent-Bench [2] | Reuse specification and staged execution/accuracy/efficiency evaluation ideas |
| PINNACLE [3], PINNacle [6] | Curate PDE cases and training methods for a PINN/FPINN track |
| FEABench [5] | Optional COMSOL workflow after the open numerical pilot |
| Group FPDE examples | Time-fractional, diffusion-wave, multi-term, and selected space-fractional cases |

The plan distinguishes arXiv 2503.06680, FEA-Bench for repository feature
implementation [4], from 2504.06260, the finite-element physics benchmark [5].
It likewise distinguishes 2404.07662, PINNACLE sampling [3], from 2306.08827,
the comprehensive PINNacle benchmark [6]. These identifiers are transcribed from
the supplied plan and await independent verification. See the source registry.

### Coverage

Begin with forward Poisson, heat, convection–diffusion, Burgers, Caputo subdiffusion,
and multi-term fractional examples. Include smooth and difficult initial-time
behavior. Use matched local/fractional cases to study spatial, temporal, and
diagnostic transfer. Add inverse problems only after forward evaluation is stable.

Every task specifies the equation, domain, coefficients, forcing, IC/BC, permitted
observations, requested outputs, tolerance, and compute limits. Fractional tasks
also specify the operator definition, order, history starting point, and applicable
exterior conditions. Generate references independently and check them before use.

### Splits and tracks

- Evolution tasks provide execution and diagnostic feedback.
- Development tasks select candidates; repeated queries count as selection.
- Sealed final tests contain new instances and reserved regimes/families. Group
  related templates and parameter variants before splitting.
- The primary track permits classical, neural, or hybrid solvers in a consistent
  Python environment. A separate PINN/FPINN track isolates learned-method gains.
- Preserve source IDs and permissions, deduplicate overlaps, and report by family.
  Published cases also support comparison with prior systems.

## Single run versus RSI

Hold model identity and available information fixed. Predeclare the same elapsed
time, hardware, model-token allowance, and tool-compute budget. Compare a single
uninterrupted agent session, a multi-generation agent with a fixed improvement
policy, and a multi-generation agent with accepted recursive policy updates.
Report allocated resources and actual consumption. Include controls for memory,
skills, solver revisions, policy adaptation, and evaluator query counts.

To establish RSI specifically, start future continuations from the same solver and
memory, vary only the improvement procedure, and compare future gains under an
equal budget. Use multiple seeds, paired uncertainty intervals, held-out
continuation cases, and one predeclared final evaluation. Report a negative or null
policy advantage when observed.

## Initial implementation and next work

The depot environment confines project files, caches, temporary files, and outputs
to `/depot/lin491/data/Binghang_project/PDE-RSI`. The historical v0.1/v0.2 scaffold
established manufactured tasks, classical solvers, independent checks, deterministic
policy comparisons, aggregate CPU accounting, per-call audits, and NPZ
solve/evaluate interfaces. Historical workflows remain in
[workflow](workflow.md) and `configs/pilot_comparison.json`.

The resumed model-backed implementation now includes:

- A one-call HTTP backend for local/hosted compatible endpoints and native Bedrock
  Converse, with project-private credential loading, structured JSON parsing,
  sanitized audits, actual provider token usage, and no retries or fallback models.
- Fixed-model single-session and recursive workflow/memory arms using the 48
  public uploaded dataset records, checked against the numerical benchmark.
- Three main improvement rounds with up to three model requests and 216 numerical
  solves per arm, equal 900-second wall and 600-second local CPU ceilings, one
  numerical-library thread, and zero local GPUs.
- Paired one-round continuations from identical solver and memory, with only the
  improvement workflow changed; each allows one model request and 120 solves.
- Absolute and relative combined/final L2 measurements on a fixed evaluator grid,
  per-case/per-family tables, public frozen solutions, lineage, gate decisions,
  source/data hashes, resource reports, and archived failures.

These agents select settings for trusted numerical routines. A full agent that
writes new solver code and evaluates it in isolation is still a separate extension.
Cloud inference compute is not locally metered; report actual tokens and local
resources alongside declared ceilings. Interpret current comparisons as public
selection evidence until independent final evaluation and repeated experiments
are available. The executable protocol is documented in
[agent experiments](agent-experiments.md).

The earlier numerical scaffold contains:

- 72 self-generated manufactured tasks, six families, grouped 24/24/24 splits;
  the current upload and model runs use only the 48 public records.
- Classical finite-difference, backward-Euler, IMEX, and graded Caputo-L1 solvers.
- Analytic reference verification, accuracy/IC/BC/resource checks, physical residual
  diagnostics, regression gates, and per-family reporting.
- Versioned solver/improvement policies and memory with rejected-proposal logs.
- Deterministic single/fixed/RSI comparisons and equal-call continuation probes.
- Explicit frozen final evaluation, source/task/snapshot hashes, prompt contracts,
  and local/Slurm entrypoints.

Next implementation steps, in order:

1. Harden evaluator trust: private account/service, isolated agent workspace,
   reproducible CPU/GPU accounting, and reference convergence certificates for
   problems without analytic solutions.
2. Verify source identities and reuse permissions, then import selected public
   benchmark cases and user-designated group FPDE examples with provenance and
   deduplication. Do not scan neighboring projects during agent episodes.
3. Add genuinely new template groups, parameter shifts, boundary conditions, and
   reserved families; independently validate reference solvers. Separate unseen
   instance, regime, and family transfer measurements.
4. Add held-out continuation sets, repeated seeds, compute-matched ablations,
   paired uncertainty estimates, a model-backed fixed-policy multi-generation
   control, and an accuracy-constrained efficiency gate.
5. Extend the connected model backend to an isolated generated-solver workflow,
   with staged public inputs, validated tool actions, and inference-cost reporting.
   Compare against a full-budget single-session code-writing agent before broader
   scientific claims about solver-generating RSI.
6. Expand to diffusion-wave orders (1,2), space-fractional operators with exterior
   conditions, and fast fractional-history methods.
7. Add the PINN/FPINN track, then inverse tasks, MATLAB adapters, and optional
   OpenFOAM/COMSOL transfer workflows after forward evaluation is stable.

Historical numerical-scaffold evidence and current model-backed evidence should
be reported separately. Imported literature-benchmark performance and broad
recursive self-improvement have not been established by this pilot.
