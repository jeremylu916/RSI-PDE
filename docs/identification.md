# PDE identification protocol

The agent is not told which equation generated a case. It receives:

- a **library of candidate equations**;
- **sparse noisy data** from one hidden true task;
- the source term and initial data, as black boxes.

It must choose the equation class, estimate its coefficients, and return an accurate solution over
the whole domain. Evaluation is unchanged: relative L2 against the true task's exact solution on the
fine grid (evaluator v5). The chosen model is scored separately and does not enter the acceptance
gate.

Code: [`identification.py`](../src/pde_rsi/identification.py) (cases, data, scoring, protocol) and
[`classical_programs/identify.py`](../src/pde_rsi/classical_programs/identify.py) (classical
baseline). Run it with `--protocol identification`:

```bash
bash scripts/run.sh code-agents --protocol identification --model-config configs/agent_models/claude_code.json \
  --start classical --rounds 6 --probe-rounds 2 --attempts-per-round 3 --wall-seconds 7200 --cpu-seconds 5400 \
  --output runs/my-identification
```

## Cases and data

Each public split has 16 cases, two per true family. The hidden truths are:

- 1D benchmark tasks (variants v1 and v2): heat, convection–diffusion, Burgers, Caputo, and
  multi-term Caputo;
- group FPDE tasks: the Caputo FPDE and the two variable-order problems.

Case IDs (`case-t00`…, `case-v00`…) follow a hash order, so neither the ID nor the position
reveals the family.

For each case the agent gets:

- the domain (L, T) and zero boundary values;
- 256 samples of the exact solution at random interior (x, t), with Gaussian noise of 1% of
  their RMS, and that noise level;
- `forcing(x, t)` (t > 0) and `initial(x)`;
- the limits and the evaluator grid size.

Each solve may take up to 30 s, because identification includes fitting.

## Library (equation classes shown to the agent)

| Class | Equation | Parameters | True families |
| --- | --- | --- | --- |
| `heat` | u_t − D u_xx = f | D | heat |
| `convection_diffusion` | u_t + v u_x − D u_xx = f | D, v | convection–diffusion |
| `burgers` | u_t + u u_x − D u_xx = f | D | Burgers |
| `caputo` | D_t^α u − D u_xx = f | D, α | 1D Caputo, Caputo FPDE (D = 1) |
| `multi_term` | w1 D_t^α1 u + D_t^α2 u − D u_xx = f | D, α1, α2, w1 | multi-term Caputo (w1 = 0.4) |
| `variable_order_t` | D_t^{α(t)} u − D u_xx = f, α = c sin²t | D, c | variable-order α(t) |
| `variable_order_xt` | D_t^{α(x,t)} u − D u_xx = f, α = c sin(x) t² | D, c | variable-order α(x,t) |

## Program contract

There is one program, `identify`, with `solve(problem)`. It returns `{"x", "t", "u", "model":
{"family": class, "parameters": {...}}}`. The problem view contains no family, coefficients, or
profile parameters. The truth lives only inside the forcing and initial-data closures, so a program
that reads `problem.diffusivity` fails.

## Feedback and gate

- **Training feedback.** Training-case diagnostics include the true class, the true parameters,
  and whether the chosen class was correct. This is labeled training data.
- **Validation feedback.** Validation diagnostics show only errors and the agent's own model.
- **Gate messages.** Gate reasons replace hidden task IDs with case IDs and family names with
  "a hidden equation class".
- **Gate.** The acceptance gate, the self-test, the budgets, and the continuation test are the
  same as in the code-writing protocol.
- **Reporting.** The report adds `class_accuracy_evolution` and `class_accuracy_development` for
  each arm.

## Classical baseline

The baseline fits each class on a coarse grid (32 × 48) with Nelder–Mead, picks a class by BIC, and
re-solves the winner on a 128 × 160 grid. It takes about 1.7 s per case. On the first 8 validation
cases it chose the right class 6 times, with D within about 1% and α within about 3%. Its two
mistakes were variable order α(t) taken for α(x,t), and Caputo taken for multi-term.
