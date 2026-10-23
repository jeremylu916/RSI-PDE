# Student protocol: identify the PDE, choose a method, write the solver

The agent works like a student who receives:

- measured data;
- a list of possible PDE families;
- two solution methods, numerical and PINN.

For each case it decides which PDE generated the data, which method to use, and writes the solver
code from scratch. The single-session and RSI harness, the self-test, the acceptance gate, and the
evaluator are the same as in the other protocols.

Run it on a GPU node:

```bash
bash scripts/run.sh code-agents --protocol student --start scratch --model-config configs/agent_models/gpt55_code.json \
  --rounds 6 --probe-rounds 2 --attempts-per-round 3 --wall-seconds 14400 --cpu-seconds 10800 --output runs/my-student
```

## What the agent gets

- **Data, library, and black boxes.** Everything from [identification](identification.md) v2: the
  7-class equation library, 256 noisy samples per case, and `forcing(x, t)`, `initial(x)` as black
  boxes. There is one case per true family per split (8 training, 8 validation). Domains and
  coefficients are drawn independently for every case.
- **Method descriptions only.** The contract describes numerical methods (FD, FV, spectral, L1-type
  Caputo schemes) and PINNs (PyTorch, GPU, residual plus data plus IC/BC losses, with a quadrature for
  Caputo history). No implementations are provided, and DeepONet is not offered.
- **Scratch start.** The starting program is a stub. There is no classical start.

## What it returns

`{"x", "t", "u", "model": {"family", "parameters", "method": "numerical" | "pinn"}}`.

## Limits

| Item | Value |
| --- | --- |
| Wall and CPU time per solve | 90 s, including about 3 s of PyTorch import |
| Peak host memory | 16 GB |
| Self-test | crashes are checked on 3 training cases per attempt |

## Scoring

The evaluator is unchanged: relative L2 against the true solution on the fine grid (512 × 500), with
IC/BC and resource checks. The report also gives:

- class accuracy on the training and validation splits;
- relative parameter error;
- the fraction of cases solved by PINN.

## Sandbox for GPU solves

The bubblewrap sandbox (`bwrap-v5-shm`) gives GPU solves:

- the NVIDIA device files and read-only `/sys`;
- `CUDA_VISIBLE_DEVICES`;
- PyTorch, pre-imported.

There is no address-space cap for GPU solves, because CUDA reserves large virtual ranges; memory is
checked by peak resident size instead.

The static policy allows `torch`, `torch.nn`, `torch.nn.functional`, `torch.optim`, and
`torch.autograd`. It rejects `hub`, `jit`, `compile`, `save`/`load`, `cpp_extension`,
`multiprocessing`, `distributed`, `onnx`, `export`, `load_library`, and `package`. CPU-only solves
cannot import PyTorch.

Every solve now has a private `/dev/shm` and a 256 MB file-size cap. Before this change, any
matrix–matrix product inside the sandbox crashed with an OpenMP shared-memory error, in every protocol.

## Environment

GPU solves use a separate interpreter, configured in `configs/gpu_env.json`. It currently points to
`/home/lu895/.conda/envs/2025.06-py313/muon`:

| Component | Version |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.9.1 with CUDA 12.8 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |

The sandbox mounts it read-only at `/opt/gpu-env`, so the home directory stays invisible.
`gpu_sandbox_self_test()` runs before every student comparison. It checks that CUDA works, that no
project files, home directory, or network are visible, and records the versions in the manifest.
CPU solves and the harness keep using `envs/pde-rsi`.

This is an exception to the project's "all environments inside the project" rule, approved by the
user on October 16, 2026. Installing CUDA PyTorch into `envs/pde-rsi` stalled on depot NFS: pip was
blocked in NFS remote calls at about 60 KB/s, after writing 10.8 GB because the download, unpack, and
install all crossed NFS. The partial install was removed.

## Results: full comparison (October 17, 2026)

Five models, scratch start, 6 rounds, up to 3 attempts per round, 2 continuation rounds, on h010 with
one H100. The report is `runs/code-agent-results-student-20261017/` (per-equation table,
score-vs-round plot, CSVs, and protocol checks); per-case folders are in `tasks/student/`. All 160
saved solutions reproduce their metrics, and both sandboxes were isolated.

Each cell gives the mean relative L2 over the 16 public cases, passes out of 16, and validation cases with the
correct equation out of 8. The last figure is the median coefficient error on correctly identified validation cases
(the median over cases of each case's median parameter error). The single run is one model request whose
solver is frozen as returned (redone on 2026-10-20). The Opus single run used Opus 5.5's 128k output maximum,
because at the 32k cap its answer ended before any code. The earlier iterated single-session arm is archived in
`runs/archive-fixed-workflow-20261020/`.

| Model | Single run | RSI |
| --- | --- | --- |
| Opus 5.5 | 1.0e-3 · 14/16 · 8/8 · 0.8% | **5.1e-4 · 16/16 · 8/8 · 0.8%** |
| GPT-6.1 | 2.8e-3 · 13/16 · 7/8 · 1.1% | **8.9e-4 · 15/16 · 8/8 · 0.8%** |
| GPT-5.5 | 1.3e-2 · **10/16** · 6/8 · 1.2% | **1.9e-3** · 8/16 · 7/8 · 1.1% |
| GPT-OSS-120B | fails (JSON-valid, never executes) · 0/16 · 0/8 | **1.5e-2 · 0/16 · 0/8** |
| Sonnet 4.5 | fails · 0/16 · 0/8 | **0.32 · 2/16 · 2/8** |

RSI is better than the single run in 4 of the 5 models. GPT-5.5's single run passes more cases (10 against 8), but
it has a 7× larger mean error. Continuation advantages are all zero apart from +1e-5 for GPT-6.1.
**No final program, and none of the 69 candidate programs, ever used PyTorch.** Every case was solved
numerically: at the 2e-3 accuracy target and 90 s per solve on smooth 1D problems, a numerical method
is the better choice, so the PINN option was never exercised.
