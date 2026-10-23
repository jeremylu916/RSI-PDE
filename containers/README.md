# RSI-PDEbench v1 containers

| File | Contents |
| --- | --- |
| `environment-lock.yml` | the harness and evaluator environment, pinned to the versions of `envs/pde-rsi` (Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3) |
| `gpu-solver-requirements.txt` | the solver-sandbox interpreter (Python 3.10.19, NumPy 2.2.6, SciPy 1.15.3, PyTorch 2.9.1+cu128) |
| `rsi-pde-bench.def` | the Apptainer recipe |
| `Dockerfile` | the same image for Docker (not built here) |

## Apptainer image (built and tested 2026-10-23)

```bash
apptainer build --fakeroot containers/build/rsi-pde-bench.sif containers/rsi-pde-bench.def   # about 4.6 GB
apptainer exec --nv --bind $PWD:/bench --pwd /bench containers/build/rsi-pde-bench.sif python ...
```

**What works inside the image** (checked):

- the pinned environment;
- `pde_rsi` imports;
- `PDE_RSI_PROJECT_ROOT=/bench`;
- the GPU interpreter at `/opt/gpu-env`;
- generating every bench split.

**What does not work: the bubblewrap solver sandbox.** On this cluster's Apptainer it cannot run nested in the
image. `bwrap` fails with `Can't bind mount /oldroot/usr on /newroot/usr: Invalid argument`. This happens with
`--userns`, `--fakeroot`, `--writable-tmpfs`, and with an unpacked sandbox-directory image.

Solves, and therefore runs and scoring, must use a host with bubblewrap 0.6+ and unprivileged user namespaces, as
on the Gautschi GPU nodes, with the same pinned versions. A Docker host may allow nesting with
`--security-opt seccomp=unconfined --cap-add SYS_ADMIN`; this has not been tested.

The hidden test (`evaluator_private/`) is never part of an image; the Dockerfile deletes it from the build
context.
