# Docker image of RSI-PDE Bench v1; see rsi-pde-bench.def (Apptainer, used on the cluster) for notes.
#   docker build -f containers/Dockerfile -t rsi-pde-bench .
FROM mambaorg/micromamba:2.0.5
USER root
RUN apt-get update && apt-get install -y --no-install-recommends bubblewrap ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY containers/environment-lock.yml /opt/env.yml
COPY containers/gpu-solver-requirements.txt /opt/gpu-solver-requirements.txt
RUN micromamba create -y -p /opt/pde-rsi -f /opt/env.yml \
    && micromamba create -y -p /opt/gpu-env -c conda-forge python=3.10.19 pip && micromamba clean -a -y \
    && /opt/gpu-env/bin/pip install --no-cache-dir -r /opt/gpu-solver-requirements.txt
ENV PATH=/opt/pde-rsi/bin:$PATH PDE_RSI_ENV_PREFIX=/opt/pde-rsi PDE_RSI_PROJECT_ROOT=/bench \
    PDE_RSI_GPU_ENV=/opt/gpu-env
WORKDIR /bench
COPY . /bench
RUN rm -rf /bench/evaluator_private /bench/.runtime /bench/envs /bench/runs \
    && /opt/pde-rsi/bin/pip install --no-deps -e /bench
