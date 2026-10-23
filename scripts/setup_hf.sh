#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
if [[ ! -x "$PDE_RSI_ENV_PREFIX/bin/python" ]]; then
    printf 'Create the project environment first: bash scripts/setup_env.sh\n' >&2
    exit 1
fi
"$PDE_RSI_ENV_PREFIX/bin/python" -m pip install \
    --constraint "$project_root/configs/core-constraints.txt" \
    --requirement "$project_root/requirements-huggingface.in"
"$PDE_RSI_ENV_PREFIX/bin/python" -m pip freeze > "$project_root/configs/huggingface-requirements-lock.txt"
"$PDE_RSI_ENV_PREFIX/bin/python" -m pde_rsi doctor
