#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
if [[ ! -x "$PDE_RSI_ENV_PREFIX/bin/python" ]]; then
    printf 'Create the depot environment first: bash scripts/setup_env.sh\n' >&2
    exit 1
fi
exec "$PDE_RSI_ENV_PREFIX/bin/python" -m pde_rsi "$@"
