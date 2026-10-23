#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec "$PDE_RSI_ENV_PREFIX/bin/python" "$project_root/scripts/push_hf.py" "$@"
