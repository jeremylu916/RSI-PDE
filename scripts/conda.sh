#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
conda_executable="${CONDA_EXE:-$(command -v conda || true)}"
conda_python="${CONDA_PYTHON_EXE:-$(dirname -- "$conda_executable")/python}"
if [[ ! -x "$conda_python" ]]; then
    printf 'Load the cluster conda module before using scripts/conda.sh\n' >&2
    exit 1
fi
exec "$conda_python" -B "$project_root/scripts/conda_local.py" "$@"
