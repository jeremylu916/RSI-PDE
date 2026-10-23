#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
conda_executable="${CONDA_EXE:-$(command -v conda || true)}"
if [[ ! -x "$conda_executable" ]]; then
    printf 'Load the cluster conda module, then rerun scripts/setup_env.sh\n' >&2
    exit 1
fi
if [[ ! -x "$PDE_RSI_ENV_PREFIX/bin/python" ]]; then
    if [[ -f /apps/external/anaconda/2025.06/pkgs/python-3.13.5-h4612cfd_100_cp313.conda ]]; then
        # Install only the dependency closure, rather than cloning the large base.
        # Package archives are read from the cluster; all copies/extractions go to depot.
        bash "$project_root/scripts/conda.sh" create --prefix "$PDE_RSI_ENV_PREFIX" --offline \
            --file "$project_root/configs/conda-cluster-explicit.txt" --yes
    else
        bash "$project_root/scripts/conda.sh" env create --prefix "$PDE_RSI_ENV_PREFIX" \
            --file "$project_root/environment.yml" --solver classic --yes
    fi
fi
"$PDE_RSI_ENV_PREFIX/bin/python" -m pip install --no-deps --no-build-isolation \
    --editable "$project_root"
lock_path="$(realpath -m -- "$project_root/configs/conda-explicit.txt")"
case "$lock_path" in
    "$project_root"/*) ;;
    *) printf 'Environment lock path escapes project: %s\n' "$lock_path" >&2; exit 1 ;;
esac
bash "$project_root/scripts/conda.sh" list --prefix "$PDE_RSI_ENV_PREFIX" --explicit --md5 > "$lock_path"
"$PDE_RSI_ENV_PREFIX/bin/python" -m pde_rsi doctor
