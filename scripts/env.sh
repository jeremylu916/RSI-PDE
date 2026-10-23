#!/usr/bin/env bash
# Source before installation or execution. Never change HOME or global shell config.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "$project_root" != /depot/lin491/data/Binghang_project/PDE-RSI ]]; then
    printf 'PDE-RSI must run from /depot/lin491/data/Binghang_project/PDE-RSI\n' >&2
    return 1 2>/dev/null || exit 1
fi
export PDE_RSI_ROOT="$project_root"
export PDE_RSI_ENV_PREFIX="$project_root/envs/pde-rsi"
export PATH="$PDE_RSI_ENV_PREFIX/bin:$PATH"
export CONDARC="$project_root/configs/condarc.yml"
export CONDA_ENVS_PATH="$project_root/envs"
export CONDA_PKGS_DIRS="$project_root/.runtime/conda/pkgs"
export CONDA_REGISTER_ENVS=false
export CONDA_NO_PLUGINS=true
export XDG_CACHE_HOME="$project_root/.runtime/cache"
export XDG_CONFIG_HOME="$project_root/.runtime/config"
export XDG_DATA_HOME="$project_root/.runtime/data"
export XDG_STATE_HOME="$project_root/.runtime/state"
export TMPDIR="$project_root/.runtime/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CONFIG_FILE=/dev/null
export PIP_REQUIRE_VIRTUALENV=false
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONUSERBASE PYTHONPYCACHEPREFIX
export PYTHONPATH="$project_root/src"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export UV_PYTHON_INSTALL_DIR="$project_root/.runtime/uv-python"
export PLAYWRIGHT_BROWSERS_PATH="$XDG_CACHE_HOME/ms-playwright"
export JUPYTER_CONFIG_DIR="$XDG_CONFIG_HOME/jupyter"
export JUPYTER_DATA_DIR="$XDG_DATA_HOME/jupyter"
export JUPYTER_RUNTIME_DIR="$project_root/.runtime/jupyter"
export IPYTHONDIR="$XDG_DATA_HOME/ipython"
export WANDB_DIR="$project_root/runs/wandb"
export WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb"
export WANDB_CONFIG_DIR="$XDG_CONFIG_HOME/wandb"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
# Check real locations before creating anything, including pre-existing symlinks.
for project_variable in PDE_RSI_ENV_PREFIX CONDARC CONDA_ENVS_PATH CONDA_PKGS_DIRS TMPDIR \
    XDG_CACHE_HOME XDG_CONFIG_HOME XDG_DATA_HOME XDG_STATE_HOME PIP_CACHE_DIR \
    HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TORCH_HOME TRITON_CACHE_DIR CUDA_CACHE_PATH \
    NUMBA_CACHE_DIR MPLCONFIGDIR UV_CACHE_DIR UV_PYTHON_INSTALL_DIR PLAYWRIGHT_BROWSERS_PATH \
    JUPYTER_CONFIG_DIR JUPYTER_DATA_DIR JUPYTER_RUNTIME_DIR IPYTHONDIR \
    WANDB_DIR WANDB_CACHE_DIR WANDB_CONFIG_DIR; do
    project_dir="${!project_variable}"
    case "$(realpath -m -- "$project_dir")" in
        "$project_root"/*) ;;
        *) printf 'Storage path escapes project: %s\n' "$project_dir" >&2; return 1 2>/dev/null || exit 1 ;;
    esac
done
mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" "$TMPDIR" \
    "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME"
cd "$project_root"
