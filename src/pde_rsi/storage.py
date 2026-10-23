"""Project-owned storage and interpreter checks, including symlink resolution."""

import os
from pathlib import Path
import sys
import tempfile

# PDE_RSI_PROJECT_ROOT relocates the project (e.g. /bench in the RSI-PDE Bench container)
PROJECT_ROOT = Path(os.environ.get("PDE_RSI_PROJECT_ROOT", "/depot/lin491/data/Binghang_project/PDE-RSI"))
ENV_PREFIX = PROJECT_ROOT / "envs/pde-rsi"


def project_path(path):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError(f"Path must remain inside {PROJECT_ROOT}: {path}")
    return resolved


def storage_environment():
    runtime = PROJECT_ROOT / ".runtime"
    cache, config, data = runtime / "cache", runtime / "config", runtime / "data"
    return {
        "PDE_RSI_ROOT": str(PROJECT_ROOT), "PDE_RSI_ENV_PREFIX": str(ENV_PREFIX),
        "CONDARC": str(PROJECT_ROOT / "configs/condarc.yml"),
        "CONDA_ENVS_PATH": str(PROJECT_ROOT / "envs"),
        "CONDA_PKGS_DIRS": str(runtime / "conda/pkgs"),
        "XDG_CACHE_HOME": str(cache), "XDG_CONFIG_HOME": str(config),
        "XDG_DATA_HOME": str(data), "XDG_STATE_HOME": str(runtime / "state"),
        "TMPDIR": str(runtime / "tmp"), "TMP": str(runtime / "tmp"), "TEMP": str(runtime / "tmp"),
        "PIP_CACHE_DIR": str(cache / "pip"), "HF_HOME": str(cache / "huggingface"),
        "HF_HUB_CACHE": str(cache / "huggingface/hub"),
        "HUGGINGFACE_HUB_CACHE": str(cache / "huggingface/hub"),
        "HF_DATASETS_CACHE": str(cache / "huggingface/datasets"),
        "TORCH_HOME": str(cache / "torch"), "TRITON_CACHE_DIR": str(cache / "triton"),
        "CUDA_CACHE_PATH": str(cache / "cuda"), "NUMBA_CACHE_DIR": str(cache / "numba"),
        "MPLCONFIGDIR": str(cache / "matplotlib"), "UV_CACHE_DIR": str(cache / "uv"),
        "UV_PYTHON_INSTALL_DIR": str(runtime / "uv-python"),
        "PLAYWRIGHT_BROWSERS_PATH": str(cache / "ms-playwright"),
        "JUPYTER_CONFIG_DIR": str(config / "jupyter"),
        "JUPYTER_DATA_DIR": str(data / "jupyter"),
        "JUPYTER_RUNTIME_DIR": str(runtime / "jupyter"), "IPYTHONDIR": str(data / "ipython"),
        "WANDB_DIR": str(PROJECT_ROOT / "runs/wandb"),
        "WANDB_CACHE_DIR": str(cache / "wandb"), "WANDB_CONFIG_DIR": str(config / "wandb"),
    }


def configure_storage():
    # Also applies to the installed console entrypoint and imported worker modules.
    # HOME is intentionally untouched, and no user-wide configuration is written.
    paths = storage_environment()
    for value in paths.values():
        project_path(value)
    os.environ.update(paths)
    os.environ.update({"CONDA_REGISTER_ENVS": "false", "CONDA_NO_PLUGINS": "true",
                       "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_CONFIG_FILE": os.devnull,
                       "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                       "PYTHONPATH": str(PROJECT_ROOT / "src")})
    for name in ("PYTHONHOME", "PYTHONUSERBASE", "PYTHONPYCACHEPREFIX"):
        os.environ.pop(name, None)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    for name in ("TMPDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"):
        Path(paths[name]).mkdir(parents=True, exist_ok=True)
    sys.dont_write_bytecode = True
    sys.pycache_prefix = None
    tempfile.tempdir = paths["TMPDIR"]


def doctor():
    import numpy
    import scipy
    paths = storage_environment()
    for name, expected in paths.items():
        if os.environ.get(name) != expected:
            raise ValueError(f"Unexpected {name}; run through scripts/run.sh")
        project_path(expected)
    if Path(sys.prefix).resolve() != ENV_PREFIX.resolve():
        raise ValueError(f"Use the project conda interpreter: {ENV_PREFIX}/bin/python")
    if sys.flags.no_user_site != 1:
        raise ValueError("Disable user-site packages with PYTHONNOUSERSITE=1 before Python starts")
    if not (ENV_PREFIX / "conda-meta/history").is_file():
        raise ValueError("The project interpreter is not a conda environment")
    return {"storage_ok": True, "project_root": str(PROJECT_ROOT),
            "python_executable": sys.executable, "python_prefix": sys.prefix,
            "python_version": sys.version, "numpy": numpy.__version__, "scipy": scipy.__version__,
            "temporary_directory": tempfile.gettempdir(), "storage_paths": paths,
            "register_home_conda_environment": False,
            "bytecode_writes": not sys.dont_write_bytecode,
            "user_site_enabled": not bool(sys.flags.no_user_site)}
