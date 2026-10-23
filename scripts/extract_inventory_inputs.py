"""Extract the DiffusionPDE test-set inputs used by the inventory suite into data/inventory/inputs.npz.

Instances 0, 1, 2 of each dataset become the training, validation, and sealed cases; instance 3 the fresh
sealed split inventory_test2. Only inputs are stored
(initial states, coefficient and forcing fields); solutions are recomputed by scripts/build_inventory_references.py.
Orientation: stored 2-D arrays have rows = y, columns = x. The Navier-Stokes data store axis 0 = x (FNO
generator, meshgrid 'ij'), so w0 is transposed; checked against the dataset's own trajectory (agreement 4.6e-4
at t = 1, the dataset's time-stepping error). Burgers 'output' is [instance, time, x] and output[:, 0] = input.
Usage (after sourcing scripts/env.sh):
    python scripts/extract_inventory_inputs.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.io

from pde_rsi.inventory import DATA_DIR, INPUTS
from pde_rsi.storage import PROJECT_ROOT

SOURCE = PROJECT_ROOT/"external/DiffusionPDE-data/testing"
INSTANCES = (0, 1, 2, 3)  # 3: the fresh sealed split inventory_test2 (2026-10-20)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, nargs="+", default=list(INSTANCES))
    parser.add_argument("--output", type=Path, default=INPUTS, help="e.g. data/expanded/inventory_inputs.npz")
    args = parser.parse_args()
    output = PROJECT_ROOT/args.output if not args.output.is_absolute() else args.output
    arrays = {}
    burgers = scipy.io.loadmat(SOURCE/"burgers.mat", variable_names=["input"])
    darcy = scipy.io.loadmat(SOURCE/"darcy.mat", variable_names=["thresh_a_data"])
    poisson = scipy.io.loadmat(SOURCE/"poisson.mat", variable_names=["f_data"])
    helmholtz = scipy.io.loadmat(SOURCE/"helmholtz.mat", variable_names=["f_data"])
    ns = scipy.io.loadmat(SOURCE/"ns-nonbounded.mat", variable_names=["a"])
    for i in args.instances:
        arrays[f"burgers_random_{i}__u0"] = burgers["input"][i].astype(float)
        arrays[f"darcy_{i}__a"] = darcy["thresh_a_data"][i].astype(float)
        arrays[f"poisson_{i}__f"] = poisson["f_data"][i].astype(float)
        arrays[f"helmholtz_{i}__f"] = helmholtz["f_data"][i].astype(float)
        arrays[f"ns_{i}__w0"] = ns["a"][i].astype(float).T
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    manifest = {"source": "DiffusionPDE test set (https://github.com/generativepde/DiffusionPDE, testing.zip)",
                "instances": list(args.instances), "arrays": {k: list(v.shape) for k, v in arrays.items()},
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_name(output.stem.replace("inputs", "inputs_manifest")+".json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
