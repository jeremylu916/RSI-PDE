"""RSI-PDE Bench splits: bench-train, bench-val (public) and bench-test (hidden), drawn from the expanded generator.

Every template of a track appears PER_TEMPLATE[split] times per split, with numbers from expanded.py's samplers
(the same distribution as the 1001-case release), seeded by "rsi-pde-bench-v1:<split>". The hidden test seed also
contains a secret (evaluator_private/bench/secret.txt), so it cannot be regenerated from the public code; its
references live in evaluator_private/bench/references. Case ids are opaque ("case-bt07") and ordered by a hash,
and task ids carry no template name. Variant indices are chosen so that training and validation cases have the
release's nonzero Dirichlet data and boundary-layer amplitude (index 1 and 2 of the samplers).
"""

import hashlib
import math
import secrets as _secrets

from .. import expanded as X
from ..storage import PROJECT_ROOT
from .spec import BENCH_VERSION, PER_TEMPLATE

SECRET = PROJECT_ROOT/"evaluator_private/bench/secret.txt"
SPLIT_NAMES = {"bench-train": "train", "bench-val": "val", "bench-test": "test", "bench-calibration": "calibration"}
TASK_SPLIT = {"train": "evolution", "val": "development", "test": "test", "calibration": "calibration"}
PREFIX = {"train": "bt", "val": "bv", "test": "bh", "calibration": "bc"}
INDICES = {"train": (1,), "val": (2,), "test": (0, 1, 2, 3, 4), "calibration": tuple(range(10))}
# calibration: a public draw of the hardened templates only, to check them with the calibration probes; it is
# never shown to an entrant and never used as test data
HARDENED = {"tex": ("3b", "3c", "6", "9", "11"), "inventory": ("burgers_gauss", "helmholtz"), "ext": ()}
# DiffusionPDE test instances of the dataset-based inventory families: 0-3 earlier splits, 4-23 the 1001 release
INSTANCES = {"train": (100,), "val": (101,), "calibration": tuple(range(110, 120))}  # test: five secret draws
# from 200-999 per family


# Hardening (step 2 of docs/rsi-pde-bench.md): templates that the strongest calibration probes solved on every
# variant of the 1001-case release get harder numbers in the bench splits. The closed forms stay exact.
def harden_tex(problem, values, r):
    sign = math.copysign(1.0, values["v"]) if "v" in values else 1.0
    if problem == "3b":  # thinner boundary layer: |Pe| 50-600 (release 5-150)
        return {**values, "v": sign*X.log_uniform(r, 50.0, 600.0)*values["D"]}
    if problem == "3c":  # |Pe| 50-500 (release 5-120)
        return {**values, "D": 1.0/X.log_uniform(r, 50.0, 500.0)}
    if problem in ("6", "9"):  # low regularity in time: u ~ t^p with p just above alpha (release p >= max(alpha, 0.5))
        return {**values, "p": values["alpha"]+float(r.uniform(0.05, 0.5))}
    if problem == "11":  # the same against the largest order c of alpha(x, t) = c t^2 sin x
        return {**values, "p": values["c"]+float(r.uniform(0.05, 0.5))}
    return values


def harden_inventory(family, variant, r):
    if family == "burgers_gauss":  # sharper fronts: D 0.02-0.05 (release 0.05-0.2)
        return {**variant, "D": float(r.uniform(0.02, 0.05))}
    if family == "helmholtz":  # more oscillatory: k 2.5-4.2, below the first Dirichlet eigenvalue pi*sqrt(2) = 4.44
        return {**variant, "k": float(r.uniform(2.5, 4.2))}
    return variant


def secret(create=False):
    if not SECRET.exists():
        if not create:
            raise FileNotFoundError(f"{SECRET} missing: the hidden test split is evaluator-only")
        SECRET.parent.mkdir(parents=True, exist_ok=True)
        SECRET.write_text(_secrets.token_hex(16)+"\n")
    return SECRET.read_text().strip()


def version(split):
    return f"{BENCH_VERSION}:{split}" + (f":{secret()}" if split == "test" else "")


def templates(track):
    from .. import inventory as inv
    return {"tex": X.TEX_CASES, "ext": X.EXT_CASES, "inventory": list(inv.FAMILIES)}[track]


def hidden_instances(family):
    r = X.rng("instances", family, version=version("test"))
    return tuple(int(i) for i in r.choice(range(200, 1000), PER_TEMPLATE["test"], replace=False))


def task_id(track, split, template, index):
    digest = hashlib.sha256(f"{version(split)}:{track}:{template}:{index}".encode()).hexdigest()[:10]
    return f"bench-{track}-{split}-{digest}"


def tasks(track, split, hide_time_dependence=False):
    """The hidden-truth tasks of one split, with the expanded generator's numbers."""
    from .. import inventory as inv
    from .. import pdebench as pb
    from .. import pdebench_ext as px
    seed, out = version(split), []
    names = HARDENED[track] if split == "calibration" else templates(track)
    for template in names:
        for k, index in enumerate(INDICES[split][:PER_TEMPLATE.get(split, 10)]):
            tid, name = task_id(track, split, template, index), TASK_SPLIT[split]
            if track == "tex":
                spec = X.tex_spec(template, index, version=seed, adjust=harden_tex)
                out.append(pb.BenchTask(tid, pb.CLASS[template], name, spec))
            elif track == "ext":
                klass, _, seconds, _ = px.CASES[template]
                out.append(px.ExtTask(tid, klass, name, X.ext_spec(template, index, version=seed), max_seconds=seconds))
            else:
                instance = (INSTANCES.get(split) or hidden_instances(template))[k] \
                    if template in X.DATASET_FAMILIES else None
                variant = X.inventory_variant(template, X.rng("inventory", template, index, version=seed), index,
                                              version=seed, instance=instance)
                variant = harden_inventory(template, variant, X.rng("adjust", template, index, version=seed))
                klass, _, _, seconds = inv.FAMILIES[template]
                out.append(inv.InvTask(tid, klass, name, inv.make_spec(template, "evolution", variant),
                                       max_seconds=seconds,
                                       relative_l2_tolerance=inv.TOLERANCES.get(template, inv.TOLERANCE)))
    if track == "tex" and hide_time_dependence:
        raise ValueError("RSI-PDE Bench poses steady cases as steady (no hidden time dependence)")
    return out


def bench_cases(track, split, noise_fraction, hide_time_dependence=False):
    from .. import inventory as inv
    from .. import pdebench as pb
    from .. import pdebench_ext as px
    if split not in SPLIT_NAMES:
        raise ValueError(f"RSI-PDE Bench splits are {tuple(SPLIT_NAMES)}")
    split = SPLIT_NAMES[split]
    items = sorted(tasks(track, split, hide_time_dependence),
                   key=lambda t: hashlib.sha256(f"{BENCH_VERSION}:{t.task_id}".encode()).hexdigest())
    case = {"tex": pb.BenchCase, "ext": px.ExtCase, "inventory": inv.InvCase}[track]
    return [case(f"case-{PREFIX[split]}{i:02d}", task, noise_fraction) for i, task in enumerate(items)]
