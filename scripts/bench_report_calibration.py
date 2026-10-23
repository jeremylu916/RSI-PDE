"""Per-template solve rates of the calibration probes (scripts/bench_calibrate.py) and the difficulty tiers.

A template's difficulty is the mean pass rate over probes and variants:
    easy    >= 0.9   every probe solves almost every variant (saturated: hardened for the bench)
    medium  0.3-0.9  the probes differ
    hard    < 0.3    few probes solve it
data/bench/calibration/summary.json gets the table; it prints as Markdown.
    python scripts/bench_report_calibration.py
"""

import collections
import json

from pde_rsi.storage import PROJECT_ROOT

OUT = PROJECT_ROOT/"data/bench/calibration"
TIERS = ((0.9, "easy"), (0.3, "medium"), (0.0, "hard"))


_CALIBRATION = {}


def template(track, case_id):
    """x-pb-2a-03 -> 2a; x-ext-12b-00 -> 12b; x-inv-burgers_gauss-07 -> burgers_gauss; x-heat-04 -> heat.
    The bench calibration draws (tracks tex-cal, inventory-cal) have opaque ids: their template comes from the spec."""
    if track.endswith("-cal"):
        if track not in _CALIBRATION:
            from pde_rsi import inventory as inv
            from pde_rsi import pdebench as pb
            cases = {"tex-cal": pb, "inventory-cal": inv}[track].make_cases("bench-calibration")
            _CALIBRATION[track] = {c.case_id: c.truth.spec.get("problem", c.truth.spec.get("family_key")) for c in cases}
        return _CALIBRATION[track][case_id]
    stem = case_id.rsplit("-", 1)[0]
    return stem.split("-", 2)[2] if track in ("tex", "ext", "inventory") else stem.split("-", 1)[1]


def tier(rate):
    return next(name for low, name in TIERS if rate >= low)


def summarize():
    table = {}
    for track_dir in sorted(p for p in OUT.iterdir() if p.is_dir()):
        track = track_dir.name
        rates = collections.defaultdict(lambda: collections.defaultdict(list))
        for path in sorted(track_dir.glob("*.jsonl")):
            seen = {}
            for line in path.read_text().splitlines():
                row = json.loads(line)
                seen[row["case_id"]] = row  # the last row of a case counts
            for case_id, row in seen.items():
                rates[template(track, case_id)][path.stem].append(bool(row.get("passed")))
        table[track] = {}
        for name, probes in rates.items():
            per_probe = {probe: sum(v)/len(v) for probe, v in probes.items()}
            mean = sum(per_probe.values())/len(per_probe)
            table[track][name] = {"variants": max(len(v) for v in probes.values()), "per_probe": per_probe,
                                  "mean": mean, "spread": max(per_probe.values())-min(per_probe.values()),
                                  "tier": tier(mean)}
    return table


def main():
    table = summarize()
    (OUT/"summary.json").write_text(json.dumps(table, indent=2)+"\n")
    for track, rows in table.items():
        probes = sorted({p for r in rows.values() for p in r["per_probe"]})
        print(f"\n{track}\n\n| Template | n | "+" | ".join(probes)+" | Mean | Tier |")
        print("| --- | ---: | "+" | ".join("---:" for _ in probes)+" | ---: | --- |")
        for name, r in rows.items():
            cells = " | ".join(f"{r['per_probe'].get(p, float('nan')):.2f}" for p in probes)
            print(f"| {name} | {r['variants']} | {cells} | {r['mean']:.2f} | {r['tier']} |")
        counts = collections.Counter(r["tier"] for r in rows.values())
        print(f"\ntiers: {dict(counts)}")


if __name__ == "__main__":
    main()
