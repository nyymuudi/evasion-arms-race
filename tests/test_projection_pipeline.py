"""
test_pipeline.py
================

End-to-end validation of the header-driven Layer A feasibility pipeline.

Checks:
  1. Source equivalence: project(..., Partition) == project(..., ClassificationReport)
     on the CICFlowMeter header. Proves the rule-derived source is a drop-in for
     the hand partition -> swapping to LYCOS later is a header swap, not a code change.
  2. Frozen enforcement: backward/flag/metadata features are reset to clean.
  3. Functional-core floor: forward rate/volume never drops below clean (DoS preserved).
  4. Physical validity: no negative durations/lengths survive.
  5. Derived recompute: Flow Packets/s etc. are consistent with atomic features,
     not with the adversarial perturbation.
  6. Idempotence: project(project(x)) == project(x).
"""

from __future__ import annotations

import numpy as np

from evasion_arms_race.features.partition import Control, PARTITION
from evasion_arms_race.features.rule_classifier import classify
from evasion_arms_race.features.projection import project, is_idempotent


def build_clean(features: list[str]) -> dict[str, float]:
    """A plausible clean DoS-Hulk-ish flow. Values are illustrative but ordered
    sensibly (positive durations, nonzero forward rate, some backward traffic)."""
    rng = np.random.default_rng(0)
    clean = {}
    for f in features:
        fl = f.lower()
        if "duration" in fl or "iat total" in fl:
            clean[f] = 500_000.0           # 0.5 s in microseconds
        elif "fwd packets/s" in fl:
            clean[f] = 800.0               # high forward rate (the flood)
        elif "total fwd packets" in fl:
            clean[f] = 400.0
        elif "init_win_bytes" in fl:
            clean[f] = 8192.0
        elif "ratio" in fl:
            clean[f] = 0.5
        else:
            clean[f] = float(abs(rng.normal(50, 20)))
    return clean


def build_adversarial(clean: dict[str, float]) -> dict[str, float]:
    """An adversarial vector that deliberately violates every constraint:
    flips frozen features, drives the DoS rate to near zero, injects negatives."""
    adv = dict(clean)
    for f in adv:
        fl = f.lower()
        if "bwd" in fl or "backward" in fl:
            adv[f] = clean[f] * 10 + 999        # tamper with server-side (illegal)
        elif "fwd packets/s" in fl:
            adv[f] = 1.0                          # kill the flood (must be floored back)
        elif "total fwd packets" in fl:
            adv[f] = 2.0                          # ditto
        elif "duration" in fl:
            adv[f] = -123.0                       # negative duration (illegal)
        elif "flag count" in fl:
            adv[f] = clean[f] + 500               # tamper connection-level flags
        elif "init_win_bytes_forward" in fl:
            adv[f] = 999_999.0                    # out of legal 16-bit window range
        else:
            adv[f] = clean[f] - 100.0             # push many controllables negative
    return adv


def main() -> int:
    header = sorted(PARTITION.all_features())
    report = classify(header)
    assert not report.unknown, f"unexpected unknowns: {report.unknown}"

    clean = build_clean(header)
    adv = build_adversarial(clean)

    proj_part = project(adv, clean, PARTITION)
    proj_rule = project(adv, clean, report)

    failures = []

    # 1. Source equivalence.
    for f in header:
        a, b = proj_part.vector[f], proj_rule.vector[f]
        if abs(a - b) > 1e-9:
            failures.append(f"source mismatch on {f!r}: {a} vs {b}")

    v = proj_rule.vector

    # 2. Frozen reset to clean.
    for f in header:
        if report.control_of(f) is Control.FROZEN and abs(v[f] - clean[f]) > 1e-9:
            failures.append(f"frozen not reset: {f!r} -> {v[f]} (clean {clean[f]})")

    # 3. Functional-core floor on forward rate/volume.
    for f in header:
        fl = f.lower()
        if "fwd packets/s" in fl or "total fwd packets" in fl:
            if v[f] < clean[f] - 1e-9:
                failures.append(f"DoS floor violated: {f!r} {v[f]} < clean {clean[f]}")

    # 4. Physical validity: no negatives among nonneg-prefixed controllables.
    for f in header:
        if report.control_of(f) is Control.CONTROLLABLE and v[f] < -1e-9:
            failures.append(f"negative controllable survived: {f!r} = {v[f]}")

    # 5. Derived consistency: Flow Packets/s recomputed from atomic, not adv.
    if "Flow Packets/s" in v:
        dur = v["Flow Duration"]
        tot = v["Total Fwd Packets"] + v["Total Backward Packets"]
        expected = 0.0 if dur <= 0 else tot / (dur / 1_000_000.0)
        if abs(v["Flow Packets/s"] - expected) > 1e-6:
            failures.append(
                f"derived inconsistent: Flow Packets/s {v['Flow Packets/s']} "
                f"!= recomputed {expected}")

    # 6. Idempotence.
    if not is_idempotent(adv, clean, report):
        failures.append("projection is not idempotent")

    print("=== Layer A header-driven pipeline test ===")
    print(report.summary())
    print(f"\nunreconstructable derived: {proj_rule.unreconstructable}")
    print(f"clamped features         : {len(proj_rule.clamped)}")
    print()
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for fmsg in failures:
            print(f"  - {fmsg}")
        return 1
    print("ALL CHECKS PASSED")
    print("  - Partition and rule-classifier sources are equivalent")
    print("  - Frozen reset, DoS floor, physical validity, derived recompute, idempotence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def test_pipeline_passes():
    """pytest entrypoint wrapping the end-to-end check."""
    assert main() == 0
