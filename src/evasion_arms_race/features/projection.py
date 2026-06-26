"""
feasibility_projection.py
==========================

The technical heart of Layer A. Given an arbitrary perturbed feature vector,
project it onto the nearest point that a DoS-Hulk attacker could ACTUALLY
realise as network traffic. This is what separates a valid evasion result from
a meaningless one.

The projection enforces, in order:

    1. FROZEN   features  -> reset to the original (clean) value. The attacker
                            does not control these (server/network determined).
    2. CONSTRAINED features -> clipped into a legality/functionality box, then
                            optionally floored to preserve the attack's
                            functional core (e.g. Fwd Packets/s must stay high
                            enough to constitute a flood).
    3. DERIVED  features  -> recomputed from atomic (controllable + constrained)
                            features, never taken from the perturbation, so the
                            vector stays internally consistent with a real flow.
    4. CONTROLLABLE features -> kept, but passed through basic physical-validity
                            clamps (no negative durations / counts / lengths).

Design notes
------------
* The projection is idempotent: project(project(x)) == project(x). This is
  required so an iterative search can call it every step without drift.
* It is feature-set agnostic. It depends on a Partition and a small set of
  recompute rules + bounds, all keyed by normalised feature name. Swapping
  CICFlowMeter for LYCOS means swapping the partition and the bounds table,
  NOT this code.
* Recompute rules are deliberately explicit and conservative. Where a faithful
  reconstruction of an aggregate from CICFlowMeter columns is not possible
  (because forward/backward components are not separable in the CSV), the rule
  falls back to clamping rather than fabricating a value, and records the
  feature in `unreconstructable` for the caller's awareness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, runtime_checkable

import numpy as np

from evasion_arms_race.features.partition import Control, Partition, PARTITION, normalize


@runtime_checkable
class ControlSource(Protocol):
    """Anything that can report a feature's control class and enumerate
    features. Both feature_partition.Partition and
    rule_classifier.ClassificationReport satisfy this, so the projection is
    fully feature-set agnostic and header-driven."""

    def control_of(self, feature: str) -> Control: ...
    def all_features(self) -> frozenset[str]: ...
    @property
    def derived(self) -> frozenset[str]: ...


# --------------------------------------------------------------------------- #
# Bounds / functional-core configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Bounds:
    """Per-feature legal box for CONSTRAINED features and validity floors.

    lo / hi are inclusive bounds. None means unbounded on that side.
    `floor_to_clean` means: never let the value drop below the original clean
    value (used to preserve the DoS functional core, e.g. forward packet rate).
    """
    lo: float | None = None
    hi: float | None = None
    floor_to_clean: bool = False


# Functional-core + legality bounds for CONSTRAINED features.
# Values chosen to be conservative and protocol-legal; tune against real data.
CONSTRAINED_BOUNDS: dict[str, Bounds] = {
    # Forward rate/volume are the DoS engine: do not let them fall below the
    # clean sample's level, or the "attack" stops being an attack.
    "Fwd Packets/s":      Bounds(lo=0.0, floor_to_clean=True),
    "Total Fwd Packets":  Bounds(lo=1.0, floor_to_clean=True),
    "Subflow Fwd Packets": Bounds(lo=0.0, floor_to_clean=True),
    "Subflow Fwd Bytes":  Bounds(lo=0.0, floor_to_clean=True),
    # TCP receive window: legal 16-bit-ish range; attacker may set within it.
    "Init_Win_bytes_forward": Bounds(lo=-1.0, hi=65535.0),
    "act_data_pkt_fwd":   Bounds(lo=0.0),
    "min_seg_size_forward": Bounds(lo=0.0),
}

# CONTROLLABLE physical-validity: features that simply cannot be negative.
NONNEGATIVE_PREFIXES = (
    "Flow Duration", "Fwd IAT", "Flow IAT", "Fwd Packet Length",
    "Total Length of Fwd", "Avg Fwd Segment Size", "Fwd Header Length",
    "Active", "Idle", "Fwd PSH", "Fwd URG", "Fwd Avg",
)


# --------------------------------------------------------------------------- #
# Derived-feature recompute rules
# --------------------------------------------------------------------------- #
# Each rule takes the working feature dict (normalised names -> float) and
# returns the recomputed value. Rules use only atomic features that the CSV
# actually separates. Where separation is impossible, the rule returns None and
# the feature is reported as unreconstructable (left at its clamped value).

def _safe(d: Mapping[str, float], k: str) -> float | None:
    v = d.get(k)
    return None if v is None else float(v)


def _recompute_flow_packets_s(d: Mapping[str, float]) -> float | None:
    dur = _safe(d, "Flow Duration")
    tfwd = _safe(d, "Total Fwd Packets")
    tbwd = _safe(d, "Total Backward Packets")
    if dur is None or tfwd is None or tbwd is None:
        return None
    if dur <= 0:
        return 0.0
    # CICFlowMeter Flow Duration is in microseconds -> packets per second.
    return (tfwd + tbwd) / (dur / 1_000_000.0)


def _recompute_flow_bytes_s(d: Mapping[str, float]) -> float | None:
    dur = _safe(d, "Flow Duration")
    lfwd = _safe(d, "Total Length of Fwd Packets")
    lbwd = _safe(d, "Total Length of Bwd Packets")
    if dur is None or lfwd is None or lbwd is None:
        return None
    if dur <= 0:
        return 0.0
    return (lfwd + lbwd) / (dur / 1_000_000.0)


def _recompute_down_up_ratio(d: Mapping[str, float]) -> float | None:
    tfwd = _safe(d, "Total Fwd Packets")
    tbwd = _safe(d, "Total Backward Packets")
    if tfwd is None or tbwd is None:
        return None
    if tfwd <= 0:
        return 0.0
    return tbwd / tfwd


def _recompute_avg_packet_size(d: Mapping[str, float]) -> float | None:
    lfwd = _safe(d, "Total Length of Fwd Packets")
    lbwd = _safe(d, "Total Length of Bwd Packets")
    tfwd = _safe(d, "Total Fwd Packets")
    tbwd = _safe(d, "Total Backward Packets")
    if None in (lfwd, lbwd, tfwd, tbwd):
        return None
    n = tfwd + tbwd
    if n <= 0:
        return 0.0
    return (lfwd + lbwd) / n


# Packet Length Mean/Std/Variance/Min/Max mix per-packet stats across both
# directions and are NOT faithfully reconstructable from CICFlowMeter's
# aggregate columns alone. We mark them unreconstructable: they are clamped to
# remain >= 0 and internally ordered, but not fabricated. (Under LYCOS, richer
# columns may permit exact recompute; that is a per-partition concern.)
DERIVED_RULES: dict[str, Callable[[Mapping[str, float]], float | None]] = {
    "Flow Packets/s": _recompute_flow_packets_s,
    "Flow Bytes/s": _recompute_flow_bytes_s,
    "Down/Up Ratio": _recompute_down_up_ratio,
    "Average Packet Size": _recompute_avg_packet_size,
    "Flow IAT Total": lambda d: _safe(d, "Flow Duration"),
}


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #
@dataclass
class ProjectionResult:
    vector: dict[str, float]
    unreconstructable: list[str]
    clamped: list[str]


def project(
    perturbed: Mapping[str, float],
    clean: Mapping[str, float],
    source: ControlSource = PARTITION,
    constrained_bounds: Mapping[str, Bounds] = CONSTRAINED_BOUNDS,
) -> ProjectionResult:
    """Project `perturbed` onto the feasible set, given the `clean` original.

    `source` supplies each feature's control class; it may be a Partition
    (hard-coded) or a ClassificationReport (rule-derived from a real header).
    Both dicts are keyed by normalised feature name and share the same keys.
    Returns a ProjectionResult whose `.vector` is realisable as DoS-Hulk traffic.
    """
    out: dict[str, float] = {}
    clamped: list[str] = []

    # Normalise keys defensively.
    pert = {normalize(k): float(v) for k, v in perturbed.items()}
    base = {normalize(k): float(v) for k, v in clean.items()}

    for feat in source.all_features():
        ctrl = source.control_of(feat)

        if ctrl is Control.FROZEN:
            # Attacker cannot move these: reset to clean.
            out[feat] = base[feat]

        elif ctrl is Control.CONSTRAINED:
            v = pert.get(feat, base[feat])
            b = constrained_bounds.get(feat, Bounds())
            if b.floor_to_clean:
                v = max(v, base[feat])          # preserve functional core
            if b.lo is not None:
                v = max(v, b.lo)
            if b.hi is not None:
                v = min(v, b.hi)
            if v != pert.get(feat, v):
                clamped.append(feat)
            out[feat] = v

        elif ctrl is Control.CONTROLLABLE:
            v = pert.get(feat, base[feat])
            if any(feat.startswith(p) for p in NONNEGATIVE_PREFIXES):
                if v < 0:
                    v = 0.0
                    clamped.append(feat)
            out[feat] = v

        elif ctrl is Control.DERIVED:
            # placeholder; recomputed in second pass below
            out[feat] = pert.get(feat, base[feat])

    # Second pass: recompute DERIVED from the now-settled atomic features.
    unreconstructable: list[str] = []
    for feat in source.derived:
        rule = DERIVED_RULES.get(feat)
        if rule is None:
            unreconstructable.append(feat)
            # leave clamped to >= 0 for sanity
            out[feat] = max(out.get(feat, 0.0), 0.0)
            continue
        val = rule(out)
        if val is None:
            unreconstructable.append(feat)
            out[feat] = max(out.get(feat, 0.0), 0.0)
        else:
            out[feat] = float(val)

    return ProjectionResult(
        vector=out,
        unreconstructable=sorted(set(unreconstructable)),
        clamped=sorted(set(clamped)),
    )


def is_idempotent(
    perturbed: Mapping[str, float],
    clean: Mapping[str, float],
    source: ControlSource = PARTITION,
    tol: float = 1e-9,
) -> bool:
    """Check project(project(x)) == project(x) within tolerance."""
    r1 = project(perturbed, clean, source).vector
    r2 = project(r1, clean, source).vector
    return all(abs(r1[k] - r2[k]) <= tol for k in r1)
