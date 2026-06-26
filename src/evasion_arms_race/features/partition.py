"""
feature_partition.py
====================

Canonical feature partition for CIC-IDS2017 (CICFlowMeter, 78 features) under a
black-box evasion threat model against a DoS Hulk classifier.

Purpose
-------
Every CICFlowMeter feature is assigned to exactly one of three control classes,
according to whether a DoS-Hulk attacker can manipulate it WITHOUT breaking
(a) protocol validity or (b) the attack's functional core (HTTP request volume
sufficient to exhaust the target).

    CONTROLLABLE : attacker freely sets these via timing / sizing / padding of
                   its own outbound (forward) packets.
    CONSTRAINED  : attacker influences these, but perturbation must be projected
                   into a protocol-/functionality-legal set.
    FROZEN       : determined by the server's response or the network; the
                   attacker cannot set them. Perturbing these is the cardinal
                   error that invalidates an evasion result.
    DERIVED      : aggregates that mix forward/backward components. Never
                   perturbed directly; recomputed from atomic features so that
                   the perturbed vector corresponds to a real packet stream.

The CICFlowMeter CSV ships with leading spaces and inconsistent casing in the
header. We normalise on read (strip + collapse internal whitespace) and key the
partition off the normalised names, so the module is robust to that quirk.

Threat model assumed: BLACK-BOX (query access to the detector only). The
partition itself is model-agnostic; it constrains the *search space*, not the
search algorithm.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping


# --------------------------------------------------------------------------- #
# Header normalisation
# --------------------------------------------------------------------------- #
def normalize(name: str) -> str:
    """Normalise a raw CICFlowMeter column name.

    Strips leading/trailing whitespace and collapses internal runs of
    whitespace to a single space. Casing is preserved because the canonical
    names below already use the dataset's casing; matching is done on the
    normalised string.
    """
    return re.sub(r"\s+", " ", name.strip())


# --------------------------------------------------------------------------- #
# Control classes
# --------------------------------------------------------------------------- #
class Control(Enum):
    CONTROLLABLE = "controllable"
    CONSTRAINED = "constrained"
    FROZEN = "frozen"
    DERIVED = "derived"


# --------------------------------------------------------------------------- #
# Canonical 78-feature partition (names already normalised)
# --------------------------------------------------------------------------- #
# Rationale comments are kept terse; the full justification lives in the design
# doc. The guiding question for each: "Can a DoS-Hulk attacker set this value
# without breaking protocol validity or the flood's functional core?"

CONTROLLABLE: set[str] = {
    # Forward inter-arrival timing -- attacker paces its own sends.
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    # Flow duration -- stretchable by spacing sends.
    "Flow Duration",
    # Forward packet sizing -- adjustable via request padding.
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Total Length of Fwd Packets", "Avg Fwd Segment Size",
    "Fwd Header Length",          # appears twice in CICFlowMeter; see DUP note
    # Active/idle rhythm -- attacker's burst/pause pattern.
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    # Forward-side flags the attacker chooses (within legality).
    "Fwd PSH Flags", "Fwd URG Flags",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
}

CONSTRAINED: set[str] = {
    # Forward volume/rate: tunable, but lowering too far kills the DoS.
    # This is the intersection of CONSTRAINED and the functional-core limit.
    "Total Fwd Packets",
    "Fwd Packets/s",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    # TCP window the attacker advertises -- legal range only.
    "Init_Win_bytes_forward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
}
# NOTE on "Fwd Header Length.1": CICFlowMeter emits "Fwd Header Length" twice;
# pandas disambiguates the second as ".1". The two columns are identical, so the
# duplicate carries no information. Recommended handling: DROP it at load time
# (see loader). It is therefore intentionally NOT placed in any class; if it
# survives into the header, validate_header() will flag it and the loader should
# remove it rather than perturb it.

FROZEN: set[str] = {
    # Everything backward = server's response. Attacker cannot set these.
    "Total Backward Packets",
    "Total Length of Bwd Packets",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Bwd Packets/s",
    "Bwd PSH Flags", "Bwd URG Flags",
    "Bwd Header Length",
    "Avg Bwd Segment Size",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_backward",
    # Connection-level flag counts largely driven by the server / stack.
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "PSH Flag Count",
    # Destination port is fixed by the target service (HTTP).
    "Destination Port",
}

# DERIVED: aggregates mixing fwd/bwd or otherwise functions of atomic features.
# These are recomputed, never perturbed directly. Each maps to a recompute fn.
DERIVED: set[str] = {
    "Flow Bytes/s",
    "Flow Packets/s",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "Min Packet Length", "Max Packet Length",
    "Average Packet Size",
    "Down/Up Ratio",
    "Flow IAT Total",          # = Flow Duration for single-flow accounting
}


# --------------------------------------------------------------------------- #
# Partition object
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Partition:
    controllable: frozenset[str]
    constrained: frozenset[str]
    frozen: frozenset[str]
    derived: frozenset[str]

    def control_of(self, feature: str) -> Control:
        f = normalize(feature)
        if f in self.controllable:
            return Control.CONTROLLABLE
        if f in self.constrained:
            return Control.CONSTRAINED
        if f in self.frozen:
            return Control.FROZEN
        if f in self.derived:
            return Control.DERIVED
        raise KeyError(f"Feature not in partition: {feature!r} (normalised {f!r})")

    def perturbable(self) -> frozenset[str]:
        """Features the search may move (controllable + constrained)."""
        return self.controllable | self.constrained

    def locked(self) -> frozenset[str]:
        """Features the search must hold fixed before recompute (frozen)."""
        return self.frozen

    def all_features(self) -> frozenset[str]:
        return self.controllable | self.constrained | self.frozen | self.derived


PARTITION = Partition(
    controllable=frozenset(CONTROLLABLE),
    constrained=frozenset(CONSTRAINED),
    frozen=frozenset(FROZEN),
    derived=frozenset(DERIVED),
)


# --------------------------------------------------------------------------- #
# Validation against an actual CSV header
# --------------------------------------------------------------------------- #
def validate_header(raw_columns: list[str], partition: Partition = PARTITION
                    ) -> dict[str, list[str]]:
    """Cross-check a real CSV header against the partition.

    Returns a report dict with three keys:
        'unpartitioned' : normalised columns present in CSV but in no class
        'missing'       : partition entries absent from the CSV
        'label_like'    : columns that look like the label / metadata
    A clean run has empty 'unpartitioned' (modulo label/metadata columns).
    """
    norm_cols = [normalize(c) for c in raw_columns]
    label_like = [c for c in norm_cols
                  if c.lower() in {"label", "flow id", "source ip",
                                   "destination ip", "source port", "timestamp",
                                   "protocol", "fwd header length"} is False
                  and c.lower() in {"label"}]
    known = partition.all_features()
    unpartitioned = [c for c in norm_cols
                     if c not in known and c.lower() != "label"]
    missing = sorted(known - set(norm_cols))
    return {
        "unpartitioned": unpartitioned,
        "missing": missing,
        "label_like": [c for c in norm_cols if c.lower() == "label"],
    }


if __name__ == "__main__":
    p = PARTITION
    print("controllable:", len(p.controllable))
    print("constrained :", len(p.constrained))
    print("frozen      :", len(p.frozen))
    print("derived     :", len(p.derived))
    print("total       :", len(p.all_features()))
