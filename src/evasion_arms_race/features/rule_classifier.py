"""
rule_classifier.py
==================

Feature-set-agnostic, rule-based control-class classifier for flow-based IDS
features (CICFlowMeter, LycoSTand/LYCOS, UNSW-NB15, ...).

Motivation
----------
Hard-coding a partition per feature set is brittle and does not transfer. The
control class of a flow feature is, however, largely a FUNCTION OF ITS NAME:
whether it describes the attacker's forward direction, the server's backward
direction, packet-level timing, or a mixed aggregate. We encode that mapping as
ordered rules so the same logic classifies any flow feature set. Anything the
rules cannot confidently place is returned as UNKNOWN for human confirmation --
never silently guessed.

Control classes (see feature_partition.Control):
    CONTROLLABLE : attacker's own forward timing / sizing / padding / flags
    CONSTRAINED  : forward volume/rate + TCP params (legal-box bounded)
    FROZEN       : backward (server) direction, connection-level flag counts,
                   fixed service port / metadata
    DERIVED      : aggregates mixing both directions; recomputed, not perturbed

Threat context: DoS-Hulk, black-box. The forward direction is the attacker;
the backward direction is the victim server. That asymmetry is the backbone of
every rule below.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from evasion_arms_race.features.partition import Control, normalize


# --------------------------------------------------------------------------- #
# Rule representation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rule:
    """A single classification rule.

    `pattern` is matched (case-insensitive, on the normalised name) as a regex.
    `control` is the class assigned on match. `why` documents the rationale.
    Rules are evaluated in order; first match wins. Order therefore encodes
    precedence: more specific / higher-priority rules come first.
    """
    pattern: str
    control: Control
    why: str

    def matches(self, norm_name: str) -> bool:
        return re.search(self.pattern, norm_name, flags=re.IGNORECASE) is not None


# --------------------------------------------------------------------------- #
# Ordered rule set. ORDER MATTERS: first match wins.
# --------------------------------------------------------------------------- #
# Strategy, general -> specific:
#   1. Metadata / label / fixed identifiers      -> FROZEN
#   2. Anything backward-direction               -> FROZEN  (server-controlled)
#   3. Connection-level flag *counts*            -> FROZEN  (mixed/stack-driven)
#   4. Mixed bidirectional aggregates            -> DERIVED (recomputed)
#   5. Forward volume / rate / TCP window        -> CONSTRAINED (legal box)
#   6. Forward timing / sizing / flags / bulk    -> CONTROLLABLE
#   7. Direction-neutral timing (Flow IAT, etc.) -> CONTROLLABLE (attacker paces)
# Unmatched -> UNKNOWN (caller must confirm).

RULES: list[Rule] = [
    # 1. Metadata / identifiers / label / fixed port.
    Rule(r"^(flow id|source ip|destination ip|source port|timestamp|protocol|label)$",
         Control.FROZEN, "metadata / identifier / label: not a perturbable feature"),
    Rule(r"^destination port$",
         Control.FROZEN, "target service port is fixed (HTTP for DoS-Hulk)"),

    # 4. Mixed bidirectional aggregates -> DERIVED. Placed BEFORE direction
    #    rules because e.g. 'Flow Bytes/s' contains no fwd/bwd token but mixes
    #    both; we catch these by explicit aggregate vocabulary.
    Rule(r"^(flow bytes/s|flow packets/s|down/up ratio|average packet size)$",
         Control.DERIVED, "aggregate mixing fwd+bwd; recompute from atomic"),
    Rule(r"^(packet length (mean|std|variance)|min packet length|max packet length)$",
         Control.DERIVED, "per-packet stat across both directions; recompute/clamp"),
    Rule(r"^flow iat total$",
         Control.DERIVED, "equals flow duration in single-flow accounting"),

    # 2. Backward direction = server response -> FROZEN.
    Rule(r"\bbwd\b|backward|init_win_bytes_backward",
         Control.FROZEN, "backward direction is server-controlled"),

    # 3. Connection-level flag COUNTS -> FROZEN (mix both dirs / stack-driven).
    #    Note: forward-specific PSH/URG flags are handled later as CONTROLLABLE;
    #    this rule targets the aggregate '... Flag Count' columns only.
    Rule(r"(fin|syn|rst|ack|urg|cwe|ece|psh) flag count",
         Control.FROZEN, "connection-level flag count, not purely attacker-set"),

    # 5. Forward volume / rate / TCP window / subflow -> CONSTRAINED.
    Rule(r"^(total fwd packets|fwd packets/s|subflow fwd (packets|bytes))$",
         Control.CONSTRAINED, "forward volume/rate: legal box + DoS functional floor"),
    Rule(r"^(init_win_bytes_forward|act_data_pkt_fwd|min_seg_size_forward)$",
         Control.CONSTRAINED, "forward TCP parameter: legal range only"),

    # 6. Forward timing / sizing / flags / bulk -> CONTROLLABLE.
    Rule(r"\bfwd\b.*(iat|length|header|segment|psh|urg|avg|bulk)|"
         r"(iat|length|header|segment|psh|urg|avg|bulk).*\bfwd\b",
         Control.CONTROLLABLE, "attacker's forward timing/sizing/flags/bulk"),
    Rule(r"^total length of fwd packets$",
         Control.CONTROLLABLE, "attacker's forward payload sizing"),

    # 7. Direction-neutral pacing the attacker controls.
    Rule(r"^(flow duration|flow iat (mean|std|max|min)|"
         r"active (mean|std|max|min)|idle (mean|std|max|min))$",
         Control.CONTROLLABLE, "attacker paces its own flow timing"),
]


@dataclass
class ClassificationReport:
    assignments: dict[str, Control]
    rationale: dict[str, str]
    unknown: list[str] = field(default_factory=list)

    # --- ControlSource protocol: lets projection consume this directly -----
    def control_of(self, feature: str) -> Control:
        f = normalize(feature)
        if f in self.assignments:
            return self.assignments[f]
        raise KeyError(
            f"Feature not classified: {feature!r} (normalised {f!r}). "
            f"It is in .unknown and needs confirmation before projection."
        )

    def all_features(self) -> frozenset[str]:
        return frozenset(self.assignments)

    @property
    def derived(self) -> frozenset[str]:
        return frozenset(f for f, c in self.assignments.items()
                         if c is Control.DERIVED)

    def by_class(self) -> dict[Control, list[str]]:
        out: dict[Control, list[str]] = {c: [] for c in Control}
        for feat, ctrl in self.assignments.items():
            out[ctrl].append(feat)
        return out

    def summary(self) -> str:
        bc = self.by_class()
        lines = [f"{c.value:13s}: {len(bc[c])}" for c in Control]
        lines.append(f"unknown      : {len(self.unknown)}")
        return "\n".join(lines)


def classify(raw_columns: list[str]) -> ClassificationReport:
    """Classify each column into a control class via ordered rules.

    Returns a report; unmatched columns land in `.unknown` for confirmation
    rather than being assigned a default class.
    """
    assignments: dict[str, Control] = {}
    rationale: dict[str, str] = {}
    unknown: list[str] = []

    for raw in raw_columns:
        name = normalize(raw)
        if name.lower() == "label":
            assignments[name] = Control.FROZEN
            rationale[name] = "label column"
            continue
        matched = False
        for rule in RULES:
            if rule.matches(name):
                assignments[name] = rule.control
                rationale[name] = rule.why
                matched = True
                break
        if not matched:
            unknown.append(name)

    return ClassificationReport(assignments, rationale, unknown)


if __name__ == "__main__":
    # CICFlowMeter canonical header (with the known leading-space quirk stripped)
    # used here purely to validate that the rules reproduce the hand partition.
    from evasion_arms_race.features.partition import PARTITION

    cic_header = sorted(PARTITION.all_features())
    report = classify(cic_header)
    print(report.summary())
    print()

    # Compare rule output against the hand-built partition.
    mismatches = []
    for feat in cic_header:
        hand = PARTITION.control_of(feat)
        rule = report.assignments.get(feat)
        if rule != hand:
            mismatches.append((feat, hand.value if hand else None,
                               rule.value if rule else "UNKNOWN"))
    if mismatches:
        print(f"MISMATCHES vs hand partition ({len(mismatches)}):")
        for feat, hand, rule in mismatches:
            print(f"  {feat:32s} hand={hand:13s} rule={rule}")
    else:
        print("Rule classifier reproduces the hand partition exactly.")
    if report.unknown:
        print("\nUNKNOWN (need confirmation):")
        for u in report.unknown:
            print(f"  {u}")
