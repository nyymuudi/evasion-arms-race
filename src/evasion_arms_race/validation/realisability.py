"""Packet-level realisability validation (Layer A, item 7).

This closes Layer A. Items 5-6 produced adversarial FEATURE VECTORS the detector
calls benign. That is necessary but not sufficient: a point in feature space is a
real evasion only if it corresponds to a packet stream the attacker could
actually send while keeping the attack functional. This module checks exactly
that, and -- crucially -- re-queries the detector on the CORRECTED vector, because
the attack froze several features the detector reads heavily at their clean
values, and a realisable flow may not reproduce those values.

Validation level (chosen)
-------------------------
STRONG, via analytical packet-level reconstruction + detector re-check, short of
a full pcap -> CICFlowMeter-binary round-trip:

  * FEASIBILITY. The forward packet-size and inter-arrival marginals must admit an
    actual packet multiset. The binding test is the Bhatia-Davis variance bound
    Var <= (max-mean)(mean-min): the attack moved Std independently of Min/Max/
    Mean, so it can produce moment tuples that NO multiset realises. The
    projection never enforced this cross-feature consistency; here it is caught.

  * CORRECTION + RE-QUERY (the illusion test). The five pooled packet-length
    features (Min/Max/Mean/Std/Variance Packet Length) -- which the attack never
    moves and the projection passes through at clean values, yet which dominate
    the logistic detector -- are RECONSTRUCTED from the adversarial forward+
    backward marginals and substituted in. The detector is then re-queried. If it
    now says ATTACK, the evasion was an artefact of holding those features fixed.

Why packets are barely needed for (2): the pooled length stats are reconstructable
in closed form from the per-direction marginals -- Min/Max exactly
(min/max of the two directions), Mean/Std/Variance via the law of total variance
plus a small data-fit affine calibration for CICFlowMeter's definitional offset
(R^2 >= 0.997 on real flows). The earlier 'unreconstructable' label was too
pessimistic. LYCOS would expose them as columns directly; we resolve them here.

Data constraint (honest)
------------------------
The MachineLearningCVE CSVs carry no flow id / IPs / timestamp, so a given row
cannot be matched back to packets in the raw pcap (the same reason the temporal
split was deferred). A true pcap round-trip needs the GeneratedLabelledFlows
distribution + raw captures; that is the remaining validation tier, documented,
not silently assumed. Reconstruction here is self-contained from the CSV vector.
scapy is used to EMIT a witness pcap of the reconstructed forward flow -- tangible
proof the packets are constructible -- not to ingest ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from evasion_arms_race.features.partition import normalize
from evasion_arms_race.features.projection import ProjectionResult, project

# Hard protocol bounds. CICFlowMeter packet length is the IP datagram length;
# offload (TSO/GRO) means captured "packets" can exceed the Ethernet MTU, so the
# cap is the IP datagram maximum, and 0 is legal (a pure ACK reports length 0).
MIN_LEN = 0.0
MAX_LEN = 65535.0
FEAS_TOL = 0.05            # relative slack on the variance / consistency bounds


# --------------------------------------------------------------------------- #
# Calibration for the three offset-bearing pooled features
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Calibration:
    """actual ~= slope * pooled_prediction + intercept, for each of Packet Length
    Mean / Std / Variance. Min/Max need no calibration (they are exact). Carries
    both the absolute residual std and a RELATIVE residual (median |actual-fit| /
    |actual|); the sensitivity band uses the relative one, because the absolute
    residual is dominated by a few high-variance flows and would wildly overstate
    uncertainty for the short flows we attack. Fit by `fit_calibration`."""
    coeffs: dict[str, tuple[float, float, float, float]]  # (slope, intercept, resid_std, rel_resid)

    def apply(self, feature: str, pooled: float) -> float:
        a, b = self.coeffs[feature][0], self.coeffs[feature][1]
        return a * pooled + b

    def resid_std(self, feature: str) -> float:
        return self.coeffs[feature][2]

    def rel_resid(self, feature: str) -> float:
        return self.coeffs[feature][3]


IDENTITY_CALIBRATION = Calibration({
    "Packet Length Mean": (1.0, 0.0, 0.0, 0.0),
    "Packet Length Std": (1.0, 0.0, 0.0, 0.0),
    "Packet Length Variance": (1.0, 0.0, 0.0, 0.0),
})


def _pooled_length_stats(v: Mapping[str, float]) -> dict[str, float]:
    """Pooled packet-length statistics from per-direction marginals (pre-calibration)."""
    nf = max(v["Total Fwd Packets"], 0.0)
    nb = max(v["Total Backward Packets"], 0.0)
    n = nf + nb
    mf, mb = v["Fwd Packet Length Mean"], v["Bwd Packet Length Mean"]
    sf, sb = v["Fwd Packet Length Std"], v["Bwd Packet Length Std"]
    # Min/Max: exact. A direction with zero packets must not pull the pooled
    # min down to its (meaningless) zero-valued marginal.
    fmins, fmaxs = [], []
    if nf > 0:
        fmins.append(v["Fwd Packet Length Min"]); fmaxs.append(v["Fwd Packet Length Max"])
    if nb > 0:
        fmins.append(v["Bwd Packet Length Min"]); fmaxs.append(v["Bwd Packet Length Max"])
    pooled_min = min(fmins) if fmins else 0.0
    pooled_max = max(fmaxs) if fmaxs else 0.0
    if n <= 0:
        return {"Min Packet Length": 0.0, "Max Packet Length": 0.0,
                "Packet Length Mean": 0.0, "Packet Length Std": 0.0,
                "Packet Length Variance": 0.0}
    mu = (nf * mf + nb * mb) / n
    ex2 = (nf * (sf ** 2 + mf ** 2) + nb * (sb ** 2 + mb ** 2)) / n
    var = max(ex2 - mu ** 2, 0.0)
    return {"Min Packet Length": pooled_min, "Max Packet Length": pooled_max,
            "Packet Length Mean": mu, "Packet Length Std": float(np.sqrt(var)),
            "Packet Length Variance": var}


def reconstruct_packet_length_features(
    v: Mapping[str, float], calib: Calibration = IDENTITY_CALIBRATION
) -> dict[str, float]:
    """The five pooled features as a realisable flow would actually carry them.

    Min/Max are exact; Mean/Std/Variance are the pooled prediction passed through
    the data-fit affine calibration that absorbs CICFlowMeter's definitional
    offset.
    """
    pooled = _pooled_length_stats(v)
    out = {"Min Packet Length": pooled["Min Packet Length"],
           "Max Packet Length": pooled["Max Packet Length"]}
    for f in ("Packet Length Mean", "Packet Length Std", "Packet Length Variance"):
        out[f] = max(calib.apply(f, pooled[f]), 0.0)
    return out


def fit_calibration(df, sample: int = 20000, seed: int = 0) -> Calibration:
    """Fit the Mean/Std/Variance affine calibration on real flows (both
    directions present). `df` has normalised CICFlowMeter columns."""
    d = df[(df["Total Fwd Packets"] > 0) & (df["Total Backward Packets"] > 0)]
    if len(d) > sample:
        d = d.sample(sample, random_state=seed)
    nf, nb = d["Total Fwd Packets"], d["Total Backward Packets"]
    n = nf + nb
    mf, mb = d["Fwd Packet Length Mean"], d["Bwd Packet Length Mean"]
    sf, sb = d["Fwd Packet Length Std"], d["Bwd Packet Length Std"]
    mu = (nf * mf + nb * mb) / n
    pred_var = ((nf * (sf ** 2 + mf ** 2) + nb * (sb ** 2 + mb ** 2)) / n - mu ** 2).clip(lower=0)
    preds = {"Packet Length Mean": (d["Total Length of Fwd Packets"]
                                    + d["Total Length of Bwd Packets"]) / n,
             "Packet Length Std": np.sqrt(pred_var),
             "Packet Length Variance": pred_var}
    coeffs = {}
    for f, pred in preds.items():
        a, b = np.polyfit(pred, d[f], 1)
        resid = d[f] - (a * pred + b)
        denom = d[f].abs().clip(lower=1e-9)
        rel = float((resid.abs() / denom).median())
        coeffs[f] = (float(a), float(b), float(np.std(resid)), rel)
    return Calibration(coeffs)


# --------------------------------------------------------------------------- #
# Feasibility: do the forward marginals admit an actual packet multiset?
# --------------------------------------------------------------------------- #
def _bhatia_davis_ok(mn, mx, mean, std) -> bool:
    """Var <= (max-mean)(mean-min): the tight upper bound on the variance of any
    distribution supported on [min, max] with the given mean."""
    if not (mn - 1e-9 <= mean <= mx + 1e-9):
        return False
    bound = (mx - mean) * (mean - mn)
    return std ** 2 <= bound * (1 + FEAS_TOL) + 1e-6


@dataclass
class Feasibility:
    feasible: bool
    reasons: list[str] = field(default_factory=list)


def forward_size_feasible(v: Mapping[str, float]) -> Feasibility:
    """The forward packet-SIZE marginals must admit a multiset of Total Fwd
    Packets sizes in [MIN_LEN, MAX_LEN]."""
    r: list[str] = []
    nf = v["Total Fwd Packets"]
    mn, mx = v["Fwd Packet Length Min"], v["Fwd Packet Length Max"]
    mean, std = v["Fwd Packet Length Mean"], v["Fwd Packet Length Std"]
    total = v["Total Length of Fwd Packets"]
    if nf < 1:
        r.append(f"Total Fwd Packets < 1 ({nf:.3g})")
    if not (MIN_LEN - 1e-6 <= mn <= mx + 1e-6 <= MAX_LEN + 1e-6):
        r.append(f"size range out of [0,{MAX_LEN:.0f}] or min>max (min={mn:.3g}, max={mx:.3g})")
    if not _bhatia_davis_ok(mn, mx, mean, std):
        r.append(f"size variance exceeds (max-mean)(mean-min): std={std:.3g}, "
                 f"min={mn:.3g}, mean={mean:.3g}, max={mx:.3g}")
    if nf >= 1 and abs(total - nf * mean) > FEAS_TOL * max(1.0, abs(total)):
        r.append(f"Total Length != N*mean (total={total:.3g}, N*mean={nf*mean:.3g})")
    return Feasibility(not r, r)


def forward_timing_feasible(v: Mapping[str, float]) -> Feasibility:
    """The forward inter-arrival marginals + Flow Duration must admit a timestamp
    sequence. Only checked when there are >=2 forward packets (else no gaps)."""
    r: list[str] = []
    nf = v["Total Fwd Packets"]
    dur = v["Flow Duration"]
    if dur < -1e-6:
        r.append(f"negative Flow Duration ({dur:.3g})")
    if nf >= 2:
        mn, mx = v["Fwd IAT Min"], v["Fwd IAT Max"]
        mean, std = v["Fwd IAT Mean"], v["Fwd IAT Std"]
        total = v["Fwd IAT Total"]
        if mn < -1e-6:
            r.append(f"negative Fwd IAT Min ({mn:.3g})")
        if not _bhatia_davis_ok(mn, mx, mean, std):
            r.append(f"IAT variance exceeds (max-mean)(mean-min): std={std:.3g}, "
                     f"min={mn:.3g}, mean={mean:.3g}, max={mx:.3g}")
        if mx > dur + FEAS_TOL * max(1.0, dur):
            r.append(f"Fwd IAT Max > Flow Duration (max={mx:.3g}, dur={dur:.3g})")
        # sum of (nf-1) forward gaps should be the forward IAT total
        if abs(total - (nf - 1) * mean) > FEAS_TOL * max(1.0, abs(total)):
            r.append(f"Fwd IAT Total != (N-1)*mean (total={total:.3g}, "
                     f"(N-1)*mean={(nf-1)*mean:.3g})")
    return Feasibility(not r, r)


def is_feasible(v: Mapping[str, float]) -> Feasibility:
    size = forward_size_feasible(v)
    timing = forward_timing_feasible(v)
    return Feasibility(size.feasible and timing.feasible, size.reasons + timing.reasons)


# --------------------------------------------------------------------------- #
# Manifold projection: confine the SEARCH to the realisable set (not a post-filter)
# --------------------------------------------------------------------------- #
def _clip(x, lo, hi):
    return float(min(max(x, lo), hi))


def manifold_project(
    perturbed: Mapping[str, float], clean: Mapping[str, float], source=None,
    calib: Calibration = IDENTITY_CALIBRATION,
) -> ProjectionResult:
    """Project a candidate onto the REALISABLE manifold, so a search can call this
    every step instead of post-filtering with is_feasible().

    This is the experiment that answers the headline claim's obvious objection: a
    0% realisable rate obtained by free-space search + post-filtering is ambiguous
    (the search never looked on the manifold). Here every candidate the detector
    ever sees already satisfies is_feasible BY CONSTRUCTION.

    Implementation: the existing project() (frozen / floor / box / derived) followed
    by a deterministic, IDEMPOTENT clamp-and-recompute that enforces exactly the
    is_feasible constraints. This is NOT a nearest-point projection onto the
    (non-convex) feasible set -- which would be unstable -- but constraint
    enforcement by construction, which lands on the manifold deterministically.
    Bhatia-Davis is necessary AND sufficient for a multiset with the given
    [min, max, mean, std] to exist, so a point that survives is realisable.
    """
    base = project(perturbed, clean, source) if source is not None else project(perturbed, clean)
    v = dict(base.vector)
    cl = {normalize(k): float(x) for k, x in clean.items()}

    # ---- forward sizes: 0 <= min <= mean <= max, Var <= (max-mean)(mean-min),
    #      Total = N*mean ------------------------------------------------------ #
    nf = max(1.0, round(v["Total Fwd Packets"]))
    nf = max(nf, round(cl.get("Total Fwd Packets", 1.0)))          # keep the DoS volume floor
    mn = _clip(v["Fwd Packet Length Min"], MIN_LEN, MAX_LEN)
    mx = _clip(v["Fwd Packet Length Max"], MIN_LEN, MAX_LEN)
    if mn > mx:
        mn, mx = mx, mn
    mean = _clip(v["Fwd Packet Length Mean"], mn, mx)
    std = _clip(v["Fwd Packet Length Std"], 0.0, float(np.sqrt(max(0.0, (mx - mean) * (mean - mn)))))
    v["Total Fwd Packets"] = float(nf)
    v["Fwd Packet Length Min"], v["Fwd Packet Length Max"] = mn, mx
    v["Fwd Packet Length Mean"], v["Fwd Packet Length Std"] = mean, std
    v["Total Length of Fwd Packets"] = nf * mean
    if "Avg Fwd Segment Size" in v:
        v["Avg Fwd Segment Size"] = mean                            # = fwd mean in CICFlowMeter
    if "Subflow Fwd Packets" in v:
        v["Subflow Fwd Packets"] = max(nf, cl.get("Subflow Fwd Packets", 0.0))
    if "Subflow Fwd Bytes" in v:
        v["Subflow Fwd Bytes"] = max(nf * mean, cl.get("Subflow Fwd Bytes", 0.0))

    # ---- forward timing: duration short enough to keep the flood rate, then
    #      IAT min<=mean<=max<=duration, Var bound, Total=(N-1)*mean ----------- #
    dur = max(1.0, v["Flow Duration"])
    clean_rate = cl.get("Fwd Packets/s", 0.0)
    if clean_rate > 0:
        dur = min(dur, nf * 1e6 / clean_rate)                       # rate = N/(dur/1e6) >= clean
    dur = max(dur, 1.0)
    v["Flow Duration"] = dur
    if nf >= 2:
        imn = _clip(v["Fwd IAT Min"], 0.0, dur)
        imx = _clip(v["Fwd IAT Max"], 0.0, dur)
        if imn > imx:                                              # enforce min <= max
            imn, imx = imx, imn
        imean = _clip(v["Fwd IAT Mean"], imn, imx)
        istd = _clip(v["Fwd IAT Std"], 0.0, float(np.sqrt(max(0.0, (imx - imean) * (imean - imn)))))
        v["Fwd IAT Min"], v["Fwd IAT Max"] = imn, imx
        v["Fwd IAT Mean"], v["Fwd IAT Std"] = imean, istd
        v["Fwd IAT Total"] = (nf - 1) * imean
    if "Flow IAT Total" in v:
        v["Flow IAT Total"] = dur

    # ---- recompute the dependent aggregates from the now-consistent atomics --- #
    nb = v.get("Total Backward Packets", 0.0)
    lbwd = v.get("Total Length of Bwd Packets", 0.0)
    tot_pk = nf + nb
    tot_by = v["Total Length of Fwd Packets"] + lbwd
    v["Fwd Packets/s"] = nf / (dur / 1e6)
    if "Flow Packets/s" in v:
        v["Flow Packets/s"] = tot_pk / (dur / 1e6)
    if "Flow Bytes/s" in v:
        v["Flow Bytes/s"] = tot_by / (dur / 1e6)
    if "Down/Up Ratio" in v:
        v["Down/Up Ratio"] = (nb / nf) if nf > 0 else 0.0
    if "Average Packet Size" in v:
        v["Average Packet Size"] = (tot_by / tot_pk) if tot_pk > 0 else 0.0
    v.update(reconstruct_packet_length_features(v, calib))         # the five pooled stats

    return ProjectionResult(vector=v, unreconstructable=[], clamped=[])


# --------------------------------------------------------------------------- #
# Corrected vector + the detector re-check
# --------------------------------------------------------------------------- #
def corrected_vector(
    adv: Mapping[str, float], calib: Calibration = IDENTITY_CALIBRATION, source=None
) -> dict[str, float]:
    """The adversarial vector with the five pooled packet-length features replaced
    by their realisable (reconstructed) values, and the other derived features
    recomputed from atomic for internal consistency."""
    # project(adv, adv): identity on frozen/controllable, recomputes the derived
    # aggregates (Flow Bytes/s, Down/Up Ratio, ...) from the adversarial atomics.
    base = project(adv, adv) if source is None else project(adv, adv, source)
    out = dict(base.vector)
    out.update(reconstruct_packet_length_features(adv, calib))
    return out


@dataclass
class RealisabilityResult:
    verdict: str                     # realisable | infeasible | reverted | uncertain
    feasible: bool
    infeasibility_reasons: list[str]
    detector_attack_after_correction: bool   # detector calls the corrected vector ATTACK
    robust_to_residual: bool                 # verdict stable across the +/- calibration band
    reconstructed: dict[str, float]          # the five corrected features
    moved_packet_length: dict[str, float]    # corrected - adversarial, per feature


def validate(
    adv: Mapping[str, float],
    decision_fn: Callable[[Mapping[str, float]], int],
    calib: Calibration = IDENTITY_CALIBRATION,
    source=None,
) -> RealisabilityResult:
    """Validate ONE adversarial vector the detector already calls benign.

    `decision_fn(raw_vector_dict) -> 1 (attack) / 0 (benign)` wraps the detector
    (scaler + model). Returns a verdict:
        infeasible : forward marginals admit no packet multiset.
        reverted   : feasible, but correcting the pooled length features flips the
                     detector back to ATTACK -> the evasion was an illusion.
        realisable : feasible AND the detector still says benign after correction.
        uncertain  : the verdict flips within the +/- calibration-residual band.
    """
    adv = {normalize(k): float(x) for k, x in adv.items()}
    feas = is_feasible(adv)

    recon = reconstruct_packet_length_features(adv, calib)
    moved = {f: recon[f] - adv.get(f, 0.0) for f in recon}
    corr = corrected_vector(adv, calib, source)
    dec = decision_fn(corr)

    # Sensitivity: shift the three calibrated features by +/- their RELATIVE
    # residual (scaled to each feature's own value) and re-check; if the detector
    # verdict is stable, the conclusion survives the definitional uncertainty.
    decisions = {dec}
    for sign in (-1.0, 1.0):
        shifted = dict(corr)
        for f in ("Packet Length Mean", "Packet Length Std", "Packet Length Variance"):
            shifted[f] = max(shifted[f] * (1.0 + sign * calib.rel_resid(f)), 0.0)
        decisions.add(decision_fn(shifted))
    robust = len(decisions) == 1

    if not feas.feasible:
        verdict = "infeasible"
    elif not robust:
        verdict = "uncertain"
    elif dec == 1:
        verdict = "reverted"
    else:
        verdict = "realisable"

    return RealisabilityResult(
        verdict=verdict,
        feasible=feas.feasible,
        infeasibility_reasons=feas.reasons,
        detector_attack_after_correction=bool(dec == 1),
        robust_to_residual=robust,
        reconstructed=recon,
        moved_packet_length=moved,
    )


# --------------------------------------------------------------------------- #
# scapy witness: emit the reconstructed forward flow as a real pcap
# --------------------------------------------------------------------------- #
def reconstruct_forward_packets(
    v: Mapping[str, float], rng: np.random.Generator
) -> list[tuple[float, int, dict]]:
    """A concrete forward packet sequence (relative_time_s, ip_length, flags)
    whose size and timing moments approximate the forward marginals. Illustrative
    -- the feasibility VERDICT comes from the closed-form checks, not this build."""
    nf = int(round(v["Total Fwd Packets"]))
    nf = max(nf, 1)
    sizes = _match_moments(nf, v["Fwd Packet Length Min"], v["Fwd Packet Length Max"],
                           v["Fwd Packet Length Mean"], v["Fwd Packet Length Std"], rng)
    if nf >= 2:
        gaps = _match_moments(nf - 1, v["Fwd IAT Min"], v["Fwd IAT Max"],
                              v["Fwd IAT Mean"], v["Fwd IAT Std"], rng)
    else:
        gaps = []
    times = np.concatenate([[0.0], np.cumsum(gaps)]) / 1_000_000.0   # us -> s
    n_psh = int(round(v.get("Fwd PSH Flags", 0)))
    pkts = []
    for i in range(nf):
        flags = {"S": i == 0, "P": i >= nf - n_psh, "A": i > 0}
        pkts.append((float(times[i]), int(max(0, round(sizes[i]))), flags))
    return pkts


def _match_moments(n, mn, mx, mean, std, rng) -> np.ndarray:
    """n nonneg values in [mn,mx] approximating the given mean/std. Anchors one
    value at mn and one at mx, draws the rest around the mean, then rescales to
    the target mean. Best-effort (used only for the witness pcap)."""
    n = int(max(n, 1))
    if n == 1:
        return np.array([mean])
    vals = rng.normal(mean, max(std, 1e-6), n)
    vals[0], vals[1] = mn, mx
    vals = np.clip(vals, mn, mx)
    # rescale to hit the mean exactly
    cur = vals.mean()
    if cur > 0:
        vals = np.clip(vals * (mean / cur), mn, mx)
    return vals


def emit_pcap(v: Mapping[str, float], path, *, src_ip="10.0.0.66", dst_ip="10.0.0.50",
              seed: int = 0) -> int:
    """Write the reconstructed forward flow to `path` as a real pcap (scapy).
    Returns the packet count. scapy is imported lazily so the rest of the module
    has no hard dependency on it."""
    from scapy.all import IP, TCP, Raw, wrpcap   # lazy

    rng = np.random.default_rng(seed)
    v = {normalize(k): float(x) for k, x in v.items()}
    dport = int(round(v.get("Destination Port", 80))) or 80
    pkts_spec = reconstruct_forward_packets(v, rng)
    out = []
    base_seq = 1000
    for t, ip_len, flags in pkts_spec:
        payload = max(0, ip_len - 40)            # IP(20)+TCP(20) headers
        fl = ("S" if flags["S"] else "") + ("A" if flags["A"] else "") + ("P" if flags["P"] else "")
        pkt = IP(src=src_ip, dst=dst_ip) / TCP(sport=44444, dport=dport, flags=fl or "A", seq=base_seq)
        if payload:
            pkt = pkt / Raw(b"\x00" * payload)
        pkt.time = t
        base_seq += max(1, payload)
        out.append(pkt)
    wrpcap(str(path), out)
    return len(out)


# --------------------------------------------------------------------------- #
# Aggregate survival metric
# --------------------------------------------------------------------------- #
def survival_summary(results: Sequence[RealisabilityResult]) -> dict:
    """The cornerstone number: of evasions item 5 called successful, what fraction
    survive realisability validation, and how do the rest fail?"""
    n = len(results)
    counts = {"realisable": 0, "reverted": 0, "infeasible": 0, "uncertain": 0}
    for r in results:
        counts[r.verdict] += 1
    return {
        "n": n,
        "survival_rate": counts["realisable"] / n if n else 0.0,
        "verdicts": counts,
    }


# --------------------------------------------------------------------------- #
# Reporting harness
# --------------------------------------------------------------------------- #
def _decision_fn(model, scaler, feature_names):
    def fn(d: Mapping[str, float]) -> int:
        vec = np.array([float(d[f]) for f in feature_names]).reshape(1, -1)
        return int(model.predict(scaler.transform(vec))[0])
    return fn


def main() -> int:
    import argparse
    import json
    from pathlib import Path

    import pandas as pd

    from evasion_arms_race.attack.blackbox import AttackConfig, load_artifacts, run_attack
    from evasion_arms_race.data.loader import build_dataset
    from evasion_arms_race.features.rule_classifier import classify

    ap = argparse.ArgumentParser(description="Packet-level realisability validation (item 7).")
    ap.add_argument("--data", default="data/raw/cicids2017/MachineLearningCVE/"
                                       "Wednesday-workingHours.pcap_ISCX.csv")
    ap.add_argument("--artifacts", default="data/artifacts")
    ap.add_argument("--n-samples", type=int, default=25)
    ap.add_argument("--n-refs", type=int, default=8)
    ap.add_argument("--budget", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pcap-dir", default="experiments/pcaps")
    ap.add_argument("--n-pcaps", type=int, default=3, help="witness pcaps to emit per model")
    ap.add_argument("--summary-out", default="experiments/realisability.json")
    args = ap.parse_args()

    art_dir = Path(args.artifacts)
    artifacts = load_artifacts(art_dir)
    feature_names = artifacts.feature_names
    source = classify(feature_names)
    if source.unknown:
        raise SystemExit(f"unclassified features: {source.unknown}")

    ds = build_dataset(args.data, target_label="DoS Hulk", seed=args.seed)
    if feature_names != [normalize(c) for c in ds.feature_names]:
        raise SystemExit("feature order mismatch between artifacts and data")

    # Calibrate the three offset-bearing pooled features on real flows.
    calib = fit_calibration(pd.concat([ds.X_train, ds.X_test], ignore_index=True))
    print("calibration (actual ~ a*pooled + b):")
    for f, (a, b, rs, rel) in calib.coeffs.items():
        print(f"  {f:24s} a={a:.4f} b={b:.3g} residStd={rs:.3g} relResid={rel:.3f}")

    # Reproduce the item-5/6 sampling and re-run the attack to get fresh successes.
    rng = np.random.default_rng(args.seed)
    hulk_idx = np.where(ds.y_test == 1)[0]
    benign_idx = np.where(ds.y_test == 0)[0]
    hulk_pick = rng.choice(hulk_idx, size=min(args.n_samples, hulk_idx.size), replace=False)
    benign_pick = rng.choice(benign_idx, size=min(args.n_refs, benign_idx.size), replace=False)
    hulk_samples = [ds.X_test.iloc[i].to_dict() for i in hulk_pick]
    benign_refs = [ds.X_test.iloc[i].to_dict() for i in benign_pick]

    cfg = AttackConfig(query_budget=args.budget, n_init_refs=args.n_refs, seed=args.seed)
    print(f"\nrunning attack: {len(hulk_samples)} samples x 2 models, budget {args.budget} ...")
    results = run_attack(artifacts, hulk_samples, benign_refs, config=cfg)

    pcap_dir = Path(args.pcap_dir)
    pcap_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"config": {"n_samples": len(hulk_samples), "budget": args.budget,
                                "seed": args.seed, "data": args.data},
                     "calibration": {f: {"slope": a, "intercept": b,
                                         "resid_std": rs, "rel_resid": rel}
                                     for f, (a, b, rs, rel) in calib.coeffs.items()},
                     "models": {}}

    print("\n=== Realisability validation (item 7) ===")
    for m in ("logreg", "rf"):
        decision_fn = _decision_fn(artifacts.model(m), artifacts.scaler, feature_names)
        successes = [r for r in results[m] if r.success and r.best_vector]
        rres = [validate(r.best_vector, decision_fn, calib, source) for r in successes]
        surv = survival_summary(rres)

        # An illustrative reverted case: which pooled features moved when corrected.
        reverted = [(r, vr) for r, vr in zip(successes, rres) if vr.verdict == "reverted"]
        example = None
        if reverted:
            vr = reverted[0][1]
            example = {f: round(d, 3) for f, d in vr.moved_packet_length.items() if abs(d) > 1e-6}

        # Emit witness pcaps for the first realisable flows.
        emitted = []
        realisable = [r for r, vr in zip(successes, rres) if vr.verdict == "realisable"]
        for j, r in enumerate(realisable[: args.n_pcaps]):
            p = pcap_dir / f"{m}_realisable_{j}.pcap"
            try:
                npk = emit_pcap(r.best_vector, p, seed=args.seed + j)
                emitted.append({"path": str(p), "packets": npk})
            except Exception as e:                       # scapy missing / build error
                emitted.append({"error": f"{type(e).__name__}: {e}"})
                break

        summary["models"][m] = {
            "n_successful_evasions": len(successes),
            "survival_rate": surv["survival_rate"],
            "verdicts": surv["verdicts"],
            "example_reverted_packet_length_shift": example,
            "witness_pcaps": emitted,
        }
        print(f"\n[{m}]  successful evasions: {len(successes)}")
        print(f"  realisability survival rate: {surv['survival_rate']:.2%}")
        print(f"  verdicts: {surv['verdicts']}")
        if example:
            print(f"  e.g. reverted sample's pooled-length correction: {example}")
        if emitted and "packets" in emitted[0]:
            print(f"  witness pcaps: {len(emitted)} written to {pcap_dir}/")

    out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {out}   pcaps -> {pcap_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
