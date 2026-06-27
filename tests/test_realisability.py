"""Unit tests for packet-level realisability validation (Layer A, item 7)."""

from __future__ import annotations

import numpy as np
import pytest

from evasion_arms_race.features.partition import PARTITION
from evasion_arms_race.validation import realisability as R


def _clean_vector() -> dict[str, float]:
    """A full, internally consistent feature vector over the partition: both
    directions present, forward size/timing marginals satisfy the Bhatia-Davis
    bound (so it is feasible by default)."""
    v = {f: 50.0 for f in sorted(PARTITION.all_features())}
    v["Total Fwd Packets"] = 100.0
    v["Total Backward Packets"] = 80.0
    v["Flow Duration"] = 500_000.0
    # forward sizes: Var bound (max-mean)(mean-min) = 40*20 = 800 >= 20^2 = 400 OK
    v["Fwd Packet Length Min"] = 40.0
    v["Fwd Packet Length Max"] = 100.0
    v["Fwd Packet Length Mean"] = 60.0
    v["Fwd Packet Length Std"] = 20.0
    v["Total Length of Fwd Packets"] = 100.0 * 60.0
    # backward sizes (a 0-length ACK is present -> pooled min is 0)
    v["Bwd Packet Length Min"] = 0.0
    v["Bwd Packet Length Max"] = 1500.0
    v["Bwd Packet Length Mean"] = 200.0
    v["Bwd Packet Length Std"] = 100.0
    v["Total Length of Bwd Packets"] = 80.0 * 200.0
    # forward IAT consistent
    v["Fwd IAT Min"] = 10.0
    v["Fwd IAT Max"] = 1000.0
    v["Fwd IAT Mean"] = 100.0
    v["Fwd IAT Std"] = 50.0
    v["Fwd IAT Total"] = (100.0 - 1.0) * 100.0
    # pooled packet-length features as the attack would freeze them (clean-ish)
    v["Min Packet Length"] = 40.0
    v["Max Packet Length"] = 1500.0
    v["Packet Length Mean"] = 120.0
    v["Packet Length Std"] = 90.0
    v["Packet Length Variance"] = 8100.0
    v["Fwd PSH Flags"] = 3.0
    v["Destination Port"] = 80.0
    return v


def test_pooled_min_max_are_exact():
    v = _clean_vector()
    rec = R.reconstruct_packet_length_features(v)
    # min over both directions: min(40, 0) = 0; max: max(100, 1500) = 1500
    assert rec["Min Packet Length"] == 0.0
    assert rec["Max Packet Length"] == 1500.0


def test_pooled_mean_matches_law_of_total_variance():
    v = _clean_vector()
    rec = R.reconstruct_packet_length_features(v)
    nf, nb = 100.0, 80.0
    expected_mean = (nf * 60.0 + nb * 200.0) / (nf + nb)
    assert rec["Packet Length Mean"] == pytest.approx(expected_mean)
    # pooled variance via law of total variance, then sqrt for std
    ex2 = (nf * (20.0**2 + 60.0**2) + nb * (100.0**2 + 200.0**2)) / (nf + nb)
    expected_var = ex2 - expected_mean**2
    assert rec["Packet Length Variance"] == pytest.approx(expected_var)
    assert rec["Packet Length Std"] == pytest.approx(np.sqrt(expected_var))


def test_feasible_vector_passes():
    assert R.is_feasible(_clean_vector()).feasible


def test_size_variance_violation_is_infeasible():
    v = _clean_vector()
    v["Fwd Packet Length Std"] = 50.0   # 2500 > (100-60)(60-40)=800 -> impossible
    feas = R.forward_size_feasible(v)
    assert not feas.feasible
    assert any("variance" in r for r in feas.reasons)


def test_timing_iat_max_exceeds_duration_is_infeasible():
    v = _clean_vector()
    v["Fwd IAT Max"] = 2_000_000.0        # > Flow Duration 500_000
    assert not R.forward_timing_feasible(v).feasible


def test_corrected_vector_substitutes_the_five_features():
    v = _clean_vector()
    corr = R.corrected_vector(v)
    rec = R.reconstruct_packet_length_features(v)
    for f in rec:
        assert corr[f] == pytest.approx(rec[f])
    # untouched controllable feature survives
    assert corr["Fwd IAT Mean"] == pytest.approx(v["Fwd IAT Mean"])


def test_validate_reverted_when_correction_flips_detector():
    v = _clean_vector()                    # frozen Min Packet Length = 40 (benign)
    # detector: benign iff the pooled minimum length is >= 30
    decision = lambda d: 0 if d["Min Packet Length"] >= 30 else 1
    res = R.validate(v, decision)
    # corrected pooled min is 0 (a 0-length backward ACK), so the detector flips
    assert res.feasible
    assert res.verdict == "reverted"
    assert res.detector_attack_after_correction


def test_validate_realisable_when_detector_stays_benign():
    v = _clean_vector()
    decision = lambda d: 0                  # detector always benign
    res = R.validate(v, decision)
    assert res.verdict == "realisable"


def test_validate_infeasible_takes_precedence():
    v = _clean_vector()
    v["Fwd Packet Length Std"] = 50.0       # make it infeasible
    decision = lambda d: 0
    res = R.validate(v, decision)
    assert res.verdict == "infeasible"
    assert not res.feasible


def test_survival_summary_counts():
    mk = lambda verdict: R.RealisabilityResult(
        verdict=verdict, feasible=True, infeasibility_reasons=[],
        detector_attack_after_correction=False, robust_to_residual=True,
        reconstructed={}, moved_packet_length={})
    rs = [mk("realisable"), mk("realisable"), mk("reverted"), mk("infeasible")]
    s = R.survival_summary(rs)
    assert s["n"] == 4
    assert s["survival_rate"] == 0.5
    assert s["verdicts"]["reverted"] == 1


def test_manifold_project_output_is_always_feasible():
    v = _clean_vector()
    v["Fwd Packet Length Std"] = 5000.0     # variance bound violation
    v["Fwd IAT Max"] = 9_000_000.0          # IAT max > duration
    v["Fwd IAT Min"] = 8_000_000.0          # IAT min > max (ordering violation)
    v["Total Length of Fwd Packets"] = 1.0  # Total != N*mean
    mp = R.manifold_project(v, v).vector
    assert R.is_feasible(mp).feasible        # snapped onto the manifold by construction


def test_manifold_project_is_idempotent():
    v = _clean_vector()
    v["Fwd Packet Length Std"] = 5000.0
    once = R.manifold_project(v, v).vector
    twice = R.manifold_project(once, v).vector
    for f in ("Fwd Packet Length Std", "Total Length of Fwd Packets",
              "Fwd IAT Total", "Fwd Packets/s", "Flow Duration"):
        assert once[f] == pytest.approx(twice[f], rel=1e-9, abs=1e-6)


def test_manifold_project_preserves_dos_rate_floor():
    v = _clean_vector()
    v["Flow Duration"] = 5_000_000.0         # attacker tries to stretch the flow (slow the flood)
    mp = R.manifold_project(v, v).vector
    # forward packet rate must not fall below the clean sample's
    assert mp["Fwd Packets/s"] >= v["Fwd Packets/s"] - 1e-6


def test_emit_pcap_writes_packets(tmp_path):
    pytest.importorskip("scapy")
    v = _clean_vector()
    p = tmp_path / "witness.pcap"
    n = R.emit_pcap(v, p, seed=0)
    assert p.exists()
    assert n == int(round(v["Total Fwd Packets"]))
