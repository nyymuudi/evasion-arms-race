"""Black-box boundary-attack tests (Layer A, item 5).

These tests validate the two outcomes that matter scientifically, on synthetic
detectors whose separating signal is placed on a feature of a KNOWN control
class:

  A. Signal on a freely-movable CONTROLLABLE feature (Idle Mean): the attacker
     can lower it without the projection objecting -> evasion SUCCEEDS, and the
     resulting vector is feasible (frozen features equal the clean original).

  B. Signal on a floored CONSTRAINED feature (Fwd Packets/s): the only way to
     look benign is to drop the forward rate, which the DoS floor forbids. The
     un-projected candidate evades but the projection reverts it -> the attack
     fails with failure_mode == 'feasibility_bound' (NOT detector_bound), and the
     reverting constraint is attributed to Fwd Packets/s.

Plus an invariant test: every reported feasible vector respects the freeze.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evasion_arms_race.features.partition import PARTITION
from evasion_arms_race.data.loader import build_dataset
from evasion_arms_race.detector.baseline import train_baseline, save_artifacts
from evasion_arms_race.attack.blackbox import (
    AttackConfig,
    load_artifacts,
    Oracle,
    attack_sample,
)


def _make_csv(tmp_path, signal_feature: str, seed: int = 3) -> str:
    """Synthetic CICFlowMeter-shaped CSV: all features are class-independent
    noise EXCEPT `signal_feature`, which separates benign (~30) from attack
    (~70). That localises the detector's reliance onto one feature of a known
    control class.

    Crucially the DERIVED features (Flow Packets/s, Flow Bytes/s, Down/Up Ratio,
    Average Packet Size, Flow IAT Total) are made CONSISTENT with the atomic
    features, using the same formulas project() applies, and the atomic scales
    are realistic (durations in the 0.5 s range, not microseconds). Without this
    the projection's recompute would blow the rate features out of distribution
    and project(clean) would diverge from clean -- an artefact of synthetic data,
    not of the attack. Real CICFlowMeter rows already satisfy these identities.
    """
    feats = sorted(PARTITION.all_features())
    rng = np.random.default_rng(seed)
    n = 3000
    y = rng.choice([0, 1], size=n, p=[0.6, 0.4])
    data = {f: np.abs(rng.normal(50, 10, n)) for f in feats}

    # Realistic atomic scales so recomputed rates land in a sane range.
    data["Flow Duration"] = np.abs(rng.normal(500_000, 50_000, n)) + 1.0   # microseconds
    data["Total Fwd Packets"] = np.abs(rng.normal(100, 15, n)) + 1.0
    data["Total Backward Packets"] = np.abs(rng.normal(80, 15, n)) + 1.0
    data["Total Length of Fwd Packets"] = np.abs(rng.normal(8000, 1000, n))
    data["Total Length of Bwd Packets"] = np.abs(rng.normal(6000, 1000, n))

    # The separating signal (overrides whatever scale the feature had).
    data[signal_feature] = rng.normal(30, 3, n) + y * 40.0

    # Derived features, consistent with project()'s recompute rules.
    dur_s = data["Flow Duration"] / 1_000_000.0
    tot = data["Total Fwd Packets"] + data["Total Backward Packets"]
    bytes_tot = data["Total Length of Fwd Packets"] + data["Total Length of Bwd Packets"]
    data["Flow IAT Total"] = data["Flow Duration"]
    data["Down/Up Ratio"] = data["Total Backward Packets"] / data["Total Fwd Packets"]
    data["Flow Packets/s"] = tot / dur_s
    data["Flow Bytes/s"] = bytes_tot / dur_s
    data["Average Packet Size"] = bytes_tot / tot

    df = pd.DataFrame(data)
    df.insert(len(df.columns), "Fwd Header Length.1", df["Fwd Header Length"])
    df[" Label"] = np.where(y == 1, "DoS Hulk", "BENIGN")
    path = tmp_path / f"{signal_feature.replace('/', '_').replace(' ', '_')}.csv"
    df.to_csv(path, index=False)
    return str(path)


def _prepare(tmp_path, signal_feature):
    """Build dataset, train + persist detectors, load artifacts, and pull one
    Hulk sample + a handful of benign references from the test split."""
    csv = _make_csv(tmp_path, signal_feature)
    ds = build_dataset(csv, target_label="DoS Hulk", seed=0)
    tb = train_baseline(ds, seed=0)
    out = tmp_path / "artifacts"
    save_artifacts(tb, out_dir=out)
    artifacts = load_artifacts(out)

    hulk_rows = np.where(ds.y_test == 1)[0]
    benign_rows = np.where(ds.y_test == 0)[0]
    clean = ds.X_test.iloc[hulk_rows[0]].to_dict()
    refs = [ds.X_test.iloc[i].to_dict() for i in benign_rows[:8]]
    return artifacts, clean, refs


def test_controllable_signal_is_evadable(tmp_path):
    """Signal on a freely-movable controllable feature -> LR is evaded with a
    feasible perturbation, within budget."""
    artifacts, clean, refs = _prepare(tmp_path, "Idle Mean")
    oracle = Oracle(artifacts.logreg, artifacts.scaler, artifacts.feature_names)
    cfg = AttackConfig(query_budget=300, n_init_refs=8, seed=0)
    res = attack_sample(clean, oracle, refs, cfg)

    assert res.init_found
    assert res.success
    assert res.failure_mode == "success"
    assert res.feasible_queries <= cfg.query_budget + 2
    # diagnostic queries shadow feasible queries one-for-one when attributing.
    assert res.diagnostic_queries == res.feasible_queries
    # a non-trivial perturbation was needed on the controllable signal axis.
    assert res.l2_controllable > 0.0


def test_feasible_vector_respects_freeze(tmp_path):
    """The reported best vector must leave frozen (server-side) features at their
    clean values -- the cardinal feasibility invariant."""
    artifacts, clean, refs = _prepare(tmp_path, "Idle Mean")
    oracle = Oracle(artifacts.logreg, artifacts.scaler, artifacts.feature_names)
    res = attack_sample(clean, oracle, refs, AttackConfig(query_budget=300, seed=0))
    assert res.success
    for f in PARTITION.frozen:
        assert res.best_vector[f] == pytest.approx(clean[f], rel=1e-4, abs=1e-3)


def test_dos_floor_blocks_evasion_is_attributed(tmp_path):
    """Signal on the floored forward rate -> evasion fails, and the failure is
    correctly attributed to the feasibility floor, not to detector strength."""
    artifacts, clean, refs = _prepare(tmp_path, "Fwd Packets/s")
    oracle = Oracle(artifacts.logreg, artifacts.scaler, artifacts.feature_names)
    cfg = AttackConfig(query_budget=300, n_init_refs=8, seed=0)
    res = attack_sample(clean, oracle, refs, cfg)

    assert not res.success
    assert res.failure_mode == "feasibility_bound"
    # the projection saw un-projected candidates that DID evade ...
    assert res.flip_attempts > 0
    # ... but none survived projection ...
    assert res.flips_survived == 0
    assert res.floor_blocked > 0
    # ... and the forward-rate floor is among the reverting constraints.
    assert "Fwd Packets/s" in res.blocking_features


def test_already_benign_sample_is_reported_not_attacked(tmp_path):
    """A boundary attack is meaningless if the detector already calls the source
    benign. Attacking a benign sample must short-circuit to 'already_benign'
    rather than spend the budget pretending to evade."""
    artifacts, _clean, refs = _prepare(tmp_path, "Idle Mean")
    # refs come from benign rows already; reuse the first as the source.
    benign_source = refs[0]
    oracle = Oracle(artifacts.logreg, artifacts.scaler, artifacts.feature_names)
    res = attack_sample(benign_source, oracle, refs[1:], AttackConfig(query_budget=300, seed=0))
    assert res.failure_mode == "already_benign"
    assert res.feasible_queries == 1   # only the sanity query was spent
