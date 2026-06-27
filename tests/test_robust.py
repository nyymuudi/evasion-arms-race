"""Tests for adversarial-training / arms-race primitives (Layer C)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evasion_arms_race.features.partition import PARTITION
from evasion_arms_race.features.rule_classifier import classify
from evasion_arms_race.attack.blackbox import AttackConfig
from evasion_arms_race.detector import robust as Rb


def _feasible_row(y: int, rng) -> dict:
    r = {f: 50.0 for f in sorted(PARTITION.all_features())}
    r.update({"Total Fwd Packets": 10.0, "Total Backward Packets": 8.0,
              "Flow Duration": 500_000.0,
              "Fwd Packet Length Min": 40.0, "Fwd Packet Length Max": 100.0,
              "Fwd Packet Length Mean": 60.0, "Fwd Packet Length Std": 20.0,
              "Total Length of Fwd Packets": 600.0,
              "Fwd IAT Min": 10.0, "Fwd IAT Max": 1000.0, "Fwd IAT Mean": 100.0,
              "Fwd IAT Std": 50.0, "Fwd IAT Total": 900.0})
    r["Idle Mean"] = float(rng.normal(20.0 if y == 0 else 80.0, 3.0))
    return r


def _frame(n, rng):
    feats = sorted(PARTITION.all_features())
    y = rng.choice([0, 1], size=n, p=[0.6, 0.4])
    X = pd.DataFrame([_feasible_row(yi, rng) for yi in y], columns=feats)
    return X, y


def test_train_one_learns_and_evaluates():
    rng = np.random.default_rng(0)
    Xtr, ytr = _frame(500, rng)
    Xte, yte = _frame(200, rng)
    det = Rb.train_one("logreg", Xtr, ytr, Xte, yte, seed=0)
    assert det.pr_auc > 0.9
    assert 0.0 <= det.hulk_recall <= 1.0
    assert det.feature_names == list(Xtr.columns)


def test_adversarial_trainset_empty_is_clean():
    rng = np.random.default_rng(0)
    Xc, yc = _frame(100, rng)
    Xtr, ytr = Rb.adversarial_trainset(Xc, yc, [], list(Xc.columns), replication=5)
    assert len(Xtr) == len(Xc)
    assert np.array_equal(ytr, yc)


def test_adversarial_trainset_appends_attack_labels():
    rng = np.random.default_rng(0)
    Xc, yc = _frame(100, rng)
    evasions = [_feasible_row(1, rng) for _ in range(4)]
    Xtr, ytr = Rb.adversarial_trainset(Xc, yc, evasions, list(Xc.columns), replication=5)
    assert len(Xtr) == len(Xc) + 4 * 5            # oversampled
    assert ytr[len(yc):].sum() == 4 * 5           # all appended labelled attack=1
    assert np.all(ytr[len(yc):] == 1)


def test_attack_detector_rates_are_consistent():
    rng = np.random.default_rng(0)
    Xtr, ytr = _frame(500, rng)
    Xte, yte = _frame(200, rng)
    det = Rb.train_one("logreg", Xtr, ytr, Xte, yte, seed=0)
    source = classify(det.feature_names)
    hulk = [_feasible_row(1, rng) for _ in range(3)]
    refs = [_feasible_row(0, rng) for _ in range(6)]
    cfg = AttackConfig(query_budget=200, n_init_refs=6, seed=0)
    out = Rb.attack_detector(det, hulk, refs, source, cfg)
    # realisable successes are a subset of feature-space successes
    assert 0.0 <= out.realisable_rate <= out.feature_success_rate <= 1.0
    assert len(out.evasions) == round(out.feature_success_rate * len(hulk))
    assert len(out.realisable_evasions) <= len(out.evasions)
