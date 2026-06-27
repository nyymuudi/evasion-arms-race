"""Tests for realisability-constrained poisoning (Layer B)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evasion_arms_race.features.partition import PARTITION
from evasion_arms_race.data.loader import Dataset
from evasion_arms_race.attack import poisoning as P


def _feasible_row(y: int, rng) -> dict:
    """A row with internally consistent forward marginals (passes is_feasible),
    separable on Idle Mean (benign ~20, Hulk ~80)."""
    r = {f: 50.0 for f in sorted(PARTITION.all_features())}
    r["Total Fwd Packets"] = 10.0
    r["Total Backward Packets"] = 8.0
    r["Flow Duration"] = 500_000.0
    r["Fwd Packet Length Min"] = 40.0
    r["Fwd Packet Length Max"] = 100.0
    r["Fwd Packet Length Mean"] = 60.0
    r["Fwd Packet Length Std"] = 20.0
    r["Total Length of Fwd Packets"] = 600.0
    r["Fwd IAT Min"] = 10.0
    r["Fwd IAT Max"] = 1000.0
    r["Fwd IAT Mean"] = 100.0
    r["Fwd IAT Std"] = 50.0
    r["Fwd IAT Total"] = 900.0
    r["Idle Mean"] = float(rng.normal(20.0 if y == 0 else 80.0, 3.0))
    return r


def _synthetic_dataset(n_train=600, n_test=200, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    feats = sorted(PARTITION.all_features())

    def block(n):
        y = rng.choice([0, 1], size=n, p=[0.6, 0.4])
        X = pd.DataFrame([_feasible_row(yi, rng) for yi in y], columns=feats)
        return X, y

    Xtr, ytr = block(n_train)
    Xte, yte = block(n_test)
    return Dataset(X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte,
                   feature_names=feats, target_label="DoS Hulk", n_dropped_nonfinite=0)


def test_generate_poison_label_flip_counts_and_labels():
    ds = _synthetic_dataset()
    Xp, yp, info = P.generate_poison(ds, n_poison=20, strategy="label_flip", seed=0)
    assert len(Xp) == 20
    assert list(Xp.columns) == ds.feature_names
    assert np.all(yp == 0)                       # all labelled benign
    assert info["skipped_infeasible"] == 0       # real feasible rows pass


def test_generate_poison_boundary_selects_near_boundary():
    ds = _synthetic_dataset()
    # scorer: attack-probability proxy increases with Idle Mean (the signal)
    scorer = lambda df: df["Idle Mean"].to_numpy()
    Xb, _, _ = P.generate_poison(ds, n_poison=20, strategy="boundary",
                                 scorer=scorer, seed=0)
    Xr, _, _ = P.generate_poison(ds, n_poison=20, strategy="label_flip", seed=0)
    # boundary picks the lowest-scoring (nearest benign) Hulk samples
    assert Xb["Idle Mean"].mean() < Xr["Idle Mean"].mean()


def test_generate_poison_skips_infeasible():
    ds = _synthetic_dataset(n_train=200, n_test=50)
    # corrupt one Hulk row into an infeasible one (variance bound violation)
    hulk = np.where(ds.y_train == 1)[0]
    ds.X_train.loc[hulk[0], "Fwd Packet Length Std"] = 5000.0
    Xp, yp, info = P.generate_poison(ds, n_poison=len(hulk), strategy="label_flip", seed=0)
    assert info["skipped_infeasible"] >= 1
    assert len(Xp) == info["kept"]


def test_subsample_clean_is_stratified():
    ds = _synthetic_dataset(n_train=600)
    Xc, yc = P.subsample_clean(ds, clean_train_size=200, seed=0)
    assert 180 <= len(yc) <= 220
    assert set(np.unique(yc)) == {0, 1}


def test_inject_keeps_test_clean():
    ds = _synthetic_dataset()
    Xc, yc = P.subsample_clean(ds, clean_train_size=200, seed=0)
    Xp, yp, _ = P.generate_poison(ds, n_poison=20, strategy="label_flip", seed=0)
    poisoned = P.inject(Xc, yc, Xp, yp, ds)
    assert len(poisoned.y_train) == len(yc) + 20
    assert poisoned.X_test.equals(ds.X_test)     # test never touched
    assert np.array_equal(poisoned.y_test, ds.y_test)


def test_run_sweep_smoke():
    ds = _synthetic_dataset(n_train=600, n_test=200)
    out = P.run_sweep(ds, fractions=(0.0, 0.1), strategies=("label_flip",),
                      clean_train_size=300, seed=0)
    pts = out["results"]["label_flip"]
    assert [p.fraction for p in pts] == [0.0, 0.1]
    assert pts[0].n_poison == 0 and pts[1].n_poison > 0
    for p in pts:
        assert 0.0 <= p.pr_auc_logreg <= 1.0
        assert 0.0 <= p.pr_auc_rf <= 1.0
    assert "logreg" in out["reference_pr_auc"]
