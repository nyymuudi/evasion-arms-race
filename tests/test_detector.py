"""Detector tests on synthetic data with a learnable, separable signal."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evasion_arms_race.features.partition import PARTITION
from evasion_arms_race.data.loader import build_dataset
from evasion_arms_race.detector.baseline import train_baseline, save_artifacts


@pytest.fixture
def separable_csv(tmp_path):
    feats = sorted(PARTITION.all_features())
    rng = np.random.default_rng(2)
    n = 4000
    y = rng.choice([0, 1], size=n, p=[0.66, 0.34])
    data = {}
    for f in feats:
        base = rng.normal(50, 10, n)
        if f in ("Flow Packets/s", "Fwd Packets/s", "Total Fwd Packets", "Flow Bytes/s"):
            base = base + y * rng.normal(40, 8, n)
        data[f] = base
    df = pd.DataFrame(data)
    df.insert(len(df.columns), "Fwd Header Length.1", df["Fwd Header Length"])
    df[" Label"] = np.where(y == 1, "DoS Hulk", "BENIGN")
    path = tmp_path / "sep.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_both_models_learn(separable_csv):
    ds = build_dataset(separable_csv, target_label="DoS Hulk", seed=0)
    tb = train_baseline(ds, seed=0, top_k=6)
    # With injected signal both should be well above chance PR-AUC (pos frac ~0.34)
    assert tb.eval_logreg.pr_auc > 0.9
    assert tb.eval_rf.pr_auc > 0.9


def test_top_features_are_the_signal(separable_csv):
    ds = build_dataset(separable_csv, target_label="DoS Hulk", seed=0)
    tb = train_baseline(ds, seed=0, top_k=4)
    signal = {"Flow Packets/s", "Fwd Packets/s", "Total Fwd Packets", "Flow Bytes/s"}
    lr_top = {name for name, _ in tb.top_logreg.ranked}
    rf_top = {name for name, _ in tb.top_rf.ranked}
    # the injected signal features should dominate both rankings
    assert len(signal & lr_top) >= 3
    assert len(signal & rf_top) >= 3


def test_artifacts_roundtrip(separable_csv, tmp_path):
    import pickle, json
    ds = build_dataset(separable_csv, target_label="DoS Hulk", seed=0)
    tb = train_baseline(ds, seed=0)
    out = tmp_path / "artifacts"
    save_artifacts(tb, out_dir=out)
    assert (out / "scaler.pkl").exists()
    assert (out / "logreg.pkl").exists()
    assert (out / "rf.pkl").exists()
    names = json.loads((out / "feature_names.json").read_text())
    assert len(names) == 78
    with open(out / "logreg.pkl", "rb") as f:
        pickle.load(f)  # unpickles without error