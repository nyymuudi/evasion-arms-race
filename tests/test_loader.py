"""Loader tests on synthetic CIC-shaped data (no real dataset needed in CI)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evasion_arms_race.features.partition import PARTITION
from evasion_arms_race.data.loader import build_dataset, load_csv


@pytest.fixture
def synthetic_csv(tmp_path):
    feats = sorted(PARTITION.all_features())
    rng = np.random.default_rng(1)
    n = 2000
    df = pd.DataFrame({f: rng.normal(50, 10, n) for f in feats})
    df.insert(len(df.columns), "Fwd Header Length.1", df["Fwd Header Length"])
    df[" Label"] = rng.choice(["BENIGN", "DoS Hulk", "DoS GoldenEye"],
                              size=n, p=[0.45, 0.45, 0.10])
    df.loc[rng.choice(n, 15, replace=False), "Flow Bytes/s"] = np.inf
    df.loc[rng.choice(n, 10, replace=False), "Flow Packets/s"] = np.nan
    path = tmp_path / "wednesday.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_header_normalised_and_duplicate_dropped(synthetic_csv):
    raw = load_csv(synthetic_csv)
    assert "Fwd Header Length.1" not in raw.columns
    assert " Label" not in raw.columns and "Label" in raw.columns


def test_binary_filter_and_cleaning(synthetic_csv):
    ds = build_dataset(synthetic_csv, target_label="DoS Hulk", seed=0)
    assert set(np.unique(ds.y_train)) <= {0, 1}
    assert set(np.unique(ds.y_test)) <= {0, 1}
    assert np.isfinite(ds.X_train.to_numpy()).all()
    assert np.isfinite(ds.X_test.to_numpy()).all()
    assert ds.n_dropped_nonfinite > 0
    assert "Label" not in ds.feature_names
    assert len(ds.feature_names) == 78


def test_stratification_preserves_class_balance(synthetic_csv):
    ds = build_dataset(synthetic_csv, target_label="DoS Hulk", seed=0)
    assert abs(ds.y_train.mean() - ds.y_test.mean()) < 0.02


def test_missing_target_raises(synthetic_csv):
    with pytest.raises(ValueError):
        build_dataset(synthetic_csv, target_label="Nonexistent Attack", seed=0)