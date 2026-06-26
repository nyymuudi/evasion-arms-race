"""Ablation must detect an artifactual single-feature shortcut."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evasion_arms_race.features.partition import PARTITION
from evasion_arms_race.data.loader import build_dataset
from evasion_arms_race.detector.ablation import run_ablations


@pytest.fixture
def port_artifact_csv(tmp_path):
    feats = sorted(PARTITION.all_features())
    rng = np.random.default_rng(2)
    n = 3000
    y = rng.choice([0, 1], size=n, p=[0.66, 0.34])
    data = {}
    for f in feats:
        base = rng.normal(50, 10, n)
        if f == "Destination Port":
            base = base + y * rng.normal(60, 5, n)  # all signal here
        data[f] = base
    df = pd.DataFrame(data)
    df.insert(len(df.columns), "Fwd Header Length.1", df["Fwd Header Length"])
    df[" Label"] = np.where(y == 1, "DoS Hulk", "BENIGN")
    path = tmp_path / "port.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_ablation_flags_port_artifact(port_artifact_csv):
    ds = build_dataset(port_artifact_csv, target_label="DoS Hulk", seed=0)
    res = {r.name: r for r in run_ablations(ds, seed=0)}
    # Dropping the port should crater PR-AUC for both models.
    assert res["drop_destination_port"].delta_logreg < -0.3
    assert res["drop_destination_port"].delta_rf < -0.3
    # A non-signal feature should barely matter.
    nonsig = [k for k in res if k.startswith("drop::") and "Destination Port" not in k]
    assert any(abs(res[k].delta_logreg) < 0.05 for k in nonsig)