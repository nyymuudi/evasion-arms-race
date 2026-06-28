"""Temporal split + metadata handling (Phase 0: closing the temporal-split debt).

Validates the three things that can silently corrupt a temporal evaluation on
CIC-IDS2017: the non-UTF-8/12h/day-first timestamp parse, the dropping of
collection-artefact metadata columns, and that the temporal cut keeps both
classes on both sides (a global early/late cut would not, because DoS Hulk is
time-localised).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evasion_arms_race.data import loader as L


def test_timestamp_parse_is_dayfirst_and_pm_aware():
    s = pd.Series(["5/7/2017 8:42", "5/7/2017 3:22", "5/7/2017 12:10", "5/7/2017 1:19"])
    t = L._parse_cic_timestamp(s)
    # day-first: 5 July, not 7 May
    assert t.iloc[0].month == 7 and t.iloc[0].day == 5
    # 8:42 stays AM; 3:22 -> 15:22 (PM); 12:10 stays noon; 1:19 -> 13:19
    assert list(t.dt.hour) == [8, 15, 12, 13]
    # the PM correction orders 3:22 (15:22) after 8:42 -- not before, as a naive parse would
    assert t.iloc[1] > t.iloc[0]


def test_timestamp_parse_rejects_non_daytime():
    # an all-midnight series would mean the heuristic does not hold -> must raise
    with pytest.raises(ValueError):
        L._parse_cic_timestamp(pd.Series(["5/7/2017 0:01"] * 100))


def _cic_csv(tmp_path):
    """Minimal CIC-style CSV: metadata + timestamp + two features + label, with
    DoS Hulk localised around 11:00 and benign spread across the day."""
    rng = np.random.default_rng(0)
    times_benign = (["5/7/2017 8:%02d" % m for m in range(0, 30)]
                    + ["5/7/2017 2:%02d" % m for m in range(0, 30)])      # 8am + 2pm
    times_hulk = ["5/7/2017 10:55", "5/7/2017 11:00", "5/7/2017 11:05"] * 10
    rows = []
    for ts in times_benign:
        rows.append((ts, "BENIGN"))
    for ts in times_hulk:
        rows.append((ts, "DoS Hulk"))
    n = len(rows)
    df = pd.DataFrame({
        "Flow ID": [f"id{i}" for i in range(n)],
        " Source IP": ["10.0.0.1"] * n,
        " Source Port": rng.integers(1024, 65535, n),
        " Destination IP": ["10.0.0.2"] * n,
        " Protocol": [6] * n,
        " Destination Port": [80] * n,
        " Timestamp": [r[0] for r in rows],
        " Flow Duration": rng.integers(1, 1000, n),
        " Total Fwd Packets": rng.integers(1, 50, n),
        " Label": [r[1] for r in rows],
    })
    p = tmp_path / "wed.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_metadata_columns_are_dropped_but_destination_port_kept(tmp_path):
    ds = L.build_dataset(_cic_csv(tmp_path), target_label="DoS Hulk", split="stratified")
    feats = set(ds.feature_names)
    for artefact in ("Flow ID", "Source IP", "Source Port", "Destination IP",
                     "Protocol", "Timestamp"):
        assert artefact not in feats
    assert "Destination Port" in feats          # genuine feature, kept


def test_temporal_split_keeps_both_classes_on_both_sides(tmp_path):
    ds = L.build_dataset(_cic_csv(tmp_path), target_label="DoS Hulk", split="temporal")
    # the cut is at the Hulk median (11:00), so train has the 10:55 burst start and
    # test the 11:00/11:05 remainder -- both classes present on both sides
    assert ds.y_train.sum() > 0 and (ds.y_train == 0).sum() > 0
    assert ds.y_test.sum() > 0 and (ds.y_test == 0).sum() > 0


def test_temporal_split_requires_timestamp(tmp_path):
    # a CSV without Timestamp cannot be split temporally
    df = pd.DataFrame({" Destination Port": [80, 80, 80, 80],
                       " Flow Duration": [1, 2, 3, 4],
                       " Label": ["DoS Hulk", "BENIGN", "DoS Hulk", "BENIGN"]})
    p = tmp_path / "no_ts.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError):
        L.build_dataset(p, target_label="DoS Hulk", split="temporal")
