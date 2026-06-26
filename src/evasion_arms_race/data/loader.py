"""Dataset loading + cleaning for CIC-IDS2017 (CICFlowMeter CSVs), with a path
to LYCOS-IDS2017 later.

Responsibilities
----------------
  - read CSV, normalise header (strip leading spaces, collapse whitespace) using
    the SAME normalize() as the feature partition, so headers are handled
    identically everywhere in the project (single source of truth)
  - drop the CICFlowMeter duplicate column 'Fwd Header Length.1'
  - filter to a binary problem: target attack class vs BENIGN
  - clean non-finite values (Inf / NaN), which CIC-IDS2017 emits in rate columns
    (Flow Bytes/s, Flow Packets/s) when Flow Duration == 0
  - produce a stratified, seeded train/test split

Split policy
------------
The MachineLearningCSV distribution has NO Timestamp column, so a genuine
TEMPORAL split (train on earlier flows, test on later) is not possible here.
We therefore use a stratified random split with a fixed seed. This is adequate
for building a target detector for the evasion study, but it does NOT exercise
concept drift. A temporal split would require the GeneratedLabelledFlows
distribution (which retains timestamps); that is deferred and documented as a
known limitation rather than silently ignored.

Target class note
-----------------
Default target is 'DoS Hulk'. The Wednesday capture also contains GoldenEye,
slowloris, Slowhttptest and Heartbleed. Hulk is a HIGH-volume HTTP flood and is
likely easy to separate from benign; if it proves trivially separable, the
low-volume variants (slowloris / Slowhttptest) sit closer to the benign manifold
and may make a more interesting evasion target. Changing the target is a single
parameter -- the loader does not hard-code Hulk anywhere except the default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evasion_arms_race.features.partition import normalize

DUPLICATE_COLUMNS = ["Fwd Header Length.1"]
LABEL_CANDIDATES = ("Label", "label")
BENIGN_LABEL = "BENIGN"


@dataclass
class Dataset:
    """A loaded, cleaned, split binary dataset.

    X_* are feature frames (normalised column names, duplicate dropped, label
    removed). y_* are 0/1 arrays (1 = attack/positive, 0 = benign).
    feature_names is the ordered list of feature columns shared by X_train/X_test.
    """
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_label: str
    n_dropped_nonfinite: int


def _find_label_column(columns: list[str]) -> str:
    """Return the normalised label column name, tolerating the leading-space
    quirk (' Label')."""
    norm = {normalize(c): c for c in columns}
    for cand in LABEL_CANDIDATES:
        if cand in norm:
            return cand
    raise KeyError(
        f"No label column found among {LABEL_CANDIDATES}; "
        f"normalised columns include: {sorted(norm)[:5]}..."
    )


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load one CICFlowMeter CSV with normalised headers and dropped duplicate.

    Non-finite cleaning and class filtering happen in build_dataset(); this
    function only handles I/O + header hygiene so it is reusable.
    """
    df = pd.read_csv(path, low_memory=False)
    df.columns = [normalize(c) for c in df.columns]
    for dup in DUPLICATE_COLUMNS:
        if dup in df.columns:
            df = df.drop(columns=[dup])
    return df


def build_dataset(
    path: str | Path,
    target_label: str = "DoS Hulk",
    test_size: float = 0.25,
    seed: int = 0,
) -> Dataset:
    """Load, filter to {target_label, BENIGN}, clean, and split.

    Parameters
    ----------
    path        : CSV path (e.g. the Wednesday capture for DoS Hulk)
    target_label: positive class; default 'DoS Hulk'
    test_size   : fraction held out for testing
    seed        : RNG seed for the stratified split (reproducibility)
    """
    df = load_csv(path)
    label_col = _find_label_column(list(df.columns))

    # Binary filter: keep only target vs benign.
    mask = df[label_col].isin([target_label, BENIGN_LABEL])
    df = df.loc[mask].copy()
    present = set(df[label_col].unique())
    if target_label not in present:
        raise ValueError(
            f"Target {target_label!r} not found in {label_col!r}. "
            f"Present labels (after benign filter): {sorted(present)}. "
            f"Check the spelling against the file's actual values."
        )
    if BENIGN_LABEL not in present:
        raise ValueError(
            f"{BENIGN_LABEL!r} not found; cannot build a binary problem."
        )

    y = (df[label_col] == target_label).astype(int).to_numpy()
    X = df.drop(columns=[label_col])

    # Coerce to numeric; CIC CSVs occasionally carry stray strings.
    X = X.apply(pd.to_numeric, errors="coerce")

    # Clean non-finite: replace +/-Inf with NaN, then drop rows with any NaN.
    before = len(X)
    X = X.replace([np.inf, -np.inf], np.nan)
    finite_mask = X.notna().all(axis=1)
    X = X.loc[finite_mask]
    y = y[finite_mask.to_numpy()]
    n_dropped = before - len(X)

    feature_names = list(X.columns)

    # Stratified, seeded split (no sklearn dependency needed for this).
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    test_idx = np.concatenate([
        rng.choice(idx[y == cls],
                   size=int(round(test_size * (y == cls).sum())),
                   replace=False)
        for cls in (0, 1)
    ])
    test_mask = np.zeros(len(X), dtype=bool)
    test_mask[test_idx] = True

    X_arr = X.reset_index(drop=True)
    return Dataset(
        X_train=X_arr.loc[~test_mask].reset_index(drop=True),
        X_test=X_arr.loc[test_mask].reset_index(drop=True),
        y_train=y[~test_mask],
        y_test=y[test_mask],
        feature_names=feature_names,
        target_label=target_label,
        n_dropped_nonfinite=n_dropped,
    )