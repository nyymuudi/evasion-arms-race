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
Two split policies are supported via `split=`:
  * 'stratified' (default): stratified random split, fixed seed. Adequate for the
    evasion study but does NOT exercise concept drift.
  * 'temporal': train on earlier flows, test on later, using the parsed Timestamp.
    Requires the GeneratedLabelledFlows distribution (the MachineLearningCSV one
    has no Timestamp). CAVEAT specific to this dataset: DoS Hulk is a single
    ~24-minute burst (10:43-11:07 on Wednesday), so a global early/late cut puts
    100% of the attack in train (test gets zero positives). The temporal cut is
    therefore placed at the MEDIAN timestamp of the TARGET class, which keeps
    attack flows on both sides; the resulting test fraction is whatever falls
    after that instant, not `test_size`. See docs/ for the full analysis.

Metadata / collection artefacts
-------------------------------
The GeneratedLabelledFlows distribution carries six metadata columns. Five are
DROPPED as catastrophic collection artefacts (a model must not learn "this IP =
attack"): Flow ID, Source IP, Source Port, Destination IP, Protocol. Destination
Port is KEPT (a genuine, ablation-confirmed feature). Timestamp is used for the
temporal split and then dropped, so both split policies see the SAME 78 features.

Encoding
--------
read is robust: utf-8 first, latin-1 fallback (the GeneratedLabelledFlows CSVs
are not always UTF-8). The Wednesday file happens to be UTF-8; others may not be.

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
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from evasion_arms_race.features.partition import normalize

DUPLICATE_COLUMNS = ["Fwd Header Length.1"]
# Collection artefacts that MUST NOT become model features (a detector would learn
# "this IP/flow == attack"). Destination Port is intentionally NOT here -- it is a
# genuine feature (ablation confirmed it is not an artefact). Timestamp is handled
# separately (used for the temporal split, then dropped).
METADATA_COLUMNS = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Protocol"]
TIMESTAMP_COLUMN = "Timestamp"
LABEL_CANDIDATES = ("Label", "label")
BENIGN_LABEL = "BENIGN"


def _read_csv_robust(path: str | Path) -> pd.DataFrame:
    """utf-8 first, latin-1 fallback. CIC distributions are inconsistently encoded;
    guessing wrong yields silent garbage, so we try strict utf-8 then fall back."""
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    # last resort: let pandas raise with the default so the error is visible
    return pd.read_csv(path, low_memory=False)


def _parse_cic_timestamp(series: pd.Series) -> pd.Series:
    """Parse CIC-IDS2017 timestamps WITHOUT guessing.

    The format is day-first 'D/M/YYYY H:MM[:SS]' with a 12-HOUR clock and NO AM/PM
    marker (a documented CIC quirk). pandas.to_datetime would (a) read the date
    month-first and (b) leave afternoon times in the morning -- both silently
    mis-ordering the data. The captures run during working hours, so hours 1-7 are
    PM (+12) and 8-12 stay as-is. We validate the result lands in a daytime window
    and raise otherwise, rather than silently producing a wrong order.
    """
    def one(s: str) -> datetime:
        dpart, tpart = s.strip().split(" ", 1)
        d, m, y = (int(x) for x in dpart.split("/"))      # day-first
        tp = tpart.split(":")
        hh, mm = int(tp[0]), int(tp[1])
        ss = int(tp[2]) if len(tp) > 2 else 0
        if 1 <= hh <= 7:                                   # working-hours 12h -> 24h
            hh += 12
        return datetime(y, m, d, hh, mm, ss)

    out = pd.to_datetime(series.astype(str).map(one))
    if ((out.dt.hour < 6) | (out.dt.hour > 22)).mean() > 0.02:   # sanity: daytime capture
        raise ValueError(
            "parsed timestamps fall outside a daytime window -- the 12h/AM-PM "
            "heuristic likely does not hold for this distribution; inspect before use."
        )
    return out


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
    """Load one CIC CSV with normalised headers, robust encoding, the duplicate
    column dropped, and metadata collection-artefact columns dropped. Timestamp is
    kept (build_dataset uses it for the temporal split, then drops it).

    Non-finite cleaning and class filtering happen in build_dataset(); this
    function only handles I/O + header hygiene so it is reusable.
    """
    df = _read_csv_robust(path)
    df.columns = [normalize(c) for c in df.columns]
    for col in DUPLICATE_COLUMNS + METADATA_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def build_dataset(
    path: str | Path,
    target_label: str = "DoS Hulk",
    test_size: float = 0.25,
    seed: int = 0,
    split: str = "stratified",
) -> Dataset:
    """Load, filter to {target_label, BENIGN}, clean, and split.

    Parameters
    ----------
    path        : CSV path (e.g. the Wednesday capture for DoS Hulk)
    target_label: positive class; default 'DoS Hulk'
    test_size   : fraction held out for the STRATIFIED split (ignored by temporal)
    seed        : RNG seed for the stratified split (reproducibility)
    split       : 'stratified' (default) or 'temporal'. Temporal needs a Timestamp
                  column and cuts at the target class's median timestamp (see the
                  module docstring for why a global early/late cut is degenerate).
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

    # Pull the timestamp aside (for the temporal split) and drop it + the label
    # from the features, so both split policies see the SAME feature set.
    ts_raw = df[TIMESTAMP_COLUMN] if TIMESTAMP_COLUMN in df.columns else None
    drop_cols = [label_col] + ([TIMESTAMP_COLUMN] if ts_raw is not None else [])
    X = df.drop(columns=drop_cols)

    # Coerce to numeric; CIC CSVs occasionally carry stray strings.
    X = X.apply(pd.to_numeric, errors="coerce")

    # Clean non-finite: replace +/-Inf with NaN, then drop rows with any NaN.
    before = len(X)
    X = X.replace([np.inf, -np.inf], np.nan)
    finite_mask = X.notna().all(axis=1)
    X = X.loc[finite_mask]
    y = y[finite_mask.to_numpy()]
    if ts_raw is not None:
        ts_raw = ts_raw.loc[finite_mask]
    n_dropped = before - len(X)

    feature_names = list(X.columns)
    X_arr = X.reset_index(drop=True)

    if split == "temporal":
        if ts_raw is None:
            raise ValueError("temporal split requires a Timestamp column (use the "
                             "GeneratedLabelledFlows distribution)")
        t = _parse_cic_timestamp(ts_raw).reset_index(drop=True)
        # Cut at the TARGET class's median timestamp: a global early/late cut would
        # put 100% of a time-localised attack (DoS Hulk = a ~24-min burst) in train.
        t_target = t[y == 1]
        t_star = t_target.sort_values().iloc[len(t_target) // 2]
        test_mask = (t.to_numpy() >= np.datetime64(t_star))
    elif split == "stratified":
        rng = np.random.default_rng(seed)
        idx = np.arange(len(X_arr))
        test_idx = np.concatenate([
            rng.choice(idx[y == cls],
                       size=int(round(test_size * (y == cls).sum())),
                       replace=False)
            for cls in (0, 1)
        ])
        test_mask = np.zeros(len(X_arr), dtype=bool)
        test_mask[test_idx] = True
    else:
        raise ValueError(f"unknown split {split!r} (use 'stratified' or 'temporal')")

    return Dataset(
        X_train=X_arr.loc[~test_mask].reset_index(drop=True),
        X_test=X_arr.loc[test_mask].reset_index(drop=True),
        y_train=y[~test_mask],
        y_test=y[test_mask],
        feature_names=feature_names,
        target_label=target_label,
        n_dropped_nonfinite=n_dropped,
    )