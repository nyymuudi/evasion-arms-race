"""Ablation diagnostics for the baseline detector.

Purpose
-------
A near-perfect detector (PR-AUC > 0.999) is as much a warning as a result: it
may have found a collection/environment artifact rather than a genuine attack
signature. Before trusting any evasion result, we must know WHAT the detector
relies on and whether that thing is something the attacker can actually touch.

Three ablations, each retraining from scratch on a reduced feature set:

  1. drop_destination_port : 'Destination Port' alone. Hulk hits the HTTP port,
     so the port can be a cheap near-perfect separator. If PR-AUC barely moves
     without it, good (the detector has other signal); if it collapses, the port
     was an artifactual shortcut and evasion over it would be meaningless.

  2. controllable_only : keep ONLY features the attacker controls (controllable
     + constrained), dropping every FROZEN feature. This is the key test: if the
     detector still separates well, evasion is genuinely hard and meaningful
     (the attacker must move the very features the detector reads). If it
     collapses, the detector's power lives in features the attacker cannot
     change -- a fundamental limit on what evasion can achieve, worth knowing.

  3. drop_each_top : remove each top-ranked feature individually and measure the
     PR-AUC drop. A single feature whose removal craters performance is a
     single point of failure (often an artifact).

Reports PR-AUC deltas relative to the full-feature baseline for both models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from evasion_arms_race.data.loader import Dataset
from evasion_arms_race.detector.baseline import train_baseline
from evasion_arms_race.features.rule_classifier import classify
from evasion_arms_race.features.partition import Control


def _subset_dataset(ds: Dataset, keep: list[str]) -> Dataset:
    """Return a copy of ds restricted to `keep` feature columns."""
    return Dataset(
        X_train=ds.X_train[keep].copy(),
        X_test=ds.X_test[keep].copy(),
        y_train=ds.y_train,
        y_test=ds.y_test,
        feature_names=list(keep),
        target_label=ds.target_label,
        n_dropped_nonfinite=ds.n_dropped_nonfinite,
    )


@dataclass
class AblationResult:
    name: str
    n_features: int
    pr_auc_logreg: float
    pr_auc_rf: float
    delta_logreg: float   # vs full-feature baseline
    delta_rf: float


def run_ablations(ds: Dataset, seed: int = 0) -> list[AblationResult]:
    results: list[AblationResult] = []

    # Full baseline.
    full = train_baseline(ds, seed=seed, top_k=10)
    base_lr = full.eval_logreg.pr_auc
    base_rf = full.eval_rf.pr_auc
    results.append(AblationResult(
        "full_baseline", len(ds.feature_names), base_lr, base_rf, 0.0, 0.0))

    # Classify features so we can identify frozen vs attacker-controlled.
    report = classify(ds.feature_names)
    frozen = {f for f in ds.feature_names
              if report.assignments.get(f) is Control.FROZEN}
    attacker = [f for f in ds.feature_names if f not in frozen]

    # 1. Drop Destination Port alone.
    if "Destination Port" in ds.feature_names:
        keep = [f for f in ds.feature_names if f != "Destination Port"]
        sub = _subset_dataset(ds, keep)
        tb = train_baseline(sub, seed=seed, top_k=5)
        results.append(AblationResult(
            "drop_destination_port", len(keep),
            tb.eval_logreg.pr_auc, tb.eval_rf.pr_auc,
            tb.eval_logreg.pr_auc - base_lr, tb.eval_rf.pr_auc - base_rf))

    # 2. Controllable-only (drop all frozen).
    if attacker:
        sub = _subset_dataset(ds, attacker)
        tb = train_baseline(sub, seed=seed, top_k=5)
        results.append(AblationResult(
            "controllable_only", len(attacker),
            tb.eval_logreg.pr_auc, tb.eval_rf.pr_auc,
            tb.eval_logreg.pr_auc - base_lr, tb.eval_rf.pr_auc - base_rf))

    # 3. Drop each top feature individually (union of both models' top-5).
    top_feats = []
    for name, _ in full.top_logreg.ranked[:5] + full.top_rf.ranked[:5]:
        if name not in top_feats:
            top_feats.append(name)
    for feat in top_feats:
        keep = [f for f in ds.feature_names if f != feat]
        sub = _subset_dataset(ds, keep)
        tb = train_baseline(sub, seed=seed, top_k=3)
        results.append(AblationResult(
            f"drop::{feat}", len(keep),
            tb.eval_logreg.pr_auc, tb.eval_rf.pr_auc,
            tb.eval_logreg.pr_auc - base_lr, tb.eval_rf.pr_auc - base_rf))

    return results