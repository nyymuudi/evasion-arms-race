"""Adversarial training + arms-race primitives (Layer C, the capstone).

Layer C closes the loop: harden the detector by training on the attacker's
REALISABLE evasion samples, then iterate the attack/retrain loop and observe its
dynamics. The honest scope discipline (see docs/game_theory.md) is enforced in
naming here: this module measures an EMPIRICAL adaptive dynamic. It does not
claim to compute a Nash equilibrium -- the preconditions for that are analysed,
and largely not met, in the docs.

Key methodological choices (decided in Phase 1):
  * Adversarial samples are the attack's REALISABLE evasions only (they pass
    validation.realisability.is_feasible), labelled with their CORRECT class
    (attack = 1). This is the defensive dual of Layer B poisoning, and it ties
    Layer C to the item-7 result: a detector against which the attack finds no
    realisable evasion (the Random Forest, 0%) cannot be adversarially hardened
    this way -- there is nothing realistic to train on, which is itself the
    finding.
  * Each round attacks FRESH Hulk samples (not a fixed set), so a falling evasion
    rate reflects generalising robustness, not memorisation of past attacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from evasion_arms_race.attack.blackbox import AttackConfig, Oracle, attack_sample
from evasion_arms_race.validation.realisability import is_feasible


def _new_model(model_name: str, seed: int):
    if model_name in ("logreg", "logistic_regression"):
        return LogisticRegression(max_iter=1000, random_state=seed)
    if model_name in ("rf", "random_forest"):
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)
    raise KeyError(f"unknown model {model_name!r}")


@dataclass
class TrainedDetector:
    model_name: str
    model: object
    scaler: StandardScaler
    feature_names: list[str]
    pr_auc: float            # on the clean held-out test
    hulk_recall: float       # TPR on Hulk at 0.5

    def attack_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X.to_numpy()))[:, 1]


def train_one(model_name: str, Xtr: pd.DataFrame, ytr: np.ndarray,
              Xte: pd.DataFrame, yte: np.ndarray, seed: int = 0) -> TrainedDetector:
    """Train a SINGLE detector (per-model so the two arms races stay independent)
    and evaluate it on the clean test set."""
    scaler = StandardScaler().fit(Xtr.to_numpy())
    model = _new_model(model_name, seed)
    model.fit(scaler.transform(Xtr.to_numpy()), ytr)
    proba = model.predict_proba(scaler.transform(Xte.to_numpy()))[:, 1]
    pred = (proba >= 0.5).astype(int)
    pos = yte == 1
    return TrainedDetector(
        model_name=model_name, model=model, scaler=scaler,
        feature_names=list(Xtr.columns),
        pr_auc=float(average_precision_score(yte, proba)),
        hulk_recall=float(pred[pos].mean()) if pos.any() else float("nan"),
    )


@dataclass
class AttackOutcome:
    feature_success_rate: float          # fraction the boundary attack flips (label only)
    realisable_rate: float               # fraction that ALSO pass is_feasible (the on-thesis metric)
    median_l2_controllable: float | None
    evasions: list[dict] = field(default_factory=list)             # ALL successful vectors (training material)
    realisable_evasions: list[dict] = field(default_factory=list)  # the is_feasible subset (tracked)


def attack_detector(det: TrainedDetector, hulk_samples: list[dict],
                    benign_refs: list[dict], source, cfg: AttackConfig,
                    projector=None) -> AttackOutcome:
    """Run the Layer A boundary attack against `det`. Returns ALL successful
    evasions -- the adversarial-training material -- and, separately, the subset
    that pass the item-7 feasibility check, whose RATE is the on-thesis metric.

    `projector` selects the search space: the default (None -> project) is the
    free-space search whose realisable yield is read off by post-filter; pass
    validation.realisability.manifold_project to confine the search to the
    realisable manifold, in which case every success is realisable by construction
    (evasions == realisable_evasions) and the rate is the true manifold rate.
    """
    n = len(hulk_samples)
    evasions, realisable, l2s = [], [], []
    for clean in hulk_samples:
        oracle = Oracle(det.model, det.scaler, det.feature_names)
        res = attack_sample(clean, oracle, benign_refs, cfg, source, projector=projector)
        if res.success and res.best_vector:
            evasions.append(res.best_vector)
            l2s.append(res.l2_controllable)
            if is_feasible(res.best_vector).feasible:
                realisable.append(res.best_vector)
    return AttackOutcome(
        feature_success_rate=len(evasions) / n if n else 0.0,
        realisable_rate=len(realisable) / n if n else 0.0,
        median_l2_controllable=float(np.median(l2s)) if l2s else None,
        evasions=evasions,
        realisable_evasions=realisable,
    )


def adversarial_trainset(
    Xc: pd.DataFrame, yc: np.ndarray, evasions: list[dict],
    feature_names: list[str], replication: int = 5,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Clean train augmented with realisable evasions labelled ATTACK (1).

    Evasions are oversampled `replication` times: they are expensive to generate,
    so a handful would otherwise be a negligible fraction of the training set. The
    correct label is attack -- this teaches the detector that these benign-looking
    flows are in fact attacks (the defensive dual of Layer B's benign mislabelling).
    """
    if not evasions:
        return Xc.reset_index(drop=True), np.asarray(yc)
    rows = evasions * replication
    Xadv = pd.DataFrame(rows, columns=feature_names)
    yadv = np.ones(len(Xadv), dtype=int)
    Xtr = pd.concat([Xc, Xadv], ignore_index=True)
    ytr = np.concatenate([np.asarray(yc), yadv])
    return Xtr, ytr
