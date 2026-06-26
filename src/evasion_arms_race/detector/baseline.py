"""Baseline DoS-Hulk detectors (roadmap weeks 1-2).

Two detectors are trained on the same scaled feature space:

  * Logistic Regression -- the PRIMARY evasion target. Its decision boundary is
    smooth, so small feasible perturbations move the score continuously; this
    makes black-box gradient estimation and boundary search informative. Its
    coefficients also show directly which features drive the decision.
  * Random Forest -- a ROBUSTNESS COMPARATOR. Strong on this data but piecewise
    constant, so it is a harder, less informative evasion target. The contrast
    ("evasion works on the smooth model, stalls on the forest") is the
    scientifically interesting result, not a single accuracy number.

Honest evaluation
-----------------
DoS Hulk vs BENIGN on CIC-IDS2017 is NOT severely imbalanced (~1:1.9), so
accuracy is uninformative and likely near-ceiling. We report PR-AUC and ROC-AUC
and, crucially, FEATURE IMPORTANCE / COEFFICIENTS -- because the whole Layer A
tension is whether the detector leans on features that sit at the DoS
functional floor (rate/volume), which the feasibility projection forbids the
attacker from lowering. Seeing that now, not at attack time, is the point.

Shared scaled space
-------------------
The StandardScaler is fit on TRAIN ONLY and persisted as an artifact. The attack
must consume the same scaler so detector and attack measure perturbation in one
agreed space; otherwise L2/Linf magnitudes are meaningless (Flow Duration, in
microseconds, would dominate any unscaled metric).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from evasion_arms_race.data.loader import Dataset

ARTIFACT_DIR = Path("data/artifacts")


@dataclass
class EvalReport:
    model_name: str
    pr_auc: float          # average precision -- the headline metric here
    roc_auc: float
    accuracy: float        # reported for context only; not the headline
    n_train: int
    n_test: int
    pos_frac_test: float


@dataclass
class TopFeatures:
    model_name: str
    # (feature_name, signed_importance) sorted by |importance| desc
    ranked: list[tuple[str, float]]


def _fit_scaler(ds: Dataset) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(ds.X_train.to_numpy())
    return scaler


def _evaluate(name: str, model, Xte: np.ndarray, yte: np.ndarray,
              n_train: int) -> EvalReport:
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return EvalReport(
        model_name=name,
        pr_auc=float(average_precision_score(yte, proba)),
        roc_auc=float(roc_auc_score(yte, proba)),
        accuracy=float((pred == yte).mean()),
        n_train=n_train,
        n_test=len(yte),
        pos_frac_test=float(yte.mean()),
    )


def _logreg_top(model: LogisticRegression, names: list[str], k: int) -> TopFeatures:
    coefs = model.coef_.ravel()
    order = np.argsort(np.abs(coefs))[::-1][:k]
    return TopFeatures("logistic_regression",
                       [(names[i], float(coefs[i])) for i in order])


def _rf_top(model: RandomForestClassifier, names: list[str], k: int) -> TopFeatures:
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1][:k]
    return TopFeatures("random_forest",
                       [(names[i], float(imp[i])) for i in order])


@dataclass
class TrainedBaseline:
    scaler: StandardScaler
    logreg: LogisticRegression
    rf: RandomForestClassifier
    feature_names: list[str]
    eval_logreg: EvalReport
    eval_rf: EvalReport
    top_logreg: TopFeatures
    top_rf: TopFeatures


def train_baseline(ds: Dataset, seed: int = 0, top_k: int = 15) -> TrainedBaseline:
    """Fit scaler + both detectors on ds; return trained models and reports."""
    scaler = _fit_scaler(ds)
    Xtr = scaler.transform(ds.X_train.to_numpy())
    Xte = scaler.transform(ds.X_test.to_numpy())

    logreg = LogisticRegression(max_iter=1000, random_state=seed)
    logreg.fit(Xtr, ds.y_train)

    rf = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=seed, class_weight=None
    )
    rf.fit(Xtr, ds.y_train)

    return TrainedBaseline(
        scaler=scaler,
        logreg=logreg,
        rf=rf,
        feature_names=ds.feature_names,
        eval_logreg=_evaluate("logistic_regression", logreg, Xte, ds.y_test, len(ds.y_train)),
        eval_rf=_evaluate("random_forest", rf, Xte, ds.y_test, len(ds.y_train)),
        top_logreg=_logreg_top(logreg, ds.feature_names, top_k),
        top_rf=_rf_top(rf, ds.feature_names, top_k),
    )


def save_artifacts(tb: TrainedBaseline, out_dir: Path = ARTIFACT_DIR) -> None:
    """Persist scaler + models + reports so the attack can reuse them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(tb.scaler, f)
    with open(out_dir / "logreg.pkl", "wb") as f:
        pickle.dump(tb.logreg, f)
    with open(out_dir / "rf.pkl", "wb") as f:
        pickle.dump(tb.rf, f)
    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(tb.feature_names, f, indent=2)
    reports = {
        "logreg": asdict(tb.eval_logreg),
        "rf": asdict(tb.eval_rf),
        "top_logreg": tb.top_logreg.ranked,
        "top_rf": tb.top_rf.ranked,
    }
    with open(out_dir / "baseline_report.json", "w") as f:
        json.dump(reports, f, indent=2)