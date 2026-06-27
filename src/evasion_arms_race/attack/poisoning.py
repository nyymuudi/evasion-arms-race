"""Realisability-constrained data poisoning (Layer B).

Research question: can an attacker who injects REALISTIC (feasibility-projected)
samples into the detector's retraining data shift the boundary so that later Hulk
traffic is classified benign -- and at what poison fraction, if any, does
detection collapse?

What separates this from typical poisoning work: the poison samples must pass the
SAME feasibility projection (features.projection.project) and packet-level
feasibility check (validation.realisability.is_feasible) as the Layer A evasion
samples, and must keep the DoS functional floor (they are still Hulk). Realistic
poisoning, not arbitrary points in feature space.

Threat model (decided in Phase 1)
---------------------------------
* Access: CHOSEN-LABEL INJECTION. The attacker adds a budget-limited set of
  samples it labels benign, constrained to realisable Hulk traffic. This measures
  the upper bound of poisoning power under the realisability constraint, cleanly
  separated from the labelling channel. The more realistic raw-traffic /
  auto-labelling attacker is strictly weaker -- a sample only enters with a benign
  label if it is a successful REALISABLE evasion, so its budget is gated by the
  item-7 result (52% for LR, 0% for RF). That corollary is argued, not re-run.
* Black-box: query access to the current detector only (consistent with Layer A).
  Boundary selection ranks Hulk samples by the detector's confidence (a soft-label
  query); no model internals, no influence-function gradients (white-box poisoning
  a la Koh & Liang is deferred).
* Budget: a log-spaced sweep over the poison fraction (0.5%-20% of clean train).

Evaluation is on the CLEAN held-out test set the attacker never saw, so we measure
poisoning, not overfitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from evasion_arms_race.data.loader import Dataset
from evasion_arms_race.detector.baseline import train_baseline
from evasion_arms_race.features.projection import project
from evasion_arms_race.features.rule_classifier import classify
from evasion_arms_race.validation.realisability import is_feasible


# --------------------------------------------------------------------------- #
# Poison sample generation
# --------------------------------------------------------------------------- #
def project_realisable(row: dict[str, float], source) -> dict[str, float]:
    """The realisable version of a flow: project it onto the feasible set against
    itself (frozen/floor identity, derived recomputed). Real captured Hulk flows
    are feasible fixed points up to the derived recompute residual."""
    return project(row, row, source).vector


def _stratified_pool(ds: Dataset):
    """Indices of the Hulk (attack) samples in the clean training pool -- the
    attacker's own traffic, which it relabels benign."""
    return np.where(ds.y_train == 1)[0]


def generate_poison(
    ds: Dataset,
    n_poison: int,
    strategy: str,
    scorer: Callable[[pd.DataFrame], np.ndarray] | None = None,
    source=None,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Generate `n_poison` realisable Hulk samples labelled benign.

    strategy:
      'label_flip' : random real Hulk samples (a baseline; sets a lower bar).
      'boundary'   : real Hulk samples nearest the current detector's boundary
                     (lowest attack-probability via `scorer`), so each injected
                     sample sits where it can move the boundary most cheaply.

    Every candidate is projected onto the feasible set and kept only if it passes
    the packet-level feasibility check -- realistic poison, not feature-space dust.
    Returns (X_poison, y_poison=zeros, info).
    """
    if source is None:
        source = classify(ds.feature_names)
    rng = np.random.default_rng(seed)
    pool = _stratified_pool(ds)

    if strategy == "label_flip":
        order = rng.permutation(pool)
    elif strategy == "boundary":
        if scorer is None:
            raise ValueError("boundary strategy needs a scorer (detector query access)")
        proba = scorer(ds.X_train.iloc[pool])           # attack-probability per Hulk sample
        order = pool[np.argsort(proba)]                  # nearest the benign side first
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    rows, kept, infeasible = [], 0, 0
    for idx in order:
        if kept >= n_poison:
            break
        rv = project_realisable(ds.X_train.iloc[idx].to_dict(), source)
        if is_feasible(rv).feasible:
            rows.append(rv)
            kept += 1
        else:
            infeasible += 1

    X_poison = pd.DataFrame(rows, columns=ds.feature_names) if rows else \
        pd.DataFrame(columns=ds.feature_names)
    y_poison = np.zeros(len(X_poison), dtype=int)          # labelled benign
    info = {"requested": n_poison, "kept": len(X_poison),
            "skipped_infeasible": infeasible, "strategy": strategy}
    return X_poison, y_poison, info


# --------------------------------------------------------------------------- #
# Retraining pipeline
# --------------------------------------------------------------------------- #
def subsample_clean(ds: Dataset, clean_train_size: int, seed: int = 0
                    ) -> tuple[pd.DataFrame, np.ndarray]:
    """A stratified subsample of the clean training set (for iteration speed)."""
    if clean_train_size >= len(ds.y_train):
        return ds.X_train.reset_index(drop=True), ds.y_train
    rng = np.random.default_rng(seed)
    frac = clean_train_size / len(ds.y_train)
    keep = []
    for cls in (0, 1):
        idx = np.where(ds.y_train == cls)[0]
        keep.append(rng.choice(idx, size=int(round(len(idx) * frac)), replace=False))
    keep = np.concatenate(keep)
    return ds.X_train.iloc[keep].reset_index(drop=True), ds.y_train[keep]


def make_dataset(Xtr: pd.DataFrame, ytr: np.ndarray, clean: Dataset) -> Dataset:
    """A Dataset with the given (poisoned) training set and the CLEAN test set."""
    return Dataset(
        X_train=Xtr.reset_index(drop=True), X_test=clean.X_test,
        y_train=np.asarray(ytr), y_test=clean.y_test,
        feature_names=clean.feature_names, target_label=clean.target_label,
        n_dropped_nonfinite=clean.n_dropped_nonfinite,
    )


def inject(Xc: pd.DataFrame, yc: np.ndarray, X_poison: pd.DataFrame,
           y_poison: np.ndarray, clean: Dataset) -> Dataset:
    """Append poison to a clean training subsample; test stays clean."""
    Xtr = pd.concat([Xc, X_poison], ignore_index=True)
    ytr = np.concatenate([yc, y_poison])
    return make_dataset(Xtr, ytr, clean)


@dataclass
class PoisonPoint:
    strategy: str
    fraction: float
    n_poison: int
    pr_auc_logreg: float
    pr_auc_rf: float
    hulk_recall_logreg: float        # TPR on Hulk at 0.5 -> drop = Hulk seen as benign
    hulk_recall_rf: float


def _hulk_recall(model, scaler, Xte: pd.DataFrame, yte: np.ndarray) -> float:
    proba = model.predict_proba(scaler.transform(Xte.to_numpy()))[:, 1]
    pred = (proba >= 0.5).astype(int)
    pos = yte == 1
    return float(pred[pos].mean()) if pos.any() else float("nan")


def evaluate_poison(ds_poisoned: Dataset, clean: Dataset, strategy: str,
                    fraction: float, n_poison: int, seed: int = 0) -> PoisonPoint:
    """Retrain both detectors on the poisoned training set, evaluate on clean test."""
    tb = train_baseline(ds_poisoned, seed=seed, top_k=5)
    return PoisonPoint(
        strategy=strategy, fraction=fraction, n_poison=n_poison,
        pr_auc_logreg=tb.eval_logreg.pr_auc, pr_auc_rf=tb.eval_rf.pr_auc,
        hulk_recall_logreg=_hulk_recall(tb.logreg, tb.scaler, clean.X_test, clean.y_test),
        hulk_recall_rf=_hulk_recall(tb.rf, tb.scaler, clean.X_test, clean.y_test),
    )


def run_sweep(
    clean: Dataset,
    fractions: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
    strategies: tuple[str, ...] = ("label_flip", "boundary"),
    clean_train_size: int = 30000,
    seed: int = 0,
) -> dict:
    """The poison-threshold experiment: detection PR-AUC on clean test as a
    function of poison fraction, per strategy, for both detectors."""
    source = classify(clean.feature_names)
    Xc, yc = subsample_clean(clean, clean_train_size, seed=seed)

    # The 'current detector' the attacker queries for boundary selection, also the
    # 0%-poison reference point.
    base = train_baseline(make_dataset(Xc, yc, clean), seed=seed, top_k=5)

    def scorer(df: pd.DataFrame) -> np.ndarray:
        return base.logreg.predict_proba(base.scaler.transform(df.to_numpy()))[:, 1]

    n_clean = len(yc)
    results: dict[str, list[PoisonPoint]] = {s: [] for s in strategies}
    gen_info: list[dict] = []

    for strat in strategies:
        for frac in fractions:
            n_poison = int(round(frac * n_clean))
            if n_poison == 0:
                point = evaluate_poison(make_dataset(Xc, yc, clean), clean,
                                        strat, 0.0, 0, seed=seed)
            else:
                Xp, yp, info = generate_poison(clean, n_poison, strat,
                                               scorer=scorer, source=source, seed=seed)
                gen_info.append({"fraction": frac, **info})
                ds_p = inject(Xc, yc, Xp, yp, clean)
                point = evaluate_poison(ds_p, clean, strat, frac, len(Xp), seed=seed)
            results[strat].append(point)

    return {"n_clean_train": n_clean, "fractions": list(fractions),
            "results": results, "generation": gen_info,
            "reference_pr_auc": {"logreg": base.eval_logreg.pr_auc,
                                 "rf": base.eval_rf.pr_auc}}


# --------------------------------------------------------------------------- #
# Reporting harness
# --------------------------------------------------------------------------- #
def main() -> int:
    import argparse
    import json
    from pathlib import Path

    from evasion_arms_race.data.loader import build_dataset

    ap = argparse.ArgumentParser(description="Realisability-constrained poisoning (Layer B).")
    ap.add_argument("--data", default="data/raw/cicids2017/MachineLearningCVE/"
                                       "Wednesday-workingHours.pcap_ISCX.csv")
    ap.add_argument("--clean-train-size", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig-dir", default="experiments/figures")
    ap.add_argument("--summary-out", default="experiments/poisoning.json")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean = build_dataset(args.data, target_label="DoS Hulk", seed=args.seed)
    print(f"clean: {len(clean.y_train)} train / {len(clean.y_test)} test; "
          f"subsampling clean train to {args.clean_train_size}")
    sweep = run_sweep(clean, clean_train_size=args.clean_train_size, seed=args.seed)

    fractions = sweep["fractions"]
    ref = sweep["reference_pr_auc"]
    summary = {"config": {"clean_train_size": sweep["n_clean_train"], "seed": args.seed,
                          "data": args.data},
               "reference_pr_auc": ref, "fractions": fractions, "strategies": {}}

    print("\n=== Realisability-constrained poisoning (Layer B) ===")
    print(f"reference PR-AUC (0% poison): logreg={ref['logreg']:.4f} rf={ref['rf']:.4f}\n")
    for strat, pts in sweep["results"].items():
        summary["strategies"][strat] = [
            {"fraction": p.fraction, "n_poison": p.n_poison,
             "pr_auc_logreg": p.pr_auc_logreg, "pr_auc_rf": p.pr_auc_rf,
             "hulk_recall_logreg": p.hulk_recall_logreg, "hulk_recall_rf": p.hulk_recall_rf}
            for p in pts]
        print(f"[{strat}]")
        for p in pts:
            print(f"  poison {p.fraction:5.1%} (n={p.n_poison:5d}): "
                  f"PR-AUC lr={p.pr_auc_logreg:.4f} rf={p.pr_auc_rf:.4f} | "
                  f"Hulk recall lr={p.hulk_recall_logreg:.4f} rf={p.hulk_recall_rf:.4f}")
        print()

    fig_dir = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    titles = {"logreg": "Logistic Regression", "rf": "Random Forest"}

    def two_panel(attr, ylabel, ylim, suptitle, fname, ref_lines):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for ax, model in zip(axes, ("logreg", "rf")):
            for strat, pts in sweep["results"].items():
                xs = [p.fraction for p in pts]
                ys = [getattr(p, f"{attr}_{model}") for p in pts]
                ax.plot(xs, ys, marker="o", ms=4, label=strat)
            if ref_lines:
                ax.axhline(ref[model], ls="--", c="grey", lw=1, label="clean reference")
            ax.set_title(titles[model]); ax.set_xlabel("poison fraction of clean train")
            ax.set_ylim(*ylim); ax.grid(alpha=0.3)
        axes[0].set_ylabel(ylabel); axes[1].legend(fontsize=8)
        fig.suptitle(suptitle); fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=120); plt.close(fig)

    # PR-AUC: threshold-independent ranking. Fixed axis so the near-flatness is honest.
    two_panel("pr_auc", "PR-AUC on CLEAN test", (0.99, 1.001),
              "Detection (PR-AUC) vs poison budget — realisable, feasibility-projected poison",
              "poisoning_pr_auc.png", ref_lines=True)
    # Hulk recall at the deployed 0.5 threshold: where the real poisoning effect shows.
    two_panel("hulk_recall", "Hulk recall on CLEAN test (@0.5)", (0.6, 1.01),
              "Hulk recall vs poison budget — operating-point degradation",
              "poisoning_hulk_recall.png", ref_lines=False)

    out = Path(args.summary_out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"figure -> {fig_dir}/poisoning_pr_auc.png   summary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
