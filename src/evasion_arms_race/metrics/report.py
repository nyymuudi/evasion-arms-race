"""Evasion reporting harness (Layer A, item 6).

Runs the item-5 black-box attack end to end (reusing attack.blackbox), turns the
per-sample results into the four required reportable products via the pure
functions in metrics.evasion, and emits:

  * matplotlib figures   -> experiments/figures/        (gitignored)
  * a small numeric JSON -> experiments/evasion_metrics.json (version-controlled)

This is a SCRIPT, not a notebook, deliberately: the figures + JSON are part of
the project's reproducible path (run it, get the same artifacts), it diffs and
tests cleanly, and the repo reserves notebooks/ for throwaway exploration. The
heavy lifting is in importable, unit-tested functions; this file is only glue.

Products
--------
1. Success vs allowed perturbation magnitude (the tradeoff curve), both models.
2. Three-class decomposition vs query budget: success / feasibility_bound (DoS
   floor blocked it) / detector_bound, both models.
3. Per-feature movement of successful evasions vs the detector's top features.
4. Logistic regression vs Random Forest on every figure.

Run:  python -m evasion_arms_race.metrics.report --n-samples 25 --budget 1200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evasion_arms_race.attack.blackbox import (
    AttackConfig,
    load_artifacts,
    run_attack,
)
from evasion_arms_race.data.loader import build_dataset
from evasion_arms_race.features.partition import Control, normalize
from evasion_arms_race.features.rule_classifier import classify
from evasion_arms_race.metrics import evasion as M

MODELS = ("logreg", "rf")
MODEL_LABEL = {"logreg": "Logistic Regression", "rf": "Random Forest"}
CLASS_COLOR = {"success": "#2a9d8f",
               "feasibility_bound": "#e9c46a",
               "detector_bound": "#e76f51"}


# --------------------------------------------------------------------------- #
# Data extraction helpers
# --------------------------------------------------------------------------- #
def _scaler_fn(scaler, feature_names):
    fn = feature_names

    def scale_dict(d: dict[str, float]) -> np.ndarray:
        vec = np.array([float(d[f]) for f in fn], dtype=float)
        return scaler.transform(vec.reshape(1, -1))[0]

    return scale_dict


def _success_deltas(hulk_samples, results, scale_dict, feature_names, perturbable):
    """Per successful sample: scaled signed delta for each perturbable feature."""
    out = []
    for clean, r in zip(hulk_samples, results):
        if r.success and r.best_vector:
            cs = scale_dict(clean)
            bs = scale_dict(r.best_vector)
            out.append({f: float(bs[i] - cs[i])
                        for i, f in enumerate(feature_names) if f in perturbable})
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _fig_perturbation(plt, per_model, eps_grid, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in MODELS:
        ax.plot(eps_grid, per_model[m]["pert_curve"], marker="o", ms=3,
                label=MODEL_LABEL[m])
    ax.set_xlabel("allowed perturbation  ε   (controllable L2, scaled space)")
    ax.set_ylabel("evasion success rate  (≤ ε)")
    ax.set_title("Success vs perturbation budget  (lower-bound; minimal found)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _fig_budget(plt, per_model, budgets, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in MODELS:
        ax.plot(budgets, per_model[m]["budget_curve"], marker="o", ms=3,
                label=MODEL_LABEL[m])
    ax.set_xlabel("query budget  (feasible queries)")
    ax.set_ylabel("evasion success rate")
    ax.set_title("Success vs query budget")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _fig_decomposition(plt, per_model, budgets, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, m in zip(axes, MODELS):
        ser = per_model[m]["decomp_series"]
        ax.stackplot(
            budgets, ser["success"], ser["feasibility_bound"], ser["detector_bound"],
            labels=["success", "feasibility_bound (DoS floor)", "detector_bound"],
            colors=[CLASS_COLOR["success"], CLASS_COLOR["feasibility_bound"],
                    CLASS_COLOR["detector_bound"]],
        )
        ax.set_title(MODEL_LABEL[m])
        ax.set_xlabel("query budget")
        ax.set_ylim(0, 1)
        ax.margins(x=0)
    axes[0].set_ylabel("fraction of samples")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Outcome decomposition vs budget: why evasion does / doesn't happen")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _fig_feature_movement(plt, per_model, detector_top, out_path, top_k=12):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, m in zip(axes, MODELS):
        mv = per_model[m]["movement"][:top_k][::-1]   # ascending for barh
        feats = [f for f, _ in mv]
        vals = [v for _, v in mv]
        det_set = {f for f, _ in detector_top[m][:10]}
        colors = ["#264653" if f in det_set else "#90a955" for f in feats]
        ax.barh(range(len(feats)), vals, color=colors)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats, fontsize=8)
        ax.set_xlabel("mean |Δ|  (scaled)")
        ax.set_title(MODEL_LABEL[m])
    fig.suptitle("Features moved most by successful evasions "
                 "(dark = also in detector's top-10)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description="Reportable evasion metrics (item 6).")
    ap.add_argument("--data", default="data/raw/cicids2017/MachineLearningCVE/"
                                       "Wednesday-workingHours.pcap_ISCX.csv")
    ap.add_argument("--artifacts", default="data/artifacts")
    ap.add_argument("--n-samples", type=int, default=25)
    ap.add_argument("--n-refs", type=int, default=8)
    ap.add_argument("--budget", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig-dir", default="experiments/figures")
    ap.add_argument("--summary-out", default="experiments/evasion_metrics.json")
    args = ap.parse_args()

    art_dir = Path(args.artifacts)
    artifacts = load_artifacts(art_dir)
    feature_names = artifacts.feature_names
    detector_report = json.loads((art_dir / "baseline_report.json").read_text())
    detector_top = {
        "logreg": [(normalize(f), w) for f, w in detector_report["top_logreg"]],
        "rf": [(normalize(f), w) for f, w in detector_report["top_rf"]],
    }

    source = classify(feature_names)
    if source.unknown:
        raise SystemExit(f"unclassified features: {source.unknown}")
    perturbable = {f for f in feature_names
                   if source.control_of(f) in (Control.CONTROLLABLE, Control.CONSTRAINED)}

    # Sample the same way the attack CLI does, for reproducibility.
    ds = build_dataset(args.data, target_label="DoS Hulk", seed=args.seed)
    if feature_names != [normalize(c) for c in ds.feature_names]:
        raise SystemExit("feature order mismatch between artifacts and data")
    rng = np.random.default_rng(args.seed)
    hulk_idx = np.where(ds.y_test == 1)[0]
    benign_idx = np.where(ds.y_test == 0)[0]
    hulk_pick = rng.choice(hulk_idx, size=min(args.n_samples, hulk_idx.size), replace=False)
    benign_pick = rng.choice(benign_idx, size=min(args.n_refs, benign_idx.size), replace=False)
    hulk_samples = [ds.X_test.iloc[i].to_dict() for i in hulk_pick]
    benign_refs = [ds.X_test.iloc[i].to_dict() for i in benign_pick]

    cfg = AttackConfig(query_budget=args.budget, n_init_refs=args.n_refs, seed=args.seed)
    print(f"running attack: {len(hulk_samples)} samples x {len(MODELS)} models, "
          f"budget {args.budget} ...")
    results = run_attack(artifacts, hulk_samples, benign_refs, config=cfg)

    scale_dict = _scaler_fn(artifacts.scaler, feature_names)

    # Shared axes across models.
    all_l2 = [r.l2_controllable for m in MODELS for r in results[m] if r.success]
    eps_grid = M.default_eps_grid(all_l2, n=25)
    budgets = np.linspace(1, args.budget, 40).astype(int)
    decomp_budgets = sorted(set(int(args.budget * f)
                                for f in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0)) | {1})

    per_model: dict = {}
    summary: dict = {"config": {"n_samples": len(hulk_samples), "n_refs": args.n_refs,
                                "budget": args.budget, "seed": args.seed,
                                "data": args.data},
                     "models": {}}

    for m in MODELS:
        rs = results[m]
        n = len(rs)
        l2 = [r.l2_controllable for r in rs]
        success = [r.success for r in rs]
        first_ev = [r.queries_to_first_evasion for r in rs]
        first_fb = [r.queries_to_first_floor_block for r in rs]

        pert_curve = M.perturbation_curve(l2, success, eps_grid)
        bcurve = M.budget_curve(first_ev, n, budgets)
        decomp_series = M.decomposition_series(first_ev, first_fb, budgets)
        floor = M.floor_binding([r.flip_attempts for r in rs],
                                [r.floor_blocked for r in rs])
        blocking = M.aggregate_blocking([r.blocking_features for r in rs])
        deltas = _success_deltas(hulk_samples, rs, scale_dict, feature_names, perturbable)
        movement = M.feature_movement(deltas)
        moved_names = [f for f, _ in movement]
        det_names = [f for f, _ in detector_top[m]]
        overlap = M.topk_overlap(moved_names, det_names, k=10)
        det_top10 = det_names[:10]
        det_movable = [f for f in det_top10 if f in perturbable]
        det_unmovable = [f for f in det_top10 if f not in perturbable]

        per_model[m] = {"pert_curve": pert_curve, "budget_curve": bcurve,
                        "decomp_series": decomp_series, "movement": movement}

        summary["models"][m] = {
            "overall_success_rate": float(np.mean(success)) if n else 0.0,
            "median_l2_controllable_success": (
                float(np.median([v for v, s in zip(l2, success) if s]))
                if any(success) else None),
            "median_feasible_queries": float(np.median([r.feasible_queries for r in rs])),
            "perturbation_curve": {"eps": [float(e) for e in eps_grid],
                                   "success_rate": [float(v) for v in pert_curve]},
            "budget_curve": {"budget": [int(b) for b in budgets],
                             "success_rate": [float(v) for v in bcurve]},
            "decomposition_at_budget": {
                str(B): M.decompose_at_budget(first_ev, first_fb, B)
                for B in decomp_budgets},
            "floor_binding": {**floor, "top_reverting_features": dict(list(blocking.items())[:8])},
            "feature_movement_top10": [[f, float(v)] for f, v in movement[:10]],
            "attack_vs_detector_top10_overlap": overlap,
            "detector_top_movable": det_movable,
            "detector_top_unmovable": det_unmovable,
        }

    # Figures.
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    _fig_perturbation(plt, per_model, eps_grid, fig_dir / "success_vs_perturbation.png")
    _fig_budget(plt, per_model, budgets, fig_dir / "success_vs_budget.png")
    _fig_decomposition(plt, per_model, budgets, fig_dir / "outcome_decomposition.png")
    _fig_feature_movement(plt, per_model, detector_top, fig_dir / "feature_movement.png")

    # Version-controllable numeric summary.
    out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    # Console digest.
    print("\n=== Evasion metrics (item 6) ===")
    for m in MODELS:
        s = summary["models"][m]
        print(f"\n[{MODEL_LABEL[m]}]")
        print(f"  success rate              : {s['overall_success_rate']:.2%}")
        print(f"  median L2 (ctrl, success) : {s['median_l2_controllable_success']}")
        print(f"  floor block rate          : {s['floor_binding']['block_rate']:.2%} "
              f"({s['floor_binding']['floor_blocked']}/{s['floor_binding']['flip_attempts']} flips)")
        print(f"  top moved features        : {[f for f, _ in s['feature_movement_top10'][:5]]}")
        print(f"  detector top-10 MOVABLE   : {len(s['detector_top_movable'])}/10  "
              f"-> {s['detector_top_movable'][:4]}")
        print(f"  detector top-10 UNMOVABLE : {len(s['detector_top_unmovable'])}/10  "
              f"-> {s['detector_top_unmovable'][:4]}")
        print(f"  attack∩detector top-10    : {s['attack_vs_detector_top10_overlap']['shared']}")
    print(f"\nfigures -> {fig_dir}/   summary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
