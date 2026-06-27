"""Attack/retrain arms-race loop (Layer C, phases 2-3).

Iterates: (i) the attacker runs the Layer A boundary attack on FRESH Hulk samples
against the current detector; (ii) the detector is adversarially retrained on the
attacker's REALISABLE evasions (labelled attack); (iii) repeat. Each round logs
the evasion success rate (feature-space and realisable), the clean-test PR-AUC and
Hulk recall, and the median controllable perturbation. The resulting time series
is the object of the empirical convergence analysis.

HONEST SCOPE (read docs/game_theory.md): this measures an empirical adaptive
dynamic and asks whether it converges, oscillates, or diverges, and whether a
fixed point (no profitable feasible deviation) is reached. It is NOT a proof of a
Nash equilibrium; the preconditions for that are examined, and mostly fail, in the
docs. The comparison to a poker solver's CFR curve is conceptual (both are
adaptive best-response-style dynamics), not a formal equivalence.

Run:  python experiments/arms_race.py --rounds 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evasion_arms_race.attack.blackbox import AttackConfig
from evasion_arms_race.attack.poisoning import subsample_clean
from evasion_arms_race.data.loader import build_dataset
from evasion_arms_race.detector.robust import (
    attack_detector,
    adversarial_trainset,
    train_one,
)
from evasion_arms_race.features.rule_classifier import classify

MODELS = ("logreg", "rf")
MODEL_LABEL = {"logreg": "Logistic Regression", "rf": "Random Forest"}


def run_model_loop(model_name, Xc, yc, clean, source, hulk_pool_rows, benign_refs,
                   rounds, n_attack, cfg, replication, seed, projector=None):
    """One model's arms race. Returns a per-round record list."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(hulk_pool_rows)            # fresh Hulk samples per round
    det = train_one(model_name, Xc, yc, clean.X_test, clean.y_test, seed=seed)
    acc_evasions: list[dict] = []
    record = []
    for r in range(rounds):
        hs_rows = order[r * n_attack:(r + 1) * n_attack]
        hulk_samples = [clean.X_train.iloc[i].to_dict() for i in hs_rows]
        outcome = attack_detector(det, hulk_samples, benign_refs, source, cfg, projector=projector)
        record.append({
            "round": r,
            "pr_auc": det.pr_auc,
            "hulk_recall": det.hulk_recall,
            "feature_success_rate": outcome.feature_success_rate,
            "realisable_rate": outcome.realisable_rate,
            "median_l2_controllable": outcome.median_l2_controllable,
            "n_realisable_evasions": len(outcome.realisable_evasions),
            "n_adversarial_train": len(acc_evasions) * replication,
        })
        print(f"  [{model_name}] round {r}: PR-AUC={det.pr_auc:.4f} "
              f"recall={det.hulk_recall:.3f} feat_succ={outcome.feature_success_rate:.2f} "
              f"realisable={outcome.realisable_rate:.2f} "
              f"(+{len(outcome.evasions)} evasions, {len(outcome.realisable_evasions)} realisable)")
        # defender adapts: retrain on accumulated (project()-constrained) evasions
        acc_evasions += outcome.evasions
        Xtr, ytr = adversarial_trainset(Xc, yc, acc_evasions, clean.feature_names, replication)
        det = train_one(model_name, Xtr, ytr, clean.X_test, clean.y_test, seed=seed)
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description="Attack/retrain arms-race loop (Layer C).")
    ap.add_argument("--data", default="data/raw/cicids2017/MachineLearningCVE/"
                                       "Wednesday-workingHours.pcap_ISCX.csv")
    ap.add_argument("--clean-train-size", type=int, default=30000)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--n-attack", type=int, default=30)
    ap.add_argument("--n-refs", type=int, default=8)
    ap.add_argument("--budget", type=int, default=800)
    ap.add_argument("--replication", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig-dir", default="experiments/figures")
    ap.add_argument("--manifold", action="store_true",
                    help="confine the attack to the realisable manifold (the follow-up "
                         "experiment: does adversarial training close the MANIFOLD gap?)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean = build_dataset(args.data, target_label="DoS Hulk", seed=args.seed)
    source = classify(clean.feature_names)
    Xc, yc = subsample_clean(clean, args.clean_train_size, seed=args.seed)
    hulk_pool_rows = np.where(clean.y_train == 1)[0]
    benign_rows = np.where(clean.y_test == 0)[0]
    benign_refs = [clean.X_test.iloc[i].to_dict()
                   for i in benign_rows[: args.n_refs]]
    cfg = AttackConfig(query_budget=args.budget, n_init_refs=args.n_refs, seed=args.seed)

    # Manifold mode: confine the search to the realisable set, so every evasion the
    # defender trains on is realisable by construction and the tracked rate is the
    # TRUE manifold realisable rate (not the free-search post-filter rate).
    projector, tag, mode = None, "", "free-search"
    if args.manifold:
        from functools import partial
        import pandas as pd
        from evasion_arms_race.validation.realisability import fit_calibration, manifold_project
        calib = fit_calibration(pd.concat([clean.X_train, clean.X_test], ignore_index=True))
        projector = partial(manifold_project, calib=calib)
        tag, mode = "_manifold", "manifold-constrained"

    print(f"arms race [{mode}]: {args.rounds} rounds, {args.n_attack} fresh attacks/round, "
          f"clean train {len(yc)}, budget {args.budget}\n")
    records = {}
    for m in MODELS:
        print(f"[{MODEL_LABEL[m]}]")
        records[m] = run_model_loop(m, Xc, yc, clean, source, hulk_pool_rows,
                                    benign_refs, args.rounds, args.n_attack, cfg,
                                    args.replication, args.seed, projector=projector)
        print()

    summary = {"config": {"clean_train_size": len(yc), "rounds": args.rounds,
                          "n_attack": args.n_attack, "budget": args.budget,
                          "replication": args.replication, "seed": args.seed,
                          "mode": mode},
               "records": records}
    out = Path(f"experiments/arms_race{tag}.json"); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    # Trajectory figure: the arms-race dynamics per model.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, m in zip(axes, MODELS):
        rs = [d["round"] for d in records[m]]
        ax.plot(rs, [d["feature_success_rate"] for d in records[m]], "o-", label="evasion success (feature-space)")
        ax.plot(rs, [d["realisable_rate"] for d in records[m]], "s-", label="evasion success (realisable)")
        ax.plot(rs, [d["pr_auc"] for d in records[m]], "^--", label="clean PR-AUC")
        ax.plot(rs, [d["hulk_recall"] for d in records[m]], "v:", label="clean Hulk recall")
        ax.set_title(MODEL_LABEL[m]); ax.set_xlabel("arms-race round")
        ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.3); ax.set_xticks(rs)
    axes[0].set_ylabel("rate")
    axes[1].legend(fontsize=8, loc="center right")
    fig.suptitle(f"Attack/retrain arms race [{mode}] — empirical dynamics (not an equilibrium claim)")
    fig.tight_layout()
    fig_dir = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"arms_race{tag}_trajectory.png", dpi=120)
    plt.close(fig)
    print(f"summary -> {out}   figure -> {fig_dir}/arms_race{tag}_trajectory.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
