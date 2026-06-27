"""Manifold-constrained attack: does the headline claim survive its objection?

The headline claim — "feature-space evasion is 100% but realisable evasion is only
52% (LR) / 0% (RF)" — rests on a free-space search followed by a realisability
POST-FILTER. The objection (see the experiment brief / docs): a 0% obtained by
post-filtering is ambiguous. It could mean (a) the detector is robust to FEASIBLE
attacks, or (b) the search merely wandered off-manifold and the filter zeroed it.
The claim needs (a); post-filtering cannot distinguish them.

This experiment removes the ambiguity by running the SAME boundary attack with the
realisability projection built INTO the search (`manifold_project`), so every
candidate the detector sees already passes `is_feasible` by construction. It also
adds search POWER (budget, restarts) so a low rate cannot be dismissed as a weak
attacker. Three configurations, both detectors:

  (i)   free search + post-filter      — the baseline (reproduces item 7)
  (ii)  manifold-constrained search    — same budget, on-manifold
  (iii) manifold + extra power         — higher budget + more restarts

Key number: does the Random Forest's realisable rate stay ~0% under (ii)/(iii)?

Run:  python experiments/manifold_attack.py --n-samples 20
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from evasion_arms_race.attack.blackbox import (
    AttackConfig, Oracle, attack_sample, load_artifacts,
)
from evasion_arms_race.data.loader import build_dataset
from evasion_arms_race.features.rule_classifier import classify
from evasion_arms_race.validation.realisability import (
    fit_calibration, is_feasible, manifold_project,
)

MODELS = ("logreg", "rf")
MODEL_LABEL = {"logreg": "Logistic Regression", "rf": "Random Forest"}


def _run_config(model, scaler, fn, hulk, refs, source, projector, cfg, post_filter):
    """Attack every sample with the given projector/config. realisable_rate is the
    fraction of successes that pass is_feasible (== success rate when the projector
    is manifold_project, since those are feasible by construction)."""
    n = len(hulk)
    succ, realisable, first_evasions = 0, 0, []
    for clean in hulk:
        oracle = Oracle(model, scaler, fn)
        r = attack_sample(clean, oracle, refs, cfg, source, projector=projector)
        if r.success and r.best_vector:
            succ += 1
            feasible = is_feasible(r.best_vector).feasible
            if (not post_filter) or feasible:
                if feasible:
                    realisable += 1
            if r.queries_to_first_evasion is not None and feasible:
                first_evasions.append(r.queries_to_first_evasion)
    return {"feature_success_rate": succ / n,
            "realisable_rate": realisable / n,
            "first_evasion_queries": first_evasions}


def main() -> int:
    ap = argparse.ArgumentParser(description="Manifold-constrained attack experiment.")
    ap.add_argument("--data", default="data/raw/cicids2017/MachineLearningCVE/"
                                       "Wednesday-workingHours.pcap_ISCX.csv")
    ap.add_argument("--artifacts", default="data/artifacts")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--n-refs", type=int, default=8)
    ap.add_argument("--budget", type=int, default=800)
    ap.add_argument("--power-budget", type=int, default=2000)
    ap.add_argument("--power-refs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig-dir", default="experiments/figures")
    ap.add_argument("--summary-out", default="experiments/manifold_attack.json")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    art = load_artifacts(args.artifacts)
    fn = art.feature_names
    source = classify(fn)
    clean = build_dataset(args.data, target_label="DoS Hulk", seed=args.seed)
    calib = fit_calibration(pd.concat([clean.X_train, clean.X_test], ignore_index=True))
    mp = partial(manifold_project, calib=calib)

    rng = np.random.default_rng(args.seed)
    hulk_idx = np.where(clean.y_test == 1)[0]
    ben_idx = np.where(clean.y_test == 0)[0]
    hulk = [clean.X_test.iloc[i].to_dict()
            for i in rng.choice(hulk_idx, min(args.n_samples, hulk_idx.size), replace=False)]
    refs = [clean.X_test.iloc[i].to_dict() for i in ben_idx[: args.power_refs]]

    cfg = AttackConfig(query_budget=args.budget, n_init_refs=args.n_refs, seed=args.seed)
    cfg_pow = AttackConfig(query_budget=args.power_budget, n_init_refs=args.power_refs, seed=args.seed)
    configs = [
        ("free_postfilter", None, cfg, True),       # projector None -> project (free space)
        ("manifold", mp, cfg, False),
        ("manifold_power", mp, cfg_pow, False),
    ]

    summary = {"config": {"n_samples": len(hulk), "budget": args.budget,
                          "power_budget": args.power_budget, "seed": args.seed},
               "results": {}}
    print(f"=== Manifold-constrained attack ({len(hulk)} samples) ===\n")
    for m in MODELS:
        summary["results"][m] = {}
        print(f"[{MODEL_LABEL[m]}]")
        for name, proj, c, pf in configs:
            res = _run_config(art.model(m), art.scaler, fn, hulk, refs, source,
                              proj, c, post_filter=pf)
            summary["results"][m][name] = res
            print(f"  {name:16s}: feature-space {res['feature_success_rate']:.2f}  "
                  f"REALISABLE {res['realisable_rate']:.2f}")
        print()

    # Figure 1: realisable rate per config, grouped by model.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cfgs = ["free_postfilter", "manifold", "manifold_power"]
    x = np.arange(len(cfgs)); w = 0.38
    for j, m in enumerate(MODELS):
        ys = [summary["results"][m][c]["realisable_rate"] for c in cfgs]
        ax.bar(x + (j - 0.5) * w, ys, w, label=MODEL_LABEL[m])
    ax.set_xticks(x); ax.set_xticklabels(["free +\npost-filter", "manifold", "manifold +\npower"])
    ax.set_ylabel("realisable evasion rate"); ax.set_ylim(0, 1.05)
    ax.set_title("Does realisable evasion survive a manifold-constrained search?")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig_dir = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "manifold_realisable_rate.png", dpi=120)
    plt.close(fig)

    # Figure 2: budget vs realisable success for the manifold_power config
    # (convergence evidence — the result does not improve with more budget).
    fig, ax = plt.subplots(figsize=(8, 4.5))
    budgets = np.linspace(1, args.power_budget, 40).astype(int)
    for m in MODELS:
        fe = summary["results"][m]["manifold_power"]["first_evasion_queries"]
        ys = [sum(1 for q in fe if q <= B) / len(hulk) for B in budgets]
        ax.plot(budgets, ys, marker="o", ms=3, label=MODEL_LABEL[m])
    ax.set_xlabel("query budget (feasible queries)")
    ax.set_ylabel("realisable evasion rate (manifold)")
    ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("Manifold attack: realisable evasion vs query budget (convergence)")
    fig.tight_layout()
    fig.savefig(fig_dir / "manifold_budget_curve.png", dpi=120)
    plt.close(fig)

    out = Path(args.summary_out); out.parent.mkdir(parents=True, exist_ok=True)
    # drop the per-sample query lists from the JSON headline (keep it small)
    slim = {"config": summary["config"], "results": {
        m: {c: {"feature_success_rate": d["feature_success_rate"],
                "realisable_rate": d["realisable_rate"],
                "n_first_evasions": len(d["first_evasion_queries"])}
            for c, d in summary["results"][m].items()} for m in MODELS}}
    out.write_text(json.dumps(slim, indent=2))
    print(f"figures -> {fig_dir}/   summary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
