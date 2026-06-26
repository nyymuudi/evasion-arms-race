#!/usr/bin/env python3
"""Train the baseline detectors on a CIC-IDS2017 CSV and report.

Usage:
    python scripts/train_baseline.py \
        data/raw/cicids2017/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv

Trains Logistic Regression + Random Forest on the scaled feature space, prints
PR-AUC / ROC-AUC and the top features each model relies on, and saves artifacts
(scaler, models, report) to data/artifacts/ for the attack stage to reuse.

The headline question is NOT "is accuracy high" (it will be) but "which features
drive the decision" -- if they are the volume/rate features behind the DoS
functional floor, Layer A's central tension is confirmed on real data.
"""
from __future__ import annotations

import sys
from pathlib import Path

from evasion_arms_race.data.loader import build_dataset
from evasion_arms_race.detector.baseline import train_baseline, save_artifacts


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = argv[1]
    target = argv[2] if len(argv) > 2 else "DoS Hulk"

    print(f"Loading {path}  (target = {target!r})")
    ds = build_dataset(path, target_label=target, seed=0)
    print(f"  train {len(ds.X_train)}  test {len(ds.X_test)}  "
          f"pos_frac {ds.y_train.mean():.3f}  dropped_nonfinite {ds.n_dropped_nonfinite}")

    print("Training logistic regression + random forest ...")
    tb = train_baseline(ds, seed=0, top_k=15)

    for ev, top in ((tb.eval_logreg, tb.top_logreg), (tb.eval_rf, tb.top_rf)):
        print(f"\n=== {ev.model_name} ===")
        print(f"  PR-AUC  {ev.pr_auc:.4f}")
        print(f"  ROC-AUC {ev.roc_auc:.4f}")
        print(f"  acc     {ev.accuracy:.4f}   (context only)")
        print("  top features:")
        for name, val in top.ranked:
            print(f"    {name:32s} {val:+.4f}")

    save_artifacts(tb)
    print(f"\nArtifacts saved to data/artifacts/  (scaler, logreg, rf, report)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))