#!/usr/bin/env python3
"""Run detector ablation diagnostics on a CIC-IDS2017 CSV.

Usage:
    python scripts/run_ablation.py \
        data/raw/cicids2017/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv

Retrains the detector on reduced feature sets and reports PR-AUC deltas vs the
full-feature baseline. Read the results like this:

  * drop_destination_port near 0  -> port is not an artifactual shortcut (good)
  * drop_destination_port large   -> port carried the signal; treat with care
  * controllable_only near 0      -> detector still works on attacker-controlled
                                     features alone; evasion is genuinely hard
                                     and meaningful (the interesting case)
  * controllable_only large       -> detector's power lives in FROZEN features
                                     the attacker cannot change; a hard ceiling
                                     on what evasion can achieve -- worth knowing
  * any single drop:: large       -> single point of failure / likely artifact
"""
from __future__ import annotations

import sys

from evasion_arms_race.data.loader import build_dataset
from evasion_arms_race.detector.ablation import run_ablations


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = argv[1]
    target = argv[2] if len(argv) > 2 else "DoS Hulk"
    print(f"Loading {path}  (target = {target!r})")
    ds = build_dataset(path, target_label=target, seed=0)
    print(f"  {len(ds.X_train)} train / {len(ds.X_test)} test, "
          f"{len(ds.feature_names)} features\n")
    print("Running ablations (each retrains both models) ...\n")
    res = run_ablations(ds, seed=0)
    print(f"{'ablation':34s} {'nfeat':>5s} {'PR_lr':>7s} {'d_lr':>8s} "
          f"{'PR_rf':>7s} {'d_rf':>8s}")
    print("-" * 76)
    for r in res:
        print(f"{r.name:34s} {r.n_features:5d} {r.pr_auc_logreg:7.4f} "
              f"{r.delta_logreg:+8.4f} {r.pr_auc_rf:7.4f} {r.delta_rf:+8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))