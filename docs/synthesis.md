# evasion-arms-race: synthesis

*One through-line ties three layers together: the **feasibility constraint** — the
rule that an attacker may only move what it actually controls, and only to values a
real packet stream could produce. Layer A shows the constraint bounds **evasion**;
Layer B shows it bounds **poisoning**; Layer C shows how it **shapes the dynamics**
of the iterated attack/retrain game. Remove the constraint and every result inflates
into the usual, unrealisable adversarial-ML numbers. Keep it, and the picture is
quieter, more honest, and — for a defender — far more actionable.*

This document is written to survive a technical reader. Where a result is negative or
partial, it is reported as such; the project's value is in the discipline, not in a
dramatic collapse.

## The object of study

A two-player interaction on **CIC-IDS2017**, DoS Hulk vs benign, with CICFlowMeter
flow features. The detector reads 78 flow statistics; the attacker emits traffic. The
threat model is **black-box** throughout (query access to the detector's decision, no
gradients, no parameters) — the realistic and harder setting.

The pivot that makes the project more than a demo: every feature is partitioned by
*who controls it* (controllable / constrained / frozen / derived), and a **feasibility
projection** confines any perturbation to (a) features the attacker sets, (b) within
protocol-legal and DoS-functional bounds, (c) recomputing the dependent aggregates.
Layer A item 7 adds the packet-level check that the moments are realisable at all.

## Layer A — the constraint bounds evasion

A decision-based **boundary attack** (consistent with the black-box model) calls the
feasibility projection on every candidate, then item 7 validates the survivors at the
packet level.

- **In feature space, both detectors are fully evadable** — 100% success for logistic
  regression and the random forest alike. Taken alone, this is the usual (inflated)
  adversarial-ML headline.
- **Under realisability, the picture reverses.** Of the successful evasions, the
  fraction that correspond to a *sendable packet stream* is **52% (logistic
  regression) and 0% (random forest)**. Every random-forest "evasion" is statistically
  infeasible — the search produced flows whose moments (e.g. variance beyond the
  Bhatia–Davis bound, `IAT max > duration`, `Total ≠ N·mean`) no packet multiset can
  realise.
- So the random forest's apparent vulnerability was an **artefact of an over-powered
  attacker roaming infeasible feature space.** The ablation diagnostics had already
  shown the discriminative signal is distributed and largely lives in features the
  attacker cannot touch; item 7 turns that into a hard, quantified realisability gap.

The five packet-length aggregates once deferred as "unreconstructable" turned out to
be reconstructable in closed form (Min/Max exactly; Mean/Std/Variance via the law of
total variance plus a data-fit calibration, R² ≥ 0.997) — an honest correction of an
earlier over-pessimistic assumption.

## Layer B — the constraint bounds poisoning

The attacker injects budget-limited, **realisability-projected** Hulk samples
(labelled benign) into the detector's retraining set; evaluation is on clean held-out
data. Same projection, same DoS floor as Layer A — realistic poison, not feature-space
dust.

- **Threshold-independent detection barely moves.** PR-AUC on clean test is nearly
  flat even at a 20% poison budget (LR 0.9993 → 0.9968, RF 1.0000 → 0.9996). The
  strong Hulk/benign separability resists poisoning of the *ranking*. Realistic
  poisoning does **not** collapse detection at modest budgets — an honest negative
  result.
- **The operating point does degrade.** At the deployed 0.5 threshold, Hulk recall
  falls to ≈0.70 (LR) / ≈0.75 (RF) at 20% label-flip poison — the poison shifts
  predicted probabilities below threshold without destroying separability. The
  operationally-honest metric (recall at the deployed threshold) shows the real, if
  budget-hungry, effect.
- **Strategy matters as theory predicts at the margins:** boundary-selected poison is
  more sample-efficient at low budgets; mass label-flip does more damage at high
  budgets; the random forest is the more poison-robust.
- **Corollary tying back to Layer A:** the *realistic* auto-labelling attacker (whose
  poison must be a successful realisable evasion) has its budget gated by Layer A's
  realisable rate — 52% for LR, **0% for RF**. The random forest cannot be poisoned
  through that channel at all.

## Layer C — the constraint shapes the arms race

Adversarial training (`detector/robust.py`) folds the attacker's feasibility-projected
evasions back into training (labelled with their *correct* class — the defensive dual
of Layer B), and `experiments/arms_race.py` iterates attack → retrain → attack,
logging the trajectory. The analysis is split, deliberately, into the part that is
always doable and the part that is not:

- **Empirical dynamics (always doable).** Read through the realisable-evasion rate, the
  logistic-regression loop **converges within ≈2 rounds**: 43% → 3% → 0% → 0%, at a
  negligible clean cost (PR-AUC 0.9993 → 0.9987; Hulk recall flat at 0.998). The
  attacker's feature-space success stays pinned at 100% — but those evasions become
  entirely infeasible, so the apparent permanent stalemate on the feature-space curve
  is an artefact; the realisability lens shows the defender closes the only contest
  that matters. The random forest is already at this fixed point from round 0 (0%
  realisable throughout — nothing to harden against), with low (≈30-40%) and
  non-decreasing feature-space success. Neither trajectory oscillates or diverges;
  both reach the same empirical fixed point — *no realisable evasion available to the
  attacker*. The feasibility constraint is thus not only what bounds the attack
  (Layers A-B) but the very axis on which the defence operates and the loop settles.
- **Equilibrium claims (mostly *not* available here).** `docs/game_theory.md` states
  the interaction as a game and checks the preconditions: the strategy spaces are
  infinite and non-convex, the payoffs discontinuous and non-concave, the game
  general-sum (clean accuracy is an axis the attacker does not pay for). **No standard
  Nash/minimax existence theorem applies.** An empirical fixed point here means "no
  profitable feasible deviation observed within the attack's budget" — an operational
  notion, not a proven equilibrium, and we do not call it one.

The relation to the project's CFR poker-solver heritage is **conceptual, not formal**:
both are iterated processes in which two strategies co-adapt and one watches the curve
for convergence. CFR's convergence is a *theorem* (finite, zero-sum, no-regret); the
arms race's behaviour is an *observation* (infinite, general-sum, ERM-retraining with
no regret bound). Keeping that distinction is the point of Layer C.

## What the project demonstrates

1. **Realisability is the difference between a number and a result.** The same attack
   reads as "100% evasion" or "0–52% realisable evasion" depending solely on whether
   the feasibility constraint is enforced. Reporting the former without the latter is
   how adversarial-ML success rates become fiction.
2. **A strong, separable detector is hard to break *realistically*.** Across all three
   layers, the honest finding is robustness, not collapse — bounded evasion, bounded
   poisoning, a dynamic that does not run away. The interesting science is in
   *measuring the bound*, not in manufacturing a breach.
3. **Discipline with terminology.** The project deliberately under-claims: empirical
   dynamics are not equilibria, feature-space success is not realisable success, and a
   near-perfect PR-AUC is treated as a warning to investigate (ablation, realisability)
   rather than a trophy.

## Reproducibility

Every figure and numeric summary is produced by a script on the reproducible path
(`metrics/report.py`, `attack/poisoning.py`, `validation/realisability.py`,
`experiments/arms_race.py`); figures land in `experiments/figures/` (gitignored),
small numeric summaries in version-controlled JSON. The test suite covers the pure
logic of every layer.
