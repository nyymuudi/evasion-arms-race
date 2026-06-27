# evasion-arms-race: synthesis

*One through-line ties three layers together: the **feasibility constraint** — the
rule that an attacker may only move what it actually controls, and only to values a
real packet stream could produce. The constraint is the apparatus of all three layers:
Layer A measures **evasion** under it, Layer B **poisoning** under it, Layer C the
**dynamics** of the iterated attack/retrain game under it. The hardest lesson is
methodological: the constraint must shape the **search**, not be applied as a filter —
and when it does, evasions that a post-filter called impossible reappear. The honest
picture is not "realisability saves the detector" (a headline this project raised and
then withdrew) but "realisability is decisive to measure correctly, and most ways of
measuring it are wrong."*

This document is written to survive a technical reader. Where a result is negative,
partial, or a correction of an earlier claim, it is reported as such; the project's
value is in the discipline and the self-correction, not in a dramatic collapse.

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

## Layer A — measuring evasion under the constraint (and a corrected headline)

A decision-based **boundary attack** (consistent with the black-box model) calls the
feasibility projection on every candidate, then item 7 validates the survivors at the
packet level.

- **In feature space, both detectors are fully evadable** — 100% success for logistic
  regression and the random forest alike. Taken alone, this is the usual (inflated)
  adversarial-ML headline.
- **A realisability *post-filter* seemed to reverse this.** Of the free-search
  evasions, the fraction passing the packet-level feasibility check is **52% (LR) and
  0% (RF)** — every random-forest "evasion" failing on moments (variance beyond the
  Bhatia–Davis bound, `IAT max > duration`, `Total ≠ N·mean`) no multiset can realise.
  The first draft of this synthesis concluded the random forest was robust to
  realisable attacks. **That conclusion was wrong, and a pre-registered stress test
  caught it.**
- **Constraining the *search* to the manifold overturns it.** Building the
  realisability projection into the search (`manifold_project`, not a post-filter)
  takes realisable evasion to **85% (LR) and 100% (RF)**, converged (flat to a 2000-query
  budget). The 0% was a **search artefact**: a free-space search simply never looked on
  the manifold, and the filter then declared its off-manifold output "impossible". The
  honest result is that *both* detectors are evadable by realisable traffic; the
  methodological lesson — realisability must shape the search, not be bolted on as a
  filter — is the sharper contribution. Full account in `docs/manifold_experiment.md`.

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
  realisable rate. With the corrected (manifold) rate — ≈85% (LR), ≈100% (RF) — that
  channel is *open*, not closed: the earlier reading that the random forest could not
  be poisoned through it depended on the 0% that the Layer A stress test withdrew.

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
  non-decreasing feature-space success. Neither trajectory oscillates or diverges.
  **Caveat (Layer A correction):** this loop measures the *free-search + post-filter*
  realisable rate, which the manifold experiment shows under-reports realisable
  evasion. So "43% → 0%" describes the free-search realisable gap closing; whether
  adversarial training closes the larger *manifold* gap is open. Re-running the loop
  with the manifold-constrained attack is the natural next step.
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

1. **How you enforce realisability decides the result.** Feature-space success (100%)
   over-states evasion; a realisability *post-filter* on a free-space search
   *under*-states it (RF 0%); only building realisability into the search measures it
   honestly (RF 100%). The same attack and the same feasibility definition yield 0% or
   100% depending solely on *where the search was allowed to look*. Reporting a
   post-filtered rate as if it measured feasible robustness is a third way — beyond the
   well-known feature-space inflation — that adversarial-ML numbers mislead.
2. **Under a correct (manifold-constrained) search, both detectors are evadable by
   realisable traffic** — LR ≈85%, RF ≈100%. The project's first headline ("realisability
   saves the random forest") was a measurement artefact, and the project overturned it
   with a pre-registered stress test rather than defending it. The robustness results
   that survive are narrower and threshold-specific (Layer B: ranking PR-AUC resists
   poisoning; the operating point does not), and are reported as such.
3. **Discipline with terminology, and the willingness to be wrong.** Empirical dynamics
   are not equilibria; feature-space success is not realisable success; a near-perfect
   PR-AUC is a prompt to investigate, not a trophy — and a headline that fails its own
   stress test is corrected, not buried. The instrument that caught the error
   (`docs/manifold_experiment.md`) is part of the contribution.

## Reproducibility

Every figure and numeric summary is produced by a script on the reproducible path
(`metrics/report.py`, `attack/poisoning.py`, `validation/realisability.py`,
`experiments/arms_race.py`); figures land in `experiments/figures/` (gitignored),
small numeric summaries in version-controlled JSON. The test suite covers the pure
logic of every layer.
