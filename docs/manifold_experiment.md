# Does the headline claim survive a manifold-constrained search?

This experiment stress-tests the project's central claim against its most obvious
objection. Both outcomes were **pre-registered** before the run; the result section
records which one occurred and what the project may therefore claim.

## The claim and the objection

**Claim (as originally stated).** Feature-space evasion succeeds 100% against both
detectors, but realisable evasion is only 52% (logistic regression) and **0% (Random
Forest)** — so the Random Forest's apparent vulnerability vanishes under the
feasibility constraint.

**Objection.** That 0% came from a *free-space* boundary search followed by a
realisability **post-filter** (`is_feasible` applied after the fact). A 0% obtained
this way is ambiguous:

- **(a)** the Random Forest is genuinely robust to *feasible* attacks, or
- **(b)** the search simply wandered through infeasible feature space and the filter
  zeroed the result — the search never looked on the manifold at all.

The headline needs (a). Post-filtering cannot distinguish (a) from (b), because the
search was never constrained to the realisable set.

## The critical distinction (two independent dimensions)

- **Dimension 1 — constrain the search space.** Build the realisability projection
  *into* the search so every candidate stays on the manifold. This is what addresses
  the objection.
- **Dimension 2 — increase search power.** More queries, more restarts. *On its own*
  this only strengthens the objection (a stronger search in free space), it does not
  answer it.

A valid test needs **both**: a *strong* search operating *only* on the realisable
manifold. Only then does a surviving 0% mean "a strong attacker finds no feasible
evasion" rather than "a weak attacker missed it".

## Method

`validation.realisability.manifold_project` confines the search: it applies the
existing `project()` and then a deterministic, idempotent clamp-and-recompute that
enforces exactly the `is_feasible` constraints (`min ≤ mean ≤ max`, `Total = N·mean`,
the Bhatia–Davis variance bound, `IAT max ≤ duration`, `IAT Total = (N−1)·mean`) and
recomputes the dependent aggregates — so the output passes `is_feasible` *by
construction*. The boundary attack (`attack/blackbox.py`) takes a `projector`
argument; swapping `project` for `manifold_project` confines the entire search to the
manifold without touching the search algorithm. Three configurations are compared,
both detectors (`experiments/manifold_attack.py`):

1. **free + post-filter** — the original baseline (reproduces item 7).
2. **manifold** — same budget, search constrained to the manifold.
3. **manifold + power** — higher query budget and more restarts (dimension 2), to
   rule out a weak-attacker artefact; a budget-vs-success curve shows convergence.

Under (2) and (3) the realisable rate equals the success rate, because every success
is feasible by construction.

## Pre-registered interpretation

- **If the Random Forest realisable rate stays ≈0% under (2)/(3):** branch (a). The
  claim is *upgraded from speculation to evidence* — the Random Forest is robust to
  realisable attacks, the 0% was not a search artefact.
- **If it rises materially:** branch (b). The original conclusion was partly a
  search/measurement artefact and must be **corrected**. This sits squarely in the
  project's ethos ("investigate, don't trophy"; never hide a negative result): the
  lesson becomes that realisability must shape the *search*, not be applied as a
  *filter*, and that post-hoc filtering can badly mislead.

## Result

N = 20 DoS-Hulk samples, both detectors, seed 0 (`experiments/manifold_attack.json`):

| config | LR realisable | RF realisable |
|---|---|---|
| free + post-filter | 0.45 | **0.00** |
| manifold | 0.85 | **1.00** |
| manifold + power (budget 800 → 2000, more restarts) | 0.85 | **1.00** |

The manifold budget-vs-success curve plateaus within ≈50 queries and is flat to 2000
queries, for both detectors — so the rates above are converged ceilings, not
budget-limited (dimension 2 is satisfied: more power does not change the answer).

## Verdict — branch (b): the claim is corrected

**The Random Forest's 0% realisable evasion was a search artefact, not robustness.**
Constraining the *same* attack to the realisable manifold takes the Random Forest from
0% to **100%** realisable evasion, and logistic regression from 45% to **85%** —
unchanged by extra budget. The original 0%/52% figures measured only that a *free-space*
search produces vectors which mostly fail the realisability filter; they did **not**
measure whether feasible evasions exist. They do, in abundance.

Three consequences, stated plainly:

1. **Corrected headline.** Both detectors are evadable by realisable traffic
   (LR ≈85%, RF ≈100%) once the search is confined to the manifold. The earlier
   "RF vulnerability vanishes under realisability" is withdrawn.
2. **The methodological contribution sharpens.** Realisability must shape the *search*,
   not be applied as a *post-filter*. Post-hoc filtering of a free-space search
   systematically *under-reports* realisable evasion (here: by 0.85 absolute for the
   Random Forest) and can invert the qualitative conclusion. This is the transferable
   lesson, and it indicts a common pattern in adversarial-NIDS evaluation.
3. **Scope and remaining stringency.** "Realisable" here means "passes `is_feasible`",
   the project's operative definition (moment-consistency + closed-form pooled
   features) — the *same* definition used for the original claim, so the comparison is
   apples-to-apples. A stricter oracle (the deferred CICFlowMeter-binary pcap
   round-trip) could lower the absolute rates; it would not rescue the original 0%,
   because the manifold search already exhibits feasible evasions the post-filter
   declared impossible.

This reversal is the system working as intended: a pre-registered stress test of the
project's own headline, reported honestly when it failed. A modest, exact result
("realisable evasion is high once you search the manifold; post-filtering misleads")
is worth more than an oversold one ("realisability saves the Random Forest").

## Follow-up: does adversarial training close the *manifold* gap?

The correction above also indicts Layer C, which had measured the arms race with the
free-search + post-filter (so its "LR realisable 43% → 0%" was the post-filter gap
closing, not the real one). The natural question: re-run the attack/retrain loop with
the **manifold-constrained** attack — does adversarial training on genuinely realisable
evasions robustify the detector against realisable attacks? (`experiments/arms_race.py
--manifold`; clean train 30k, 30 fresh attacks/round, budget 800, 4 rounds.)

| round | LR realisable | RF realisable |
|---|---|---|
| 0 (clean detector) | 1.00 | 0.97 |
| 1 | 1.00 | 0.30 |
| 2 | 1.00 | 0.33 |
| 3 | 1.00 | 0.40 |

Clean-test PR-AUC stays ≈flat (LR 0.9993 → 0.9986; RF ≈1.0), so cost is negligible.
The answer is detector-dependent and, in both cases, **the gap does not close**:

- **Logistic regression: adversarial training does nothing.** Realisable evasion stays
  pinned at **100%** for all four rounds. A linear boundary cannot carve the benign
  half of a large manifold out from under the attacker while preserving clean accuracy;
  fresh realisable evasions are always available. The free-search loop's tidy "43% → 0%
  convergence" was an artefact of the post-filter — the true (manifold) dynamic shows no
  robustification at all.
- **Random forest: partial, and it plateaus.** Realisable evasion drops sharply after
  one round (97% → 30%) — the forest memorises the round-0 evasion regions as attack
  leaves — but then **stalls around 30–40% and even drifts back up** as the attacker
  finds realisable evasions in the leaves the defence has not covered. The defence
  reduces but never eliminates realisable evasion.

**Conclusion.** Adversarial training against a feasibility-constrained attacker buys
*limited, detector-dependent, and incomplete* robustness here — nothing for logistic
regression, a partial and plateauing reduction for the random forest — and closes the
realisable gap for neither. As with the headline, this is only visible because the
attack searches the manifold: the post-filtered free-search loop reported a reassuring
"converges to zero" for logistic regression that the manifold loop flatly contradicts.
The measurement method is decisive for the *defence* conclusion too, not only the attack.
