# Game-theoretic framing of the detector–attacker interaction (Layer C, phase 4)

This document does the thing the arms-race literature usually skips: it states the
detector-vs-attacker interaction as a game *and then checks, honestly, which of the
preconditions for an equilibrium claim actually hold.* The conclusion is deliberately
modest. The empirical loop in `experiments/arms_race.py` measures an adaptive
dynamic; it does **not** compute a Nash equilibrium, and this document explains
precisely why the usual existence theorems do not apply here.

The discipline is borrowed from the project's prior life (a CFR poker solver). There,
"convergence to equilibrium" is a theorem with hypotheses one can verify. Importing
the *vocabulary* of equilibrium without importing the *hypotheses* is the error this
document is written to avoid.

## 1. The game

**Players.** A defender `D` and an attacker `A`.

**Strategy spaces.**
- `D` chooses a detector. In this setup the action is a *training-set augmentation
  policy*; retraining (empirical risk minimisation) maps the augmented set to model
  parameters `θ` (logistic-regression weights, or a random-forest structure). The
  reachable set `Θ` is therefore the *image of ERM over augmented datasets* — a
  discrete, combinatorial, non-convex subset of parameter space, not an arbitrary
  ball in `R^d`.
- `A` chooses perturbations of attack flows within the **feasible set** `F`: the
  realisability-constrained manifold defined in Layers A and validated in Layer C's
  item 7 — frozen features fixed, the DoS floor, and the packet-level moment
  constraints (e.g. the Bhatia–Davis bound `Var ≤ (max−mean)(mean−min)`). `F` is
  bounded but **non-convex** (the variance bound alone carves out a non-convex
  region) and possibly disconnected.

**Utilities.**
- `A` is paid the evasion success rate — the fraction of its feasible attacks that
  the current detector labels benign — optionally net of perturbation magnitude.
- `D` is paid detection performance, but on **two axes**: performance on the clean
  distribution (PR-AUC / Hulk recall on held-out clean traffic) and performance on
  the adversarial distribution. Hardening trades the second against the first.

**Information and timing.** Black-box: `A` has *query* access to `D`'s current
detector, not to `θ`; `D` observes `A`'s past attack samples. The interaction is
**sequential and repeated**, not one-shot simultaneous.

## 2. Degree of zero-sum

On the evaded samples, `A`'s gain is `D`'s loss — a zero-sum core. But `D` also pays
for clean-performance loss, a dimension that is **not** part of `A`'s payoff. The
full game is therefore **general-sum**: "almost zero-sum" on the adversarial axis,
with an orthogonal clean-accuracy cost that no minimax reduction captures. This
matters: the von Neumann minimax theorem is a statement about *zero-sum* games, and
this is not strictly one.

## 3. Do the equilibrium preconditions hold? (the honest checklist)

| Precondition (for the relevant existence theorem) | Holds here? |
|---|---|
| **Finite strategy sets** (Nash 1950) | **No.** Perturbations and parameters are continuous; the game is infinite. |
| **Compact strategy spaces** (Glicksberg/Debreu/Fan) | **Partially.** `F` is bounded/closed but non-convex; `Θ` is the combinatorial image of ERM, not a compact convex set. |
| **Continuous utilities** (Glicksberg, for mixed-Nash existence) | **No.** Classification is 0/1; evasion success jumps as a perturbation crosses the boundary; PR-AUC is piecewise-constant in `θ`. Payoffs are discontinuous. |
| **Quasi-concave / convex–concave utilities** (pure-Nash; von Neumann minimax) | **No.** Evasion success is non-concave in the perturbation; detection loss is non-convex in `θ` (the random forest is piecewise-constant — non-convex and non-smooth). |
| **Zero-sum** (minimax / saddle-point) | **No (general-sum).** See §2. |

Every standard existence theorem requires hypotheses this setup violates. The
consequence is blunt and worth stating plainly: **there is no guarantee that a Nash
equilibrium exists here, and certainly none that the iterated-retraining dynamic
converges to one.** A setup can fail to have a (pure or mixed) equilibrium, or have
one that the specific dynamic never reaches.

## 4. What the empirical fixed point *does* mean

Suppose the loop reaches a state where the attacker's best *feasible* response yields
no meaningful additional success and the defender's retraining no longer changes the
detector. That is an **empirical fixed point of one particular best-response-style
dynamic** — operationally, "no profitable feasible deviation was observed within the
attack's query budget." Three honest qualifications:

1. It is **contingent on the attacker's power.** A stronger search (more budget, a
   different algorithm, white-box gradients) might surface deviations this one missed.
   A fixed point of *this* dynamic is not a fixed point of *all* dynamics.
2. It is **contingent on the retraining procedure** (ERM on clean ∪ augmentation with
   a fixed oversampling rate). A different defender update could move from it.
3. It carries **no optimality certificate.** Unlike an ε-Nash equilibrium, it does
   not bound how much either player could gain by deviating in principle — only what
   was observed.

This is a legitimate and useful object. It is just not an equilibrium, and we do not
call it one.

## 5. Relation to CFR / poker solving — conceptual, not formal

The temptation is to say "this is like CFR converging to a Nash equilibrium." It is
not, and the difference is exactly the set of hypotheses in §3.

- **CFR** is a *no-regret* algorithm with a *theorem*: in a **finite, two-player,
  zero-sum, extensive-form** game, if both players' regrets grow sublinearly then the
  **time-averaged** strategies converge to a Nash equilibrium (regret → 0 ⇒ the
  average strategy is an ε-equilibrium). Every hypothesis — finiteness, zero-sum, the
  regret-matching update — is satisfied by construction.
- **The arms-race loop** satisfies *none* of these: the game is infinite, general-sum,
  with discontinuous non-convex payoffs, and ERM-retraining is **not** a no-regret
  procedure — it carries no regret bound, and we track *last-iterate* detectors, not a
  time-average with any guarantee.

So the analogy is genuine but strictly **conceptual**: both are iterated processes in
which two strategies co-adapt, each responding to the other, and one can *watch* the
trajectory for convergence, oscillation, or divergence. The poker solver's curve is
backed by a convergence theorem; the arms-race curve is an **empirical observation**
with no such backing. Treating the two as mathematically equivalent would be a
category error, and the project does not.

## 6. Summary

The detector–attacker interaction is a genuine game, but an infinite, general-sum one
with discontinuous, non-convex payoffs and a black-box, repeated information
structure. None of the classical equilibrium-existence theorems apply. The
contribution of Layer C is therefore **empirical**: it measures whether a specific,
realisability-constrained attack/retrain dynamic converges, and what that fixed point
operationally means — while keeping the equilibrium vocabulary on the shelf, where,
for this setup, it belongs.
