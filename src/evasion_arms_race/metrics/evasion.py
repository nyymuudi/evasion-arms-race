"""Evasion metrics (Layer A, todo item 6).

Pure, matplotlib-free, dependency-light functions that turn item-5's per-sample
attack results into a scientifically reportable picture. Everything here operates
on plain sequences so it is trivially unit-testable; plotting and file I/O live
in `metrics.report`.

The headline metric is NOT a single success rate. A success rate without the
perturbation cost that bought it is meaningless: an attacker that "succeeds" only
by moving controllable features arbitrarily far has not demonstrated much. So the
core object is the **success-vs-perturbation tradeoff curve**, plus a three-class
decomposition that separates the project's central failure mode -- evasion
blocked by the DoS feasibility floor -- from a detector that is simply strong.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# 1. Success vs allowed perturbation magnitude (the tradeoff curve)
# --------------------------------------------------------------------------- #
def default_eps_grid(l2_values: Sequence[float], n: int = 25) -> np.ndarray:
    """A 0..max(L2) grid for the perturbation-budget axis. Empty/all-zero input
    falls back to a unit grid so callers never divide by zero."""
    finite = [v for v in l2_values if np.isfinite(v) and v > 0]
    hi = max(finite) if finite else 1.0
    return np.linspace(0.0, hi, n)


def perturbation_curve(
    l2: Sequence[float], success: Sequence[bool], eps_grid: Sequence[float]
) -> np.ndarray:
    """Fraction of ALL samples evaded with controllable-L2 <= eps, for each eps.

    A failed sample never counts (its perturbation did not buy an evasion). The
    curve is monotone non-decreasing and plateaus at the overall success rate.

    Interpretation note (honest): because the boundary attack reports the
    SMALLEST perturbation it FOUND (not a certified minimum), this curve is a
    LOWER BOUND on the success achievable at each budget -- the true minimal
    perturbation could be smaller, never larger.
    """
    l2 = np.asarray(l2, dtype=float)
    success = np.asarray(success, dtype=bool)
    n = len(l2)
    if n == 0:
        return np.zeros(len(eps_grid))
    # per-sample evasion cost: inf if it never succeeded
    cost = np.where(success, l2, np.inf)
    return np.array([(cost <= e).sum() / n for e in eps_grid])


# --------------------------------------------------------------------------- #
# 2. Success / three-class decomposition vs query budget
# --------------------------------------------------------------------------- #
def budget_curve(
    first_evasion: Sequence[int | None], n_samples: int, budgets: Sequence[int]
) -> np.ndarray:
    """Fraction of samples evaded within B feasible queries, for each B.

    Uses the per-sample query index at which the first benign seed was found, so
    one full-budget run yields the whole curve without re-running.
    """
    fe = [q for q in first_evasion]
    return np.array([
        sum(1 for q in fe if q is not None and q <= B) / n_samples
        for B in budgets
    ])


def decompose_at_budget(
    first_evasion: Sequence[int | None],
    first_floor_block: Sequence[int | None],
    budget: int,
) -> dict[str, int]:
    """Classify every sample at query budget B into exactly one of three classes:

        success           : a benign feasible seed was found by query B.
        feasibility_bound  : not yet evaded, but the projection had already
                             reverted >=1 label flip by B -> the DoS floor is the
                             active obstacle (the project's central failure mode).
        detector_bound     : not yet evaded and not even an un-projected flip was
                             floored by B -> the detector resists on the movable
                             subspace.
    """
    succ = feas = det = 0
    for fe, ff in zip(first_evasion, first_floor_block):
        if fe is not None and fe <= budget:
            succ += 1
        elif ff is not None and ff <= budget:
            feas += 1
        else:
            det += 1
    return {"success": succ, "feasibility_bound": feas, "detector_bound": det}


def decomposition_series(
    first_evasion: Sequence[int | None],
    first_floor_block: Sequence[int | None],
    budgets: Sequence[int],
) -> dict[str, np.ndarray]:
    """The three-class decomposition as a function of budget (fractions)."""
    n = len(list(first_evasion))
    rows = [decompose_at_budget(first_evasion, first_floor_block, B) for B in budgets]
    return {
        k: np.array([r[k] / n for r in rows]) if n else np.zeros(len(rows))
        for k in ("success", "feasibility_bound", "detector_bound")
    }


# --------------------------------------------------------------------------- #
# 3. Floor binding -- how much does the feasibility constraint actually bite?
# --------------------------------------------------------------------------- #
def floor_binding(
    flip_attempts: Sequence[int], floor_blocked: Sequence[int]
) -> dict[str, float]:
    """Across all samples: of the candidate moves that flipped the label BEFORE
    projection, what fraction did the projection revert (the floor biting)?

    This surfaces the feasibility constraint's effect even when the final success
    rate is 100%: the floor may not PREVENT evasion yet still raise its cost by
    reverting a large share of the attack's flips.
    """
    fa = int(np.sum(flip_attempts))
    fb = int(np.sum(floor_blocked))
    return {
        "flip_attempts": fa,
        "floor_blocked": fb,
        "block_rate": (fb / fa) if fa else 0.0,
    }


def aggregate_blocking(blocking_dicts: Sequence[Mapping[str, int]]) -> dict[str, int]:
    """Sum per-feature floor-reversion counts across samples."""
    total: Counter = Counter()
    for d in blocking_dicts:
        total.update(d)
    return dict(total.most_common())


# --------------------------------------------------------------------------- #
# 4. Per-feature movement of successful evasions vs detector reliance
# --------------------------------------------------------------------------- #
def feature_movement(
    deltas: Sequence[Mapping[str, float]], top_k: int | None = None
) -> list[tuple[str, float]]:
    """Mean absolute (scaled) perturbation per feature over successful evasions,
    ranked descending. `deltas[i]` maps feature -> signed scaled delta for the
    i-th successful sample.
    """
    acc: dict[str, list[float]] = {}
    for d in deltas:
        for f, v in d.items():
            acc.setdefault(f, []).append(abs(float(v)))
    ranked = sorted(((f, float(np.mean(v))) for f, v in acc.items()),
                    key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k] if top_k else ranked


def topk_overlap(a: Sequence[str], b: Sequence[str], k: int) -> dict:
    """Overlap between the top-k of two ranked feature lists: the shared set and
    its Jaccard. Used to ask whether the features the attack moves most are the
    features the detector weights most."""
    sa, sb = set(list(a)[:k]), set(list(b)[:k])
    inter = sa & sb
    union = sa | sb
    return {
        "k": k,
        "shared": sorted(inter),
        "n_shared": len(inter),
        "jaccard": (len(inter) / len(union)) if union else 0.0,
    }
