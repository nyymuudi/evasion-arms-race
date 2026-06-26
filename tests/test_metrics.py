"""Unit tests for the pure evasion-metric functions (Layer A, item 6)."""

from __future__ import annotations

import numpy as np

from evasion_arms_race.metrics import evasion as M


def test_perturbation_curve_is_cdf_of_successful_cost():
    # three samples: two succeed at L2 1.0 and 3.0, one fails (L2 ignored).
    l2 = [1.0, 3.0, 0.5]
    success = [True, True, False]
    grid = [0.0, 1.0, 2.0, 3.0, 10.0]
    curve = M.perturbation_curve(l2, success, grid)
    # failed sample never counts even though its L2 (0.5) is small.
    assert list(curve) == [0.0, 1 / 3, 1 / 3, 2 / 3, 2 / 3]
    # monotone non-decreasing, plateaus at overall success rate (2/3).
    assert np.all(np.diff(curve) >= 0)
    assert curve[-1] == 2 / 3


def test_budget_curve_counts_first_evasion():
    first_evasion = [10, 100, None, 50]   # one never evaded
    curve = M.budget_curve(first_evasion, n_samples=4, budgets=[5, 10, 60, 100, 1000])
    assert list(curve) == [0.0, 1 / 4, 2 / 4, 3 / 4, 3 / 4]


def test_decompose_at_budget_three_classes():
    # sample A: evades at 10; B: floor-blocked at 5, never evades; C: nothing.
    first_evasion = [10, None, None]
    first_floor = [None, 5, None]
    # at B=4: nobody has evaded, floor not yet seen -> all detector_bound
    assert M.decompose_at_budget(first_evasion, first_floor, 4) == {
        "success": 0, "feasibility_bound": 0, "detector_bound": 3}
    # at B=8: A not yet evaded (10>8) but no floor for A -> detector_bound;
    #         B floor-blocked at 5 -> feasibility_bound; C -> detector_bound
    assert M.decompose_at_budget(first_evasion, first_floor, 8) == {
        "success": 0, "feasibility_bound": 1, "detector_bound": 2}
    # at B=20: A evaded; B feasibility_bound; C detector_bound
    assert M.decompose_at_budget(first_evasion, first_floor, 20) == {
        "success": 1, "feasibility_bound": 1, "detector_bound": 1}


def test_decomposition_series_fractions_sum_to_one():
    fe = [10, None, 30]
    ff = [None, 5, None]
    budgets = [1, 8, 20, 40]
    ser = M.decomposition_series(fe, ff, budgets)
    total = ser["success"] + ser["feasibility_bound"] + ser["detector_bound"]
    assert np.allclose(total, 1.0)
    # success is non-decreasing in budget
    assert np.all(np.diff(ser["success"]) >= 0)


def test_floor_binding_rate():
    fb = M.floor_binding(flip_attempts=[10, 0, 5], floor_blocked=[3, 0, 5])
    assert fb["flip_attempts"] == 15
    assert fb["floor_blocked"] == 8
    assert fb["block_rate"] == 8 / 15
    # empty / no attempts -> zero, no division error
    assert M.floor_binding([], [])["block_rate"] == 0.0


def test_feature_movement_ranks_by_mean_abs_delta():
    deltas = [
        {"Idle Mean": 2.0, "Fwd IAT Min": -1.0},
        {"Idle Mean": -4.0, "Fwd IAT Min": 1.0},
    ]
    ranked = M.feature_movement(deltas)
    assert ranked[0][0] == "Idle Mean"        # mean|Δ| = 3.0
    assert ranked[1][0] == "Fwd IAT Min"      # mean|Δ| = 1.0
    assert abs(ranked[0][1] - 3.0) < 1e-9
    assert M.feature_movement(deltas, top_k=1) == [ranked[0]]


def test_topk_overlap_jaccard():
    a = ["x", "y", "z", "w"]
    b = ["y", "z", "q", "r"]
    ov = M.topk_overlap(a, b, k=3)        # a3={x,y,z} b3={y,z,q}
    assert ov["shared"] == ["y", "z"]
    assert ov["n_shared"] == 2
    assert ov["jaccard"] == 2 / 4
