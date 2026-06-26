"""Black-box evasion search (Layer A, todo item 5).

A query-based, DECISION-BASED boundary attack that calls
features.projection.project() EVERY step, so every candidate the detector ever
sees is realisable as DoS-Hulk traffic.

Design decisions (see commit message / session notes for the full rationale)
---------------------------------------------------------------------------
* Algorithm: a projected Boundary Attack (Brendel et al. 2018), NOT a
  zeroth-order gradient estimator (NES/SPSA).
    - The stated threat model is query access to the DECISION (hard label),
      not to a score. Boundary attack is natively hard-label.
    - It does not estimate the gradient of any continuous score, so it degrades
      gracefully on the piecewise-constant Random Forest. Running the SAME
      hard-label attack against both detectors keeps the "LR breaks, RF resists"
      comparison clean, rather than letting it become an artefact of feeding the
      attacker soft labels on one model and step-function noise on the other.
    - The feasibility projection composes trivially: propose -> project -> query.

* Search space: CONTROLLABLE u CONSTRAINED only (the features the attacker
  actually sets). DERIVED features (including the unreconstructable packet-length
  statistics) are never moved directly; the projection recomputes / passes them
  through. This keeps the search honestly scoped to what the attacker controls.

* Geometry is done in the SCALED space (the detector's space) so step sizes are
  comparable across features; every candidate is inverse-transformed to raw,
  projected (the floors/frozen resets are physical and live in raw space), then
  forward-transformed to query. Perturbation magnitude is reported in scaled
  space over CONTROLLABLE coordinates only.

Initialisation falls back from benign-blend seeding to RANDOM-RESTART feasible
exploration, so a failure verdict means the WHOLE query budget was spent without
finding a feasible benign point -- not merely that the handful of init blends
missed.

Failure attribution (the reporting core)
-----------------------------------------
Three outcomes are distinguished, not two:
    success           : a projected vector the detector calls benign.
    feasibility_bound : label-flipping candidates were found, but project() pulled
                        every one back across the boundary (DoS floor / frozen).
                        The binding constraint is FEASIBILITY, not the detector.
    detector_bound    : within the full budget, not even an UN-projected candidate
                        in the attacker's perturbable subspace flipped the label.
                        (This is distinct from 'detector is strong on movable
                        features': the signal may live in frozen features the
                        attacker never moves. The item-4 controllable-only
                        ablation is the cross-reference that tells the two apart.)
The distinction is made measurable: each rejected candidate is also queried
UN-projected, in a separate diagnostic counter (not part of the attacker's
budget), and the projection's `clamped` features are tallied to attribute which
constraints did the reverting.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from evasion_arms_race.features.partition import Control, normalize
from evasion_arms_race.features.projection import project
from evasion_arms_race.features.rule_classifier import classify

ARTIFACT_DIR = Path("data/artifacts")


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
@dataclass
class Artifacts:
    """Everything the attack needs from disk: the shared scaler, both detectors,
    and the feature ordering the scaler/models were fit on."""

    scaler: object
    logreg: object
    rf: object
    feature_names: list[str]

    def model(self, name: str):
        if name in ("logreg", "logistic_regression"):
            return self.logreg
        if name in ("rf", "random_forest"):
            return self.rf
        raise KeyError(f"unknown model {name!r}")


def load_artifacts(artifact_dir: Path = ARTIFACT_DIR) -> Artifacts:
    d = Path(artifact_dir)
    with open(d / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(d / "logreg.pkl", "rb") as f:
        logreg = pickle.load(f)
    with open(d / "rf.pkl", "rb") as f:
        rf = pickle.load(f)
    feature_names = json.loads((d / "feature_names.json").read_text())
    return Artifacts(scaler, logreg, rf, [normalize(c) for c in feature_names])


# --------------------------------------------------------------------------- #
# Decision oracle (hard label) with separate budget / diagnostic counters
# --------------------------------------------------------------------------- #
class Oracle:
    """Wraps one detector + the shared scaler. Decision-only: returns the hard
    label (1 = attack, 0 = benign). Counts feasible queries (on projected points,
    the attacker's real budget) separately from diagnostic queries (on
    un-projected points, used only for floor attribution)."""

    def __init__(self, model, scaler, feature_names: list[str]):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.feasible_queries = 0
        self.diagnostic_queries = 0

    # --- space conversions -------------------------------------------------- #
    def to_vec(self, d: dict[str, float]) -> np.ndarray:
        return np.array([float(d[f]) for f in self.feature_names], dtype=float)

    def to_dict(self, v: np.ndarray) -> dict[str, float]:
        return {f: float(v[i]) for i, f in enumerate(self.feature_names)}

    def scale(self, raw_vec: np.ndarray) -> np.ndarray:
        return self.scaler.transform(raw_vec.reshape(1, -1))[0]

    def unscale(self, scaled_vec: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(scaled_vec.reshape(1, -1))[0]

    # --- decisions ---------------------------------------------------------- #
    def decision_scaled(self, scaled_vec: np.ndarray, *, diagnostic: bool) -> int:
        if diagnostic:
            self.diagnostic_queries += 1
        else:
            self.feasible_queries += 1
        return int(self.model.predict(scaled_vec.reshape(1, -1))[0])


# --------------------------------------------------------------------------- #
# Config / result
# --------------------------------------------------------------------------- #
@dataclass
class AttackConfig:
    query_budget: int = 1500          # feasible queries (the attacker's cost)
    n_init_refs: int = 8              # benign reference samples to seed from
    alpha_grid: tuple[float, ...] = (1.0, 0.6, 0.3)   # blend toward benign at init
    sigma0: float = 0.05              # initial orthogonal step (rel. to ||adv-clean||)
    sigma_min: float = 1e-3
    explore_sigmas: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)   # random-restart scales
    bsearch_tol: float = 1e-3         # geometric (toward-source) binary-search tol
    attribute_floor: bool = True      # spend diagnostic queries to attribute floor
    seed: int = 0


@dataclass
class AttackResult:
    model_name: str
    success: bool
    failure_mode: str                 # success | feasibility_bound | detector_bound
    init_found: bool
    feasible_queries: int
    diagnostic_queries: int
    # perturbation of the best feasible adversarial, scaled space, CONTROLLABLE only
    l2_controllable: float
    linf_controllable: float
    l2_perturbable: float             # controllable u constrained, for context
    # floor attribution
    flip_attempts: int                # candidates benign BEFORE projection
    flips_survived: int               # ... still benign AFTER projection
    floor_blocked: int                # benign before, reverted to attack after
    blocking_features: dict[str, int] # which clamped/frozen features did the reverting
    # Traces so a single run reconstructs success-vs-budget and the three-class
    # decomposition at any budget B <= query_budget (item 6), without re-running:
    queries_to_first_evasion: int | None = None    # feasible-query count at first benign seed
    queries_to_first_floor_block: int | None = None  # ... at first floor-reverted flip
    best_vector: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Core single-sample attack
# --------------------------------------------------------------------------- #
def attack_sample(
    clean_raw: dict[str, float],
    oracle: Oracle,
    benign_refs_raw: list[dict[str, float]],
    config: AttackConfig = AttackConfig(),
    source=None,
) -> AttackResult:
    """Run a projected boundary attack on a single Hulk sample.

    `clean_raw` is the original Hulk flow (raw feature dict). `benign_refs_raw`
    are real benign flows used to seed an initial adversarial point. Every point
    the oracle scores as the attack's real budget is projected onto the feasible
    set first, via project(perturbed, clean_raw, source).

    `source` is any ControlSource (a Partition or a rule-derived
    ClassificationReport). It MUST be keyed by the SAME feature set as the data;
    when omitted it is derived from the oracle's feature names via classify(),
    so the projection always matches the actual columns (the hard-coded PARTITION
    carries 'Flow IAT Total', which CICFlowMeter does not emit).
    """
    if source is None:
        source = classify(oracle.feature_names)
    rng = np.random.default_rng(config.seed)
    names = oracle.feature_names

    # Perturbable / controllable coordinate masks (indices into the scaled vector).
    ctrl_idx = np.array(
        [i for i, f in enumerate(names)
         if source.control_of(f) is Control.CONTROLLABLE], dtype=int
    )
    pert_idx = np.array(
        [i for i, f in enumerate(names)
         if source.control_of(f) in (Control.CONTROLLABLE, Control.CONSTRAINED)],
        dtype=int,
    )

    clean_vec = oracle.to_vec(clean_raw)
    clean_scaled = oracle.scale(clean_vec)

    # Bookkeeping for floor attribution.
    stats = {"flip_attempts": 0, "flips_survived": 0, "floor_blocked": 0,
             "first_floor_block_q": None}
    blocking: Counter = Counter()
    first_evasion_q: int | None = None

    def feasible_eval(cand_scaled: np.ndarray):
        """Project a scaled candidate, query the detector on the PROJECTED point.
        Optionally also query the un-projected point (diagnostic) to attribute
        whether the projection is what reverted a label flip.

        Returns (decision_projected, proj_scaled, projres).
        """
        cand_raw = oracle.unscale(cand_scaled)
        projres = project(oracle.to_dict(cand_raw), clean_raw, source)
        proj_scaled = oracle.scale(oracle.to_vec(projres.vector))
        dec_proj = oracle.decision_scaled(proj_scaled, diagnostic=False)

        if config.attribute_floor:
            dec_unproj = oracle.decision_scaled(cand_scaled, diagnostic=True)
            if dec_unproj == 0:                       # un-projected candidate evades
                stats["flip_attempts"] += 1
                if dec_proj == 0:
                    stats["flips_survived"] += 1
                else:                                 # projection reverted the flip
                    stats["floor_blocked"] += 1
                    if stats["first_floor_block_q"] is None:
                        stats["first_floor_block_q"] = oracle.feasible_queries
                    for f in projres.clamped:
                        blocking[f] += 1
        return dec_proj, proj_scaled, projres

    def out_of_budget() -> bool:
        return oracle.feasible_queries >= config.query_budget

    # ---- 0. Sanity: the clean sample must start in the attack class --------- #
    # A boundary attack is meaningless if the detector already calls the source
    # benign. Spend one feasible query to confirm; report it honestly if not.
    dec_clean, clean_proj_scaled, _ = feasible_eval(clean_scaled)
    if dec_clean == 0:
        return _result(oracle, "", clean_scaled, clean_proj_scaled, ctrl_idx,
                       pert_idx, success=True, mode="already_benign",
                       init_found=True, stats=stats, blocking=blocking, names=names,
                       first_evasion=oracle.feasible_queries)

    # ---- 1. Initialisation: blend toward benign references until one evades -- #
    adv_scaled = None
    refs = benign_refs_raw[: config.n_init_refs]
    for ref in refs:
        if out_of_budget():
            break
        ref_scaled = oracle.scale(oracle.to_vec(ref))
        for alpha in config.alpha_grid:              # full blend first (likeliest benign)
            if out_of_budget():
                break
            cand = clean_scaled.copy()
            cand[pert_idx] = (
                (1.0 - alpha) * clean_scaled[pert_idx] + alpha * ref_scaled[pert_idx]
            )
            dec, proj_scaled, _ = feasible_eval(cand)
            if dec == 0:
                adv_scaled = proj_scaled
                first_evasion_q = oracle.feasible_queries
                break
        if adv_scaled is not None:
            break

    # ---- 1b. Random-restart fallback: if blending toward benign references did
    #          not flip the label, spend the remaining budget on random feasible
    #          perturbations. This turns a 'detector_bound' verdict into a
    #          budget-EXHAUSTED claim rather than 'the 24 init blends missed'. --- #
    if adv_scaled is None:
        while not out_of_budget():
            sigma_e = float(rng.choice(config.explore_sigmas))
            cand = clean_scaled.copy()
            cand[pert_idx] = (
                clean_scaled[pert_idx]
                + rng.standard_normal(pert_idx.size) * sigma_e
            )
            dec, proj_scaled, _ = feasible_eval(cand)
            if dec == 0:
                adv_scaled = proj_scaled
                first_evasion_q = oracle.feasible_queries
                break

    if adv_scaled is None:
        # No feasible adversarial seed. The failure mode hinges on WHY:
        # if un-projected blends evaded but projection reverted them -> floor.
        mode = _classify_failure(stats)
        return _result(oracle, "", clean_scaled, clean_scaled, ctrl_idx, pert_idx,
                       success=False, mode=mode, init_found=False,
                       stats=stats, blocking=blocking, names=names)

    # ---- 2. Boundary refinement: shrink perturbation, walk the boundary ----- #
    best_scaled = adv_scaled
    sigma = config.sigma0

    while not out_of_budget():
        progressed = False

        # (a) Geometric step toward the source (clean): binary-search the
        #     smallest blend coefficient lambda that keeps the point benign.
        #     The interpolation ANCHOR is fixed for the duration of the search;
        #     mutating it mid-loop would let the segment collapse onto clean.
        anchor = best_scaled.copy()
        hi_point = anchor                # benign endpoint (lambda = 1)
        lo, hi = 0.0, 1.0   # lambda=0 -> clean (attack); lambda=1 -> anchor (benign)
        while hi - lo > config.bsearch_tol and not out_of_budget():
            mid = 0.5 * (lo + hi)
            cand = clean_scaled.copy()
            cand[pert_idx] = (
                clean_scaled[pert_idx]
                + mid * (anchor[pert_idx] - clean_scaled[pert_idx])
            )
            dec, proj_scaled, _ = feasible_eval(cand)
            if dec == 0:
                hi = mid
                hi_point = proj_scaled
                progressed = True
            else:
                lo = mid
        best_scaled = hi_point

        # (b) Orthogonal exploration: random step on the boundary, projected.
        delta = best_scaled[pert_idx] - clean_scaled[pert_idx]
        norm = np.linalg.norm(delta)
        if norm < 1e-12:
            break  # essentially at the source already; cannot do better
        eta = rng.standard_normal(pert_idx.size)
        eta -= (eta @ delta) / (norm ** 2) * delta     # orthogonalise vs (adv-clean)
        en = np.linalg.norm(eta)
        if en > 1e-12:
            eta = eta / en * sigma * norm
            cand = best_scaled.copy()
            cand[pert_idx] = best_scaled[pert_idx] + eta
            dec, proj_scaled, _ = feasible_eval(cand)
            if dec == 0:
                best_scaled = proj_scaled
                progressed = True

        # Adapt the orthogonal step size; stop when it collapses without progress.
        if not progressed:
            sigma *= 0.5
            if sigma < config.sigma_min:
                break

    return _result(oracle, "", clean_scaled, best_scaled, ctrl_idx, pert_idx,
                   success=True, mode="success", init_found=True,
                   stats=stats, blocking=blocking, names=names,
                   first_evasion=first_evasion_q)


def _classify_failure(stats: dict[str, int]) -> str:
    """feasibility_bound if un-projected flips existed but none survived
    projection; detector_bound if not even an un-projected flip was found."""
    if stats["flip_attempts"] > 0 and stats["flips_survived"] == 0:
        return "feasibility_bound"
    return "detector_bound"


def _result(oracle, model_name, clean_scaled, best_scaled, ctrl_idx, pert_idx,
            *, success, mode, init_found, stats, blocking, names,
            first_evasion=None) -> AttackResult:
    dctrl = best_scaled[ctrl_idx] - clean_scaled[ctrl_idx]
    dpert = best_scaled[pert_idx] - clean_scaled[pert_idx]
    best_vec = {names[i]: float(best_scaled_i)
                for i, best_scaled_i in enumerate(oracle.unscale(best_scaled))}
    return AttackResult(
        model_name=model_name,
        success=success,
        failure_mode=mode,
        init_found=init_found,
        feasible_queries=oracle.feasible_queries,
        diagnostic_queries=oracle.diagnostic_queries,
        l2_controllable=float(np.linalg.norm(dctrl)),
        linf_controllable=float(np.max(np.abs(dctrl))) if ctrl_idx.size else 0.0,
        l2_perturbable=float(np.linalg.norm(dpert)),
        flip_attempts=stats["flip_attempts"],
        flips_survived=stats["flips_survived"],
        floor_blocked=stats["floor_blocked"],
        blocking_features=dict(blocking.most_common()),
        queries_to_first_evasion=first_evasion,
        queries_to_first_floor_block=stats["first_floor_block_q"],
        best_vector=best_vec if success else {},
    )


# --------------------------------------------------------------------------- #
# Batch runner over several Hulk samples vs both detectors
# --------------------------------------------------------------------------- #
def run_attack(
    artifacts: Artifacts,
    hulk_samples: list[dict[str, float]],
    benign_refs: list[dict[str, float]],
    model_names: tuple[str, ...] = ("logreg", "rf"),
    config: AttackConfig = AttackConfig(),
) -> dict[str, list[AttackResult]]:
    """Attack each Hulk sample against each named detector. Returns
    {model_name: [AttackResult per sample]}."""
    # Derive the control source once from the actual feature set; reused for
    # every sample/model so projection always matches the real columns.
    source = classify(artifacts.feature_names)
    if source.unknown:
        raise ValueError(f"unclassified features block projection: {source.unknown}")

    out: dict[str, list[AttackResult]] = {m: [] for m in model_names}
    for m in model_names:
        for i, clean in enumerate(hulk_samples):
            oracle = Oracle(artifacts.model(m), artifacts.scaler,
                            artifacts.feature_names)
            cfg = AttackConfig(**{**config.__dict__, "seed": config.seed + i})
            res = attack_sample(clean, oracle, benign_refs, cfg, source=source)
            res.model_name = m
            out[m].append(res)
    return out


def summarise(results: dict[str, list[AttackResult]]) -> dict:
    """Compact per-model summary: success rate, median feasible queries, median
    controllable-L2 on successes, and the failure-mode breakdown."""
    summary: dict = {}
    for m, rs in results.items():
        n = len(rs)
        succ = [r for r in rs if r.success]
        modes = Counter(r.failure_mode for r in rs)
        summary[m] = {
            "n": n,
            "success_rate": len(succ) / n if n else 0.0,
            "median_feasible_queries": float(np.median([r.feasible_queries for r in rs])) if n else 0.0,
            "median_l2_controllable_success": (
                float(np.median([r.l2_controllable for r in succ])) if succ else None
            ),
            "failure_modes": dict(modes),
            "total_floor_blocked": sum(r.floor_blocked for r in rs),
        }
    return summary


# --------------------------------------------------------------------------- #
# CLI: load artifacts + a few real Hulk samples, attack both detectors
# --------------------------------------------------------------------------- #
def main() -> int:
    import argparse

    from evasion_arms_race.data.loader import build_dataset

    ap = argparse.ArgumentParser(description="Black-box evasion against the baseline detectors.")
    ap.add_argument("--data", default="data/raw/cicids2017/MachineLearningCVE/"
                                       "Wednesday-workingHours.pcap_ISCX.csv")
    ap.add_argument("--artifacts", default=str(ARTIFACT_DIR))
    ap.add_argument("--n-samples", type=int, default=15)
    ap.add_argument("--n-refs", type=int, default=8)
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ARTIFACT_DIR / "attack_report.json"),
                    help="where to persist per-sample results + summary (item 6/7 input)")
    args = ap.parse_args()

    artifacts = load_artifacts(Path(args.artifacts))
    ds = build_dataset(args.data, target_label="DoS Hulk", seed=args.seed)

    # Sanity: artifact ordering must match the freshly loaded feature order.
    if artifacts.feature_names != [normalize(c) for c in ds.feature_names]:
        raise SystemExit("feature order mismatch between artifacts and loaded data")

    rng = np.random.default_rng(args.seed)
    Xte, yte = ds.X_test, ds.y_test
    hulk_idx = np.where(yte == 1)[0]
    benign_idx = np.where(yte == 0)[0]
    hulk_pick = rng.choice(hulk_idx, size=min(args.n_samples, hulk_idx.size), replace=False)
    benign_pick = rng.choice(benign_idx, size=min(args.n_refs, benign_idx.size), replace=False)

    hulk_samples = [Xte.iloc[i].to_dict() for i in hulk_pick]
    benign_refs = [Xte.iloc[i].to_dict() for i in benign_pick]

    cfg = AttackConfig(query_budget=args.budget, n_init_refs=args.n_refs, seed=args.seed)
    results = run_attack(artifacts, hulk_samples, benign_refs, config=cfg)
    summary = summarise(results)

    print("=== Black-box evasion (projected boundary attack) ===")
    print(f"samples: {len(hulk_samples)}  budget(feasible queries): {args.budget}\n")
    for m, s in summary.items():
        print(f"[{m}]")
        print(f"  success rate           : {s['success_rate']:.2%}")
        print(f"  median feasible queries: {s['median_feasible_queries']:.0f}")
        print(f"  median L2 (controllable, success): {s['median_l2_controllable_success']}")
        print(f"  failure modes          : {s['failure_modes']}")
        print(f"  total floor-blocked flips: {s['total_floor_blocked']}")
        # Attribute the floor: which constraints reverted the most flips?
        blk: Counter = Counter()
        for r in results[m]:
            blk.update(r.blocking_features)
        if blk:
            print(f"  top reverting constraints: {dict(blk.most_common(5))}")
        print()

    # Persist for the metrics (item 6) and realisability (item 7) sessions.
    from dataclasses import asdict
    payload = {
        "config": {"n_samples": len(hulk_samples), "n_refs": args.n_refs,
                   "budget": args.budget, "seed": args.seed,
                   "data": args.data, "artifacts": args.artifacts},
        "summary": summary,
        "results": {m: [asdict(r) for r in rs] for m, rs in results.items()},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
