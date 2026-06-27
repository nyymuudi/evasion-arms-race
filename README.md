# evasion-arms-race

**Adversarial evasion against ML-based network intrusion detection, framed as a heads-up game between a detector and an attacker.**

evasion-arms-race studies a single, sharp question: *can an attacker modify malicious network traffic so that an ML intrusion detector misclassifies it as benign — while the attack still works and the traffic stays protocol-legal?* It is an arms race played heads-up: detector and attacker sit across the table, each adapting to the other. The detector reads signals (flow features); the attacker controls what signals it emits without giving up its hand (the functional attack). Optimising the *observable* signal beneath an opponent's decision boundary, under hard constraints on what you actually control, is the same structure whether the table is poker or a network.

That adversarial examples can fool ML classifiers is well established [1, 2]. What is far less settled — and what this project targets — is whether such evasions survive the **problem-space** constraints of a real domain, where the adversary manipulates objects (packets), not feature vectors [3, 4].

## Feasibility: the problem-space commitment

Most *feature-space* evasion work perturbs whatever features lower the classifier's score, including features the attacker does not control in reality (server responses, network-side aggregates); the resulting adversarial examples are often not realisable as traffic, so the reported attack-success numbers can be illusory [3, 4]. This project's central commitment is therefore a **feasibility (problem-space) constraint** [3]: every feature is partitioned by *who controls it*, and the search may move only features the attacker can actually set, while preserving the attack's functional core — a DoS flood must remain a flood [4].

Feature control classes:

| Class | Meaning | Perturbable? |
|-------|---------|--------------|
| **Controllable** | Attacker's own forward timing / sizing / padding / flags | Yes |
| **Constrained** | Forward volume / rate / TCP params | Within a legal box, with a DoS functional floor |
| **Frozen** | Backward (server) direction, connection-level flag counts, fixed service port | No — reset to original |
| **Derived** | Aggregates mixing both directions | Never set directly; recomputed from atomic features |

## Threat model

Black-box: the attacker has query access to the detector's decision only, not its gradients or parameters. This is the realistic setting (the attacker rarely owns the model) and the harder one.

## Method and positioning

The evasion search is a *decision-based* black-box attack: it queries only the detector's hard label, in the spirit of the Boundary Attack [8] and HopSkipJump [9], rather than estimating gradients from confidence scores as score-based methods such as NES do [10]. Running an identical hard-label attack against a smooth logistic regression and a piecewise-constant Random Forest [11] isolates the effect of model geometry on evasion cost; the tree-ensemble case connects to known results on evading and hardening tree ensembles [12]. The distinguishing element relative to feature-space adversarial ML is the **problem-space** discipline [3, 4]: candidate perturbations are projected onto what an attacker can physically realise and then validated back at the packet level, where a Bhatia–Davis variance bound [13] rejects feature vectors that no packet multiset can produce.

## Dataset

Primary: **LYCOS-IDS2017** [7] (the LycoSTand flow extractor over the CIC-IDS2017 [5] PCAPs), which corrects documented flow-construction and labelling errors in the original CIC-IDS2017 CSVs. Target attack class: **DoS Hulk** (HTTP flood) — chosen because timing/volume features dominate and are genuinely attacker-controlled, making the functional-preservation constraint tractable. CICFlowMeter [6] naming is supported as a fallback and is what the feature partition was first validated against.

The feature partition is **rule-based and header-driven**: it classifies each column by name pattern, so swapping CICFlowMeter for LYCOS is a header swap, not a code change. Anything the rules cannot confidently classify is surfaced for human confirmation, never silently guessed.

## Project layout

```
evasion-arms-race/
├── src/evasion_arms_race/
│   ├── data/            # dataset loading, header normalisation, stratified split
│   │   └── loader.py
│   ├── features/        # the heart of Layer A — feature control model
│   │   ├── partition.py        # control classes + bounds + derived recompute rules
│   │   ├── rule_classifier.py  # header-driven, feature-set-agnostic classifier
│   │   └── projection.py       # feasibility projection (frozen/constrained/derived)
│   ├── detector/        # baseline IDS the attack evades (PR-AUC, temporal eval)
│   │   └── baseline.py
│   ├── attack/          # black-box evasion search + realisability-constrained poisoning
│   │   ├── blackbox.py
│   │   └── poisoning.py
│   ├── metrics/         # evasion success rate, perturbation size, query budget
│   │   └── evasion.py
│   └── validation/      # pcap-level realisability check (resolves aggregates)
│       └── realisability.py
├── configs/             # experiment configs (dataset paths, target class, budgets)
├── tests/               # pytest suite; projection pipeline is fully validated
├── scripts/             # CLI entrypoints (download, preprocess, train, attack)
├── notebooks/           # exploratory analysis (not part of the reproducible path)
├── experiments/         # run outputs (gitignored)
├── data/{raw,processed,artifacts}/   # gitignored; never commit captures
├── docs/
├── pyproject.toml
└── README.md
```

## Roadmap

The project is built in three layers of increasing ambition.

**Layer A — Evasion** *(complete)*. Attacker evades a fixed detector under feasibility + functional constraints.
- [x] (1) Threat model fixed: black-box
- [x] (2) Feature partition into control classes — generalised to a header-driven rule classifier
- [x] (3) Feasibility projection — validated, idempotent, source-agnostic
- [x] (4) Functional-core preservation — built into the projection (DoS floor)
- [x] (5) Black-box search algorithm calling the projection each step — projected, decision-based boundary attack
- [x] (6) Metrics module: success-vs-perturbation tradeoff curve, three-class outcome decomposition, per-feature movement vs detector reliance
- [x] (7) Packet-level realisability validation — feasibility checks + reconstruct the pooled length features + re-query the detector; scapy witness pcap

**Layer B — Poisoning** *(complete)*. Attacker injects budget-limited, **realisability-projected** Hulk samples — they pass the same `project()` + packet-level feasibility check as the evasion samples and keep the DoS floor — labelled benign into the detector's retraining data [14]; evaluation is on the clean held-out test set the attacker never saw. Black-box, chosen-label injection (the more realistic auto-labelling attacker is strictly weaker: its budget is gated by the item-7 realisable-evasion rate, 52% LR / 0% RF).
- [x] Threat model + retraining pipeline (clean train ∪ projected poison → retrain → evaluate on clean test)
- [x] Two strategies: random label-flip vs boundary-selected poison (nearest the current boundary via black-box query)
- [x] Poison-threshold sweep: PR-AUC and Hulk recall vs poison fraction, both strategies, both detectors

**Finding.** Realistic poisoning does **not** collapse threshold-independent detection: PR-AUC is nearly flat even at a 20% poison budget (LR 0.9993 → 0.9968, RF 1.0000 → 0.9996). The strong Hulk/benign separability the ablation diagnostics found is exactly what makes the detector hard to poison in ranking terms — an honest negative result, reported as such. What *does* degrade is the **operating point**: at the deployed 0.5 threshold, Hulk recall falls to ≈0.70 (LR) / ≈0.75 (RF) at 20% label-flip poison — roughly a quarter to a third of attacks slip through — because the poison shifts predicted probabilities below threshold without destroying the ranking. Strategy nuance: boundary-selected poison is more sample-efficient at low budgets (it dents LR recall already at 5%), while random label-flip, carrying more confidently-wrong mass, does more damage at high budgets. As anticipated from Layer A, the Random Forest is the more poison-robust of the two.

**Layer C — Robust defence + game-theoretic analysis**. Adversarial training, then model the attack/retrain loop explicitly as a game and study whether it converges to an equilibrium or oscillates.

## Setup

```bash
git clone https://github.com/nyymuudi/evasion-arms-race.git
cd evasion-arms-race
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q          # the feasibility pipeline should pass out of the box
```

## Current status

Layer A is complete (items 1–7) and tested. The feasibility pipeline reproduces a hand-built feature partition exactly via the rule classifier, enforces frozen/constrained/derived constraints, preserves the DoS functional core, and is idempotent. The black-box attack (item 5) is a projected, decision-based **boundary attack** with random-restart initialisation: it calls the feasibility projection on every candidate, so each queried point stays realisable as traffic, and runs identically against both baseline detectors. The search space is the controllable + constrained features only; perturbation magnitude is reported over controllable features in the detector's scaled space.

Headline finding (DoS-Hulk samples, 1200-query budget): **both detectors are evaded at 100% success**. The contrast is cost, not success — the Random Forest needs both more queries (it climbs to 100% gradually as budget grows, while logistic regression saturates almost immediately) and, typically, a larger perturbation. The earlier ablation-based expectation that the Random Forest would *resist* did not survive contact with the attack: classification robustness (PR-AUC under feature ablation) is not adversarial robustness under a feasibility-constrained search. Whether the Random Forest evasions correspond to **physically realisable** flows — rather than exploiting the forest's arbitrary extrapolation outside its training support — is settled by the realisability check (item 7), below, and the answer reverses this picture.

The metrics module (item 6, `python -m evasion_arms_race.metrics.report`) turns the per-sample results into four reportable products: (1) a **success-vs-perturbation tradeoff curve** (success rate as a function of the allowed controllable-L2, never a single number); (2) a **three-class outcome decomposition vs query budget** — success / `feasibility_bound` (the DoS floor blocked the perturbation) / `detector_bound` (the detector resists on the movable subspace); (3) **per-feature movement** of successful evasions compared against the detector's top-weighted features; (4) logistic regression vs Random Forest on every figure. Two findings worth flagging: the DoS floor rarely *terminates* an attack at these budgets, yet it reverts a large share of individual label-flips (a per-step block rate the report quantifies) — it raises cost without preventing evasion; and the detector's most-weighted features are mostly **unmovable** (backward/derived), with the Random Forest evasions moving *none* of its top-10 features — the attack exploits movable correlates, not the detector's stated signal. Figures land in `experiments/figures/` (gitignored); a small numeric summary in `experiments/evasion_metrics.json` (version-controlled).

**Realisability (item 7, the cornerstone, `python -m evasion_arms_race.validation.realisability`).** A feature vector the detector calls benign is a real evasion only if it corresponds to a sendable packet stream. Each successful evasion is checked for feasibility — its forward size and inter-arrival *moments* must admit an actual packet multiset (mean/total consistency, `min ≤ mean ≤ max`, and the Bhatia–Davis variance bound [13]) — and the five pooled packet-length statistics, previously deferred as "unreconstructable", are recovered in closed form from the per-direction marginals (Min/Max exactly; Mean/Std/Variance via the law of total variance with a data-fit affine calibration, R² ≥ 0.997), after which the detector is re-queried on the corrected vector. The result reframes Layer A: **feature-space evasion (100% for both models) does not survive problem-space validation.** Of 25 successful evasions each, the realisable fraction is **52% for logistic regression and 0% for the Random Forest** — every Random-Forest evasion is statistically infeasible (the search produced flows whose moments no packet sequence can have); none were reverted by the length-feature correction, so the collapse is due to infeasibility, not the pooled features. The apparent Random-Forest vulnerability of items 5–6 was therefore an artefact of the attack roaming infeasible regions of feature space; under a realisability constraint it vanishes. Feasibility is sound but not exhaustive, so the realisable fractions are upper bounds (0% is definitive); the clear next step is to fold these moment constraints into the projection so the search only ever proposes realisable flows. A scapy [witness pcap](experiments/pcaps/) of each realisable flow is emitted (gitignored); the numeric summary is `experiments/realisability.json` (version-controlled).

## Notes on integrity

- Honest evaluation only: PR-AUC over accuracy. The current split is stratified random; a temporal split (to make concept drift visible) requires the timestamped GeneratedLabelledFlows distribution and is deferred, not silently assumed.
- The feasibility constraint is the project's reason to exist; relaxing it to chase higher evasion rates would make the results meaningless.
- Evasion success is reported in **problem space, not just feature space**: a vector the detector misclassifies counts as an evasion only once it is shown to be realisable as traffic (item 7). This is what turns a 100% feature-space success rate into the honest 52% / 0% realisable figures.
- Known CIC-IDS2017 data-quality issues are handled by preferring the corrected LYCOS-IDS2017 extraction.

## References

[1] C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. Goodfellow, R. Fergus. "Intriguing Properties of Neural Networks." *ICLR*, 2014.
[2] I. Goodfellow, J. Shlens, C. Szegedy. "Explaining and Harnessing Adversarial Examples." *ICLR*, 2015.
[3] F. Pierazzi, F. Pendlebury, J. Cortellazzi, L. Cavallaro. "Intriguing Properties of Adversarial ML Attacks in the Problem Space." *IEEE Symposium on Security and Privacy (S&P)*, 2020.
[4] G. Apruzzese, M. Andreolini, L. Ferretti, M. Marchetti, M. Colajanni. "Modeling Realistic Adversarial Attacks against Network Intrusion Detection Systems." *ACM Digital Threats: Research and Practice (DTRAP)*, 3(3), 2022.
[5] I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani. "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization." *ICISSP*, 2018. (CIC-IDS2017)
[6] A. H. Lashkari, G. Draper-Gil, M. S. I. Mamun, A. A. Ghorbani. "Characterization of Tor Traffic Using Time Based Features." *ICISSP*, 2017. (CICFlowMeter)
[7] A. Rosay et al. "From CIC-IDS2017 to LYCOS-IDS2017: A Corrected Dataset for Better Performance." *IEEE/WIC/ACM International Conference on Web Intelligence (WI-IAT)*, 2021.
[8] W. Brendel, J. Rauber, M. Bethge. "Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models." *ICLR*, 2018.
[9] J. Chen, M. I. Jordan, M. J. Wainwright. "HopSkipJumpAttack: A Query-Efficient Decision-Based Attack." *IEEE Symposium on Security and Privacy (S&P)*, 2020.
[10] A. Ilyas, L. Engstrom, A. Athalye, J. Lin. "Black-box Adversarial Attacks with Limited Queries and Information." *ICML*, 2018.
[11] L. Breiman. "Random Forests." *Machine Learning*, 45(1), 2001.
[12] A. Kantchelian, J. D. Tygar, A. D. Joseph. "Evasion and Hardening of Tree Ensemble Classifiers." *ICML*, 2016.
[13] R. Bhatia, C. Davis. "A Better Bound on the Variance." *American Mathematical Monthly*, 107(4), 2000.
[14] B. Biggio, B. Nelson, P. Laskov. "Poisoning Attacks against Support Vector Machines." *ICML*, 2012.