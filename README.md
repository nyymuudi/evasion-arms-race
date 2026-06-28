# evasion-arms-race

**Adversarial evasion against ML-based network intrusion detection, framed as a heads-up game between a detector and an attacker.**

evasion-arms-race studies a single, sharp question: *can an attacker modify malicious network traffic so that an ML intrusion detector misclassifies it as benign — while the attack still works and the traffic stays protocol-legal?* It is an arms race played heads-up: detector and attacker sit across the table, each adapting to the other. The detector reads signals (flow features); the attacker controls what signals it emits without giving up its hand (the functional attack). Optimising the *observable* signal beneath an opponent's decision boundary, under hard constraints on what you actually control, is the same structure whether the table is poker or a network.

That adversarial examples can fool ML classifiers is well established [1, 2]. What is far less settled — and what this project targets — is whether such evasions survive the **problem-space** constraints of a real domain, where the adversary manipulates objects (packets), not feature vectors [3, 4].

## Feasibility: the problem-space commitment

Most *feature-space* evasion work perturbs whatever features lower the classifier's score, including features the attacker does not control in reality (server responses, network-side aggregates); the resulting adversarial examples are often not realisable as traffic, so the reported attack-success numbers can be illusory [3, 4]. This project's central commitment is therefore a **feasibility (problem-space) constraint** [3]: every feature is partitioned by *who controls it*, and the search may move only features the attacker can actually set, while preserving the attack's functional core — a DoS flood must remain a flood [4].

| Class | Meaning | Perturbable? |
|-------|---------|--------------|
| **Controllable** | Attacker's own forward timing / sizing / padding / flags | Yes |
| **Constrained** | Forward volume / rate / TCP params | Within a legal box, with a DoS functional floor |
| **Frozen** | Backward (server) direction, connection-level flag counts, fixed service port | No — reset to original |
| **Derived** | Aggregates mixing both directions | Never set directly; recomputed from atomic features |

## Threat model

Black-box throughout: the attacker has query access to the detector's decision only, not its gradients or parameters. This is the realistic setting (the attacker rarely owns the model) and the harder one.

## Method and positioning

The evasion search is a *decision-based* black-box attack: it queries only the detector's hard label, in the spirit of the Boundary Attack [8] and HopSkipJump [9], rather than estimating gradients from confidence scores as score-based methods such as NES do [10]. Running an identical hard-label attack against a smooth logistic regression and a piecewise-constant Random Forest [11] isolates the effect of model geometry on evasion cost; the tree-ensemble case connects to known results on evading and hardening tree ensembles [12]. The distinguishing element relative to feature-space adversarial ML is the **problem-space** discipline [3, 4]: candidate perturbations are projected onto what an attacker can physically realise and then validated back at the packet level, where a Bhatia–Davis variance bound [13] rejects feature vectors that no packet multiset can produce.

## Dataset

The implementation runs on the **CICFlowMeter CIC-IDS2017** [5, 6] CSVs (Wednesday capture; DoS Hulk vs benign). **LYCOS-IDS2017** [7] — the LycoSTand extractor over the same PCAPs, which corrects documented flow-construction and labelling errors — is the *preferred* extraction and the design target; because the feature partition is rule-based and header-driven, switching to it is a header swap rather than a code change, but it is not yet the default. **DoS Hulk** (HTTP flood) is the target class because timing/volume features dominate and are genuinely attacker-controlled, which keeps the functional-preservation constraint tractable. Features the classifier rules cannot confidently place are surfaced for human confirmation, never silently guessed.

## Project architecture

```
evasion-arms-race/
├── src/evasion_arms_race/
│   ├── data/            # dataset loading, header normalisation, stratified split
│   │   └── loader.py
│   ├── features/        # feature control model — the heart of the feasibility constraint
│   │   ├── partition.py        # control classes + bounds + derived recompute rules
│   │   ├── rule_classifier.py  # header-driven, feature-set-agnostic classifier
│   │   └── projection.py       # feasibility projection (frozen/constrained/derived)
│   ├── detector/        # baseline IDS, ablation diagnostics, adversarial training
│   │   ├── baseline.py · ablation.py · robust.py
│   ├── attack/          # black-box evasion search + realisability-constrained poisoning
│   │   ├── blackbox.py · poisoning.py
│   ├── metrics/         # evasion success / perturbation / query-budget reporting
│   │   └── evasion.py · report.py
│   └── validation/      # packet-level realisability check + manifold_project (search confinement)
│       └── realisability.py
├── experiments/         # run harnesses (arms_race.py, manifold_attack.py) + outputs; figures gitignored, JSON tracked
├── docs/                # game_theory.md, synthesis.md, manifold_experiment.md (the headline stress test)
├── tests/               # pytest suite covering the pure logic of every layer
├── data/{raw,artifacts}/   # gitignored; never commit captures
└── pyproject.toml
```

The work is organised as three layers of increasing ambition, unified by the feasibility constraint:

- **Layer A — Evasion** *(complete)*. A black-box, decision-based boundary attack evades a fixed detector while every candidate is confined to the feasible set (`features/projection.py`) and validated at the packet level (`validation/realisability.py`). Spans the threat model, the control-class partition + rule classifier, the feasibility projection with its DoS functional floor, the boundary search, the metrics module, and the packet-level realisability validation.
- **Layer B — Poisoning** *(complete)*. The attacker injects budget-limited, feasibility-projected Hulk samples (labelled benign) into the detector's retraining set; evaluation is on clean held-out data (`attack/poisoning.py`). Two strategies — random label-flip and boundary-selected — swept over the poison fraction for both detectors.
- **Layer C — Robust defence + game-theoretic framing** *(complete)*. Adversarial training (`detector/robust.py`) folds the attacker's feasibility-projected evasions back into training, labelled with their correct class; an attack/retrain loop (`experiments/arms_race.py`) is analysed **empirically**. The honest equilibrium analysis lives in `docs/game_theory.md`, the cross-layer synthesis in `docs/synthesis.md`.

## Setup

```bash
git clone https://github.com/nyymuudi/evasion-arms-race.git
cd evasion-arms-race
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ".[dev,pcap]" for the scapy witness pcaps
pytest -q                        # the feasibility pipeline should pass out of the box
```

## Notes on integrity

- Honest evaluation only: PR-AUC over accuracy. The temporal split is now done (`docs/temporal_split.md`): on the timestamped GeneratedLabelledFlows distribution, with the IP/Flow-ID metadata dropped and the 12h/day-first timestamps parsed correctly, concept drift is small (LR −0.0031, RF −0.0010 PR-AUC vs stratified) — so the results are not a split artefact. DoS Hulk being a 24-minute burst makes a *global* early/late split degenerate (zero positives in test); the cut is at the attack's median timestamp.
- The feasibility constraint is the project's reason to exist; relaxing it to chase higher evasion rates would make the results meaningless.
- Evasion success is reported in **problem space, not just feature space**: a vector counts as an evasion only once it is realisable as traffic. And realisability must shape the **search**, not be applied as a post-filter — a free-space search + filter reported 52% / 0% realisable evasion, but a manifold-constrained search finds 85% / 100%. The project corrected its own headline when this pre-registered test failed (`docs/manifold_experiment.md`); a modest exact result beats an oversold one.
- Terminology is kept conservative: empirical dynamics are not called equilibria, and a near-perfect PR-AUC is treated as a prompt to investigate (ablation, realisability), not a trophy.

## Findings

**Layer A — evasion, and the right way to measure realisability.** The boundary attack evades both detectors at **100% in feature space**. Applying a realisability *post-filter* to that free-space search leaves only **52% (LR) / 0% (RF)** passing the packet-level feasibility check — which first read as "the Random Forest's vulnerability vanishes under realisability." **A pre-registered stress test overturned that reading.** When the *same* attack is constrained to search **on** the realisable manifold (`manifold_project` built into the search, not a filter), realisable evasion jumps to **85% (LR) and 100% (RF)**, unchanged by extra query budget (it plateaus within ≈50 queries; see `docs/manifold_experiment.md`). So the 0% was a **search artefact**: post-hoc filtering of a free-space search under-reports realisable evasion and here inverted the conclusion. The corrected statement: **both detectors are evadable by realisable traffic** (RF ≈100%); realisability must shape the *search*, not be bolted on as a filter — a lesson that indicts a common pattern in adversarial-NIDS evaluation. Enabling detail: the five packet-length aggregates once deferred as "unreconstructable" are recovered in closed form (Min/Max exactly; Mean/Std/Variance via the law of total variance + a data-fit calibration, R² ≥ 0.997), which is what makes an on-manifold search possible; the DoS floor still *raises* evasion cost without terminating it. *Reproduce:* `python experiments/manifold_attack.py`, `python -m evasion_arms_race.validation.realisability`.

**Layer B — poisoning is bounded by separability.** Poison injected into the detector's retraining data [14] barely dents threshold-independent detection: clean-test PR-AUC stays nearly flat to a 20% poison budget (LR 0.9993 → 0.9968, RF 1.0000 → 0.9996), because the Hulk/benign separability the ablation diagnostics found resists poisoning of the *ranking*. What degrades is the **operating point** — at the deployed 0.5 threshold, Hulk recall falls to ≈0.70 (LR) / ≈0.75 (RF) at 20% label-flip poison, as the poison pushes predicted probabilities under threshold without destroying the ranking. Boundary-selected poison is more sample-efficient at low budgets; mass label-flip more damaging at high; the Random Forest is the more poison-robust. (A corollary stated earlier — that the realistic auto-labelling attacker is gated by a low realisable-evasion rate — is itself revised by the Layer A correction below: on the manifold that rate is ≈85% / ≈100%, so the auto-labelling channel is *open*, not closed.) An honest negative-leaning result, reported as such. *Reproduce:* `python -m evasion_arms_race.attack.poisoning`.

**Layer C — the arms race converges, on the realisability axis.** Adversarial training folds the attacker's feasibility-projected evasions back into the training set (labelled with their *correct* class) and the loop re-attacks *fresh* Hulk samples each round. Read through the **realisable**-evasion rate — the metric that matters — the logistic-regression arms race **converges within ≈2 rounds**: the realisable evasion rate falls **43% → 3% → 0% → 0%**, at a negligible clean cost (PR-AUC 0.9993 → 0.9987; Hulk recall unchanged at 0.998). The attacker's *feature-space* success stays pinned at 100% throughout — but those evasions are now entirely infeasible, so the permanent "stalemate" one would read off the feature-space curve is an illusion; the realisability lens shows the defender has closed the only contest that matters. The Random Forest is already at this fixed point from round 0 (0% realisable throughout — nothing to harden against); its feature-space success is low (≈30–40% at this budget) and does not systematically move. Neither trajectory oscillates or diverges; both reach an empirical fixed point — which `docs/game_theory.md` is careful to call exactly that, **not** a Nash equilibrium: the existence preconditions (finite / compact / convex–concave / zero-sum) do not hold, and the relation to the project's CFR poker-solver heritage is conceptual, not formal. **Manifold follow-up (resolves the caveat).** Re-running the loop with the *manifold-constrained* attack (`--manifold`) — so the defender trains on genuinely realisable evasions and the rate is the true manifold rate — shows adversarial training **does not close the realisable gap**: LR realisable evasion stays at **100% all four rounds** (a linear boundary cannot carve out the benign half of the manifold; the free-search loop's "43% → 0%" was a post-filter artefact), while RF drops 97% → ~30% after one round but then **plateaus at ~30–40% and drifts back up**, never reaching zero — at negligible clean cost. So adversarial training against a feasibility-constrained attacker buys *limited, detector-dependent, incomplete* robustness (nothing for LR, partial for RF), visible only because the attack searches the manifold. `docs/synthesis.md` draws the three layers together; full account in `docs/manifold_experiment.md`. *Reproduce:* `python experiments/arms_race.py --manifold`.

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
