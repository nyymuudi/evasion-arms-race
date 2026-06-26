# evasion-arms-race

**Adversarial evasion against ML-based network intrusion detection, framed as a heads-up game between a detector and an attacker.**

evasion-arms-race studies a single, sharp question: *can an attacker modify malicious network traffic so that an ML intrusion detector misclassifies it as benign — while the attack still works and the traffic stays protocol-legal?* It is an arms race played heads-up: detector and attacker sit across the table, each adapting to the other. The detector reads signals (flow features); the attacker controls what signals it emits without giving up its hand (the functional attack). Optimising the *observable* signal beneath an opponent's decision boundary, under hard constraints on what you actually control, is the same structure whether the table is poker or a network.

## Why this is not "just another adversarial ML demo"

Most evasion work perturbs whatever features lower the classifier's score, including features the attacker does not control in reality (server responses, network-side aggregates). The resulting "adversarial examples" are not realisable as traffic, so the attack-success numbers are fiction. This project's central commitment is a **feasibility constraint**: every feature is partitioned by *who controls it*, and the search may only move features the attacker can actually set, while preserving the attack's functional core (a DoS flood must remain a flood).

Feature control classes:

| Class | Meaning | Perturbable? |
|-------|---------|--------------|
| **Controllable** | Attacker's own forward timing / sizing / padding / flags | Yes |
| **Constrained** | Forward volume / rate / TCP params | Within a legal box, with a DoS functional floor |
| **Frozen** | Backward (server) direction, connection-level flag counts, fixed service port | No — reset to original |
| **Derived** | Aggregates mixing both directions | Never set directly; recomputed from atomic features |

## Threat model

Black-box: the attacker has query access to the detector's decision only, not its gradients or parameters. This is the realistic setting (the attacker rarely owns the model) and the harder one.

## Dataset

Primary: **LYCOS-IDS2017** (LycoSTand flow extractor over the CIC-IDS2017 PCAPs), which corrects documented flow-construction and labelling errors in the original CIC-IDS2017 CSVs. Target attack class: **DoS Hulk** (HTTP flood) — chosen because timing/volume features dominate and are genuinely attacker-controlled, making the functional-preservation constraint tractable. CICFlowMeter naming is supported as a fallback and is what the feature partition was first validated against.

The feature partition is **rule-based and header-driven**: it classifies each column by name pattern, so swapping CICFlowMeter for LYCOS is a header swap, not a code change. Anything the rules cannot confidently classify is surfaced for human confirmation, never silently guessed.

## Project layout

```
evasion-arms-race/
├── src/evasion_arms_race/
│   ├── data/            # dataset loading, header normalisation, temporal split
│   │   └── loader.py
│   ├── features/        # the heart of Layer A — feature control model
│   │   ├── partition.py        # control classes + bounds + derived recompute rules
│   │   ├── rule_classifier.py  # header-driven, feature-set-agnostic classifier
│   │   └── projection.py       # feasibility projection (frozen/constrained/derived)
│   ├── detector/        # baseline IDS the attack evades (PR-AUC, temporal eval)
│   │   └── baseline.py
│   ├── attack/          # black-box evasion search calling project() each step
│   │   └── blackbox.py
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

**Layer A — Evasion** *(in progress)*. Attacker evades a fixed detector under feasibility + functional constraints.
- [x] (1) Threat model fixed: black-box
- [x] (2) Feature partition into control classes — generalised to a header-driven rule classifier
- [x] (3) Feasibility projection — validated, idempotent, source-agnostic
- [x] (4) Functional-core preservation — built into the projection (DoS floor)
- [ ] (5) Black-box search algorithm calling the projection each step
- [ ] (6) Metrics module: success rate, controllable-only perturbation size, query count
- [ ] (7) pcap-level realisability validation (resolves the unreconstructable aggregates)

**Layer B — Poisoning**. Attacker corrupts the detector's retraining data; quantify the poisoning threshold at which detection collapses.

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

Layer A items 1–4 complete and tested. The feasibility pipeline reproduces a hand-built feature partition exactly via the rule classifier, enforces frozen/constrained/derived constraints, preserves the DoS functional core, and is idempotent. Five CICFlowMeter aggregate features (packet-length statistics) are flagged *unreconstructable* from CSV aggregates alone and are deferred to the pcap-level validation in item 7; LYCOS's richer feature set may permit exact recomputation.

## Notes on integrity

- Honest evaluation only: PR-AUC over accuracy, temporal train/test split so concept drift is visible.
- The feasibility constraint is the project's reason to exist; relaxing it to chase higher evasion rates would make the results meaningless.
- Known CIC-IDS2017 data-quality issues are handled by preferring the corrected LYCOS-IDS2017 extraction.