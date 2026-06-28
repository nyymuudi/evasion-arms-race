# Temporal split: closing a documented limitation (Phase 0)

The project's evaluation used a stratified random split, with a temporal split
listed as deferred future work (it needs the GeneratedLabelledFlows distribution,
which retains timestamps the MachineLearningCSV distribution strips). This document
closes that debt: the temporal split is now implemented, the dataset's traps are
documented, and the headline results are shown to hold under it.

## The traps (all silent if unhandled)

Diagnostics on `TrafficLabelling/Wednesday-workingHours.pcap_ISCX.csv` (85 columns:
78 features + Label + 6 metadata) surfaced three:

1. **Encoding.** CIC distributions are inconsistently encoded. The loader now tries
   UTF-8, then falls back to latin-1. (The Wednesday file is UTF-8; others are not.)
2. **Collection-artefact metadata.** Five columns — Flow ID, Source IP, Source Port,
   Destination IP, Protocol — are dropped: a model must not learn "this IP = attack."
   Destination Port is kept (a genuine feature; the ablation diagnostics had already
   shown it is not an artefact). Timestamp is used for the split, then dropped, so the
   stratified and temporal runs see the *same* 78 features.
3. **Timestamp format.** The format is day-first `D/M/YYYY H:MM` on a **12-hour clock
   with no AM/PM marker** and no seconds. `pandas.to_datetime` would read the date
   month-first (5/7 → May 7, not 5 July) and leave afternoon times in the morning
   (3:22 → 03:22, not 15:22), silently mis-ordering the data. The parser is explicit:
   day-first, and — since the captures run in working hours — hours 1–7 are PM (+12).
   It validates the result lands in a daytime window and raises otherwise, rather
   than guessing.

## The decisive finding: DoS Hulk is a 24-minute burst

Parsed, the attack occupies **10:43–11:07 on 5 July 2017** — all 231,073 Hulk flows
in a single 24-minute window. A conventional early-train / late-test split is
therefore **degenerate**: cutting at the 75th row-percentile by time puts 100% of the
attack in train and leaves the test set with **zero positives** (PR-AUC undefined).
This is intrinsic to CIC-IDS2017's scheduled-attack design, and it is the real reason
"temporal split" is rarely done honestly on this data.

The cut is therefore placed at the **median timestamp of the target class** (≈10:55),
which keeps attack flows on both sides. Caveat: because the burst is homogeneous over
24 minutes, the attack halves are near-identical, so this split mainly stresses
**benign** concept drift (morning-train vs afternoon-test), not attack drift.

## Result: the headline holds under temporal evaluation

Baseline detectors, GeneratedLabelledFlows Wednesday, DoS Hulk vs BENIGN:

| split | train / test | LR PR-AUC | RF PR-AUC |
|---|---|---|---|
| stratified | 502k / 167k | 0.9995 | 1.0000 |
| temporal (target-median cut) | 236k / 434k | 0.9964 | 0.9990 |

Two conclusions:

1. **Dropping the metadata did not dent the stratified baseline** (still 0.9995 /
   1.0000) — direct evidence, on a distribution that *contains* the IP/Flow-ID
   columns, that the detector was not riding collection artefacts. This corroborates
   the earlier ablation on a harder distribution.
2. **Concept drift is small:** LR −0.0031, RF −0.0010 PR-AUC — both far below the 0.05
   threshold pre-registered as "dramatic". So the Layer A–C results are not artefacts
   of the stratified split, and a full re-run of the pipeline under temporal evaluation
   is not warranted. The debt is closed with a confirmatory (mildly negative) result.

*Reproduce:* `build_dataset(path, split="temporal")` vs `split="stratified"`, then
`train_baseline`. Tests in `tests/test_temporal_split.py`.
