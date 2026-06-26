"""Dataset loading + cleaning for LYCOS-IDS2017 (and CICFlowMeter fallback).

Responsibilities:
  - read CSV, normalise header (strip leading spaces, collapse whitespace)
  - drop duplicate columns (e.g. CICFlowMeter 'Fwd Header Length.1')
  - filter to the target class (DoS Hulk) vs benign
  - temporal train/test split (NOT random) so concept drift is visible
"""
