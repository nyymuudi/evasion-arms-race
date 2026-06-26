#!/usr/bin/env python3
"""Smoke demo: classify a header and project an adversarial vector.
Run: python scripts/run_projection_demo.py
"""
from evasion_arms_race.features.partition import PARTITION
from evasion_arms_race.features.rule_classifier import classify

if __name__ == "__main__":
    header = sorted(PARTITION.all_features())
    report = classify(header)
    print(report.summary())
    print(f"unknown (need confirmation): {report.unknown or 'none'}")
