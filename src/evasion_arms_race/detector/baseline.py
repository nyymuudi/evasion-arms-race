"""Baseline DoS-Hulk detector (roadmap weeks 1-2).

Train a classifier with HONEST evaluation: PR-AUC (not accuracy), temporally
separated train/test. This is the target the Layer A attack evades; without a
credible detector, evasion numbers are meaningless.
"""
