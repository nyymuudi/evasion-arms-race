"""Black-box evasion search (Layer A, todo item 5).

A query-based search (boundary attack / zeroth-order) that calls
features.projection.project() EVERY step, so every candidate stays realisable
as DoS-Hulk traffic. Reports: success rate, perturbation size (controllable
features only), query count.
"""
