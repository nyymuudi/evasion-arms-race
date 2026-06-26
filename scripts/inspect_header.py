#!/usr/bin/env python3
"""Read a CSV's header and run it through the rule classifier.

Usage:
    python scripts/inspect_header.py path/to/file.csv

Prints the control-class breakdown and, crucially, any features the rules
could not classify (.unknown) -- those are the ones needing rule tuning for a
new feature set such as LYCOS. Does not load the full file; reads only the
header row, so it is instant even on multi-GB CSVs.
"""
from __future__ import annotations

import csv
import sys

from evasion_arms_race.features.rule_classifier import classify


def read_header(path: str) -> list[str]:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        return next(reader)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    path = argv[1]
    header = read_header(path)
    print(f"Read {len(header)} columns from {path}\n")

    report = classify(header)
    print(report.summary())
    print()

    if report.unknown:
        print(f"UNKNOWN -- need rule tuning ({len(report.unknown)}):")
        for u in report.unknown:
            print(f"  {u!r}")
    else:
        print("No unknowns: every column was classified by an existing rule.")

    # Also dump the per-feature assignment for review.
    print("\nFull assignment (feature -> class):")
    for feat in sorted(report.assignments):
        print(f"  {feat:40s} {report.assignments[feat].value}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except BrokenPipeError:
        # Occurs when piping output to head/less; not an error.
        sys.stderr.close()