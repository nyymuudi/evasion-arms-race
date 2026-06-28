# Paper sources

Venue-agnostic preprint of the methodological result (post-filter vs.
manifold-constrained realisability measurement). Written against a standard
`article` class with only standard packages, so it builds anywhere; for a
camera-ready, swap `\documentclass` for the target venue's template (the body
needs no changes).

## Files
- `main.tex` — the paper.
- `references.bib` — bibliography (15 entries).
- `make_figures.py` — regenerates `figures/*.pdf` from the committed experiment
  JSON under `experiments/` (run from the repo root).
- `figures/three_levels.pdf` — the central figure (three measurement levels).

## Build
Locally (needs TeX Live with `pdflatex` + `bibtex`):
```bash
cd docs/paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
Or upload `main.tex`, `references.bib`, and `figures/` to Overleaf.

Regenerate the figure first if the experiments were re-run:
```bash
python docs/paper/make_figures.py      # from the repo root
```

## Status / TODO before submission
- Confirm venue (CSET / WTMC / DTRAP) and switch to its template.
- Fill the author affiliation.
- Optional second figure: the manifold budget-convergence curve (already in
  `experiments/figures/manifold_budget_curve.png`) if a reviewer wants the
  not-a-weak-attacker evidence shown rather than stated.
