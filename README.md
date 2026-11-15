# PSDF: feedback-driven weighting vs. preference elicitation (code, data, manuscript sources)

Code, derived data and LaTeX sources for *"When Does Feedback-Driven Weighting Outperform Preference Elicitation? Evidence from
Simulation and Empirical Evaluation"* (Punch Score Decision Framework, PSDF, compared with static and periodically re-elicited AHP).

## Contents
| Path | What it is |
|---|---|
| `psdf_core.py` | Simulation engine (environment, rolling/expanding/EW/static ridge, CRITIC, AHP, one replication, statistics) |
| `run_experiments.py` | Baseline, ablation, partial feedback, noise, factorial regime maps; writes CSV, LaTeX tables and number macros |
| `seed_robustness.py` | Baseline repeated under 20 independent master seeds |
| `empirical_tourism.py` | Empirical application (77 provinces x 45 months, 32 decision months) |
| `build_panel.py`, `make_prov_csv.py` | Build and validate the panel / province feature table from the MOTS workbooks and the compiled 2024 table |
| `test_temporal_integrity.py`, `test_empirical.py` | Temporal-integrity and sanity tests |
| `run_all.sh` | Reproduces everything |
| `prov.csv`, `panel.csv` | Derived data (see below) |
| `results/`, `results_empirical/` | Outputs the manuscript numbers are taken from (CSV, LaTeX tables, macros, manifests) |
| `main.tex`, `supplementary.tex` (+ PDFs) | Manuscript and supplementary material (supplementary.tex reproduces all code inline) |

## Reproduce
```
pip install -r requirements.txt
sh run_all.sh prov.csv mots2024.xlsx mots2025.xlsx mots2026.xlsx    # tests, experiments (~3 min), 20-seed robustness, panel, empirical
python make_supplement.py                                          # regenerates supplementary.tex
pdflatex main.tex && pdflatex main.tex                             # twice
```
Master seed 6621; run identifier and checksums are in `results/manifest.json`. The monthly MOTS workbooks are not redistributed here;
they are available from the Ministry of Tourism and Sports (https://www.mots.go.th/news/category/411).

## Data notice
`prov.csv` (2024 province values) and `panel.csv` (balanced 77 x 45 panel) are derived from public statistics of the Ministry of
Tourism and Sports, Thailand. Check the Ministry's terms of reuse before redistributing them. The MIT licence covers the code only.

## Archiving / citing
Create a GitHub *release* (e.g. `v1.0.0`) and let Zenodo archive it; use the DOI of that release in the manuscript
(`\RepoDOI` at the top of `main.tex`). Every new release gets a new version DOI; the concept DOI always resolves to the latest.
