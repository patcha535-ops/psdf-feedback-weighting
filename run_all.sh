#!/bin/sh
# Full reproduction.  Usage: sh run_all.sh [prov.csv] [mots2024.xlsx mots2025.xlsx mots2026.xlsx]
set -e
python3 test_temporal_integrity.py
python3 test_empirical.py
if [ -n "$1" ]; then python3 run_experiments.py --outdir results --features "$1"; else python3 run_experiments.py --outdir results; fi
python3 seed_robustness.py --outdir results --n-seeds 20
if [ -n "$4" ]; then python3 build_panel.py "$2" "$3" "$4" panel.csv "$1"; fi
python3 empirical_tourism.py panel.csv --outdir results_empirical
