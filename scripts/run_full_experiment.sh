#!/usr/bin/env bash
# end-to-end pipeline. assumes:
#   * .env carries GROQ_API_KEY
#   * banking77 features parquet already built (python -m evofeat.datasets.banking77 --build)
#   * any vLLM-hosted backbones in the experiment are already running locally
#     (or skipped via --only)
set -euo pipefail

EXP=${1:-configs/experiments/banking77_full.yaml}

# 0) data prep (idempotent)
python -m evofeat.datasets.banking77 --build

# 1) classical baselines first — fast, cheap, doesn't need the LLM
python scripts/run_baselines.py --experiment "$EXP"

# 2) per-backbone evolutionary search
python scripts/run_evofeat.py --experiment "$EXP"

# 3) statistical comparison + SHAP + reporting
python scripts/run_comparison.py --experiment "$EXP"

echo "results written to results/. open results/tables/headline.md for the summary."
