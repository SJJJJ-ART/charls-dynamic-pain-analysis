#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 /path/to/harmonized-charls /private/output/root" >&2
  exit 2
fi

RAW_DIR=$1
OUTPUT_ROOT=$2
DERIVED_DIR="$OUTPUT_ROOT/derived"
PRIMARY_DIR="$OUTPUT_ROOT/primary"
SECONDARY_DIR="$OUTPUT_ROOT/secondary"
FULL_DIR="$OUTPUT_ROOT/full-sensitivities"
FIGURE_DIR="$OUTPUT_ROOT/figures"

export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

python code/01_build_analysis_data.py \
  --raw-dir "$RAW_DIR" \
  --output-dir "$DERIVED_DIR"

python code/02_run_analysis.py \
  --raw-dir "$RAW_DIR" \
  --derived-dir "$DERIVED_DIR" \
  --output-dir "$PRIMARY_DIR" \
  --imputations 30 --jobs 4

python code/03_run_secondary.py \
  --primary-results-dir "$PRIMARY_DIR" \
  --output-dir "$SECONDARY_DIR" \
  --imputations 30 --jobs 4

python code/04_run_full_sensitivities.py \
  --raw-dir "$RAW_DIR" \
  --primary-results-dir "$PRIMARY_DIR" \
  --output-dir "$FULL_DIR" \
  --imputations 30 --jobs 4 --bootstrap-replicates 500

python code/05_validate_results.py \
  --derived-dir "$DERIVED_DIR" \
  --primary-results-dir "$PRIMARY_DIR" \
  --secondary-results-dir "$SECONDARY_DIR" \
  --full-sensitivity-dir "$FULL_DIR" \
  --report "$OUTPUT_ROOT/validation_report.md"

python code/06_make_figures.py \
  --secondary-results-dir "$SECONDARY_DIR" \
  --full-sensitivity-dir "$FULL_DIR" \
  --output-dir "$FIGURE_DIR"

echo "Analysis complete. Review $OUTPUT_ROOT/validation_report.md."
