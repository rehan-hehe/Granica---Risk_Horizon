#!/bin/bash
# Risk Horizon model pipeline (~3 min). Run from anywhere: bash run_all.sh
cd "$(dirname "$0")"; set -e
ROOT="$(cd ../.. && pwd)"
for s in 00_export_model_inputs 01_static_risk 02_condition_multipliers 03_signal_bands 04_report; do
  echo "=== $s"; python3 $s.py "$ROOT"
done
echo "All done -> $ROOT/2_readable/11_model_results"
