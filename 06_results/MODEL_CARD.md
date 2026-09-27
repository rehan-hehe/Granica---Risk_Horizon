# Risk Horizon: model card

**What it does:** gives every 500 m of road in Assam a crash-risk band from 0 to 3 for the current light and weather, so a driver-assist car can warn or slow down before reaching a dangerous stretch.

**How long it takes:** the full pipeline trains in about 2–3 minutes on a laptop CPU.

**Data:** every model reads only the readable CSVs in `2_readable/10_model_inputs/`.

## Architecture

![architecture](figures/01_model_architecture.png)

| Stage | Question | Method | Why this method | Input CSV |
|---|---|---|---|---|
| **A1** | Which roads are dangerous? | Negative-binomial safety performance function (SPF) with a road-length offset | Highway Safety Manual standard. Its rate ratios are easy to read and provide the "reasons" in the signal. | `where_segments.csv` |
| **A2** | Same | Gradient-boosted Poisson trees (sklearn HistGradientBoosting, 300 trees) | Tree boosting (XGBoost/RF) beats NB in SPF comparisons; it captures non-linear effects. | `where_segments.csv` |
| **Blend** | Same | 0.3 × A1 + 0.7 × A2 | Weight chosen on 2023–24 out-of-fold deviance only | – |
| **EB** | Same | Empirical Bayes blend with the segment's own crash history (k = 0.74) | Best-tested hotspot-identification method in the literature | – |
| **B1** | Do rain or fog make crashes more likely? | Time-stratified case-crossover + conditional logistic regression | Standard design for weather-and-crash studies: place and clock time cancel out | `internal_do_not_share/when_crash_hours.csv` |
| **B2** | Do conditions make crashes deadlier? | Logistic regression: fatal vs non-fatal, on light, street lighting, road class, weather, season | Simple, interpretable, stable across periods | Same |
| **C** | What does the car receive? | risk now = static risk × darkness multiplier × weather multiplier, cut into bands by share of road length | Fixed cut-offs, so darkness can push a segment up a band | Outputs of A and B |

**Safety rule:** a condition can raise risk but never lower it. A multiplier below 1 is set to 1.

## Results (trained on 2023–24 only, tested on Jan 2025 – May 2026)

| Ranking | Serious crashes in top 5% of road length | New blackspots flagged (of 91) | Serious crashes in top 1% |
|---|---|---|---|
| Past crashes (today's blackspot-list approach) | 37.2% | 64 | 14.1% |
| A1 SPF | 48.0% | 83 | 15.4% |
| A2 trees | 47.7% | 80 | 14.1% |
| Blend | 48.8% | 86 | 15.0% |
| **Final (EB)** | **49.2%** | **85** | **16.4%** |

**Band validation (test model):**

| Band | Share of road | Share of 2025–26 serious crashes | Serious crashes per 100 km |
|---|---|---|---|
| 3 very high | 1% | 16.7% | 290 |
| 2 high | 4% | 33.1% | 144 |
| 1 elevated | 15% | 31.5% | 36 |
| 0 low | 80% | 18.7% | 4 |

**Conditions:**

- **Rain slightly lowers crash likelihood:** matched OR 0.96 (0.92–0.99). Fog has no clear effect. Rain across main roads shows no interaction. Rain's multiplier is therefore 1.0.
- **Darkness raises the odds that a crash is fatal:** OR 1.43 (1.27–1.60). Twilight: OR 1.22 (1.04–1.44).
  - Darkness multiplier: 1.21 on dark rural roads up to 1.28 in bright urban areas.
  - Rural and dim roads are deadlier at any hour. Their fatal share is 29% by day on dark rural roads, against 14% by day in bright urban areas.
- **Alert load:** band 2 or higher covers 5.0% of road length by day and 6.0% after dark.

## Limits (say these in the pitch)

- **No traffic volumes.** Risk is per km, not per vehicle, so busy roads rank high partly for being busy.
- **Absolute levels run about 40% low.** Reported serious crashes per month rose 46% in 2025–26. The ranking, which the signal uses, holds.
- **The severity model is weak for single crashes** (AUC 0.56 on 2025–26). It is used only for average multipliers.
- **Reason flags come from the SPF only,** not the tree model. Bends lower the SPF rate (likely slower driving), so bends are never listed as a reason.
- **Rain can look protective** because people travel less and slower in rain. Without traffic counts we cannot separate the two, so we never lower a warning for rain.

## Files

| File | What it is |
|---|---|
| `A_static_risk/A_static_risk_scores.csv` | Every segment: A1, A2, blend, EB scores, deploy risk, reasons |
| `A_static_risk/A_prospective_test.csv`, `A_calibration_by_decile.csv`, `A1_spf_rate_ratios.csv`, `A2_tree_feature_importance.csv` | Test results and interpretation |
| `B_conditions/B1_*.csv`, `B2_*.csv` | Condition odds ratios and darkness multipliers |
| `C_signal/C_segment_signal_table.csv` | **The signal:** band by day, twilight, dark and rain; reasons; pedestrian flag; confidence |
| `C_signal/C_band_definitions.csv`, `C_band_validation_2025_26.csv`, `C_alert_load_by_condition.csv` | Band rules, validation, alert load |
| `Risk_Horizon_Model_Results.xlsx` | All of the above in one workbook |
| `figures/` | 8 figures for the slides |

**To re-run:** `3_code/07_model/run_all.bat` on Windows, or `bash run_all.sh`.
