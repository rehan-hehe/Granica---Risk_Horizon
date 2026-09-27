# 05 · Model (final version)

## Architecture

```
WHERE  A1 negative-binomial safety performance function (20%) ┐
       A2 gradient-boosted Poisson trees (80%)                ┴▶ blend ▶ Empirical Bayes with each segment's own crash record
WHEN   B1 conditional logit: does rain/fog raise crash odds?    B2 logistic: does darkness raise the chance a crash is fatal?
SIGNAL risk = WHERE × darkness multiplier (× weather, kept at 1.0) ▶ bands: 3 = top 1% of road, 2 = next 4%, 1 = next 15%, 0 = rest
NOW    live TomTom speed decides when to warn
```

## Time split, and why

| Role | Window | Use |
|---|---|---|
| Selection-train | Jan 2023 – Jun 2024 | Fit candidates in the feature study |
| Validation | Jul 2024 – Jun 2025 | Choose features, parameters, blend weight |
| Final training | Jan 2023 – Jun 2025 (28 data months, ~72% of crashes) | Fit the final model |
| **Test (untouched)** | **Jul 2025 – May 2026** | Report results once |

**Why:**
- **Split by time, not at random:** a car uses the past to warn about the future.
- **Spatial folds:** scores are out-of-fold over 0.2° blocks, so no segment is scored by a model that saw its own area.

## Files, in running order ([`run_pipeline.ipynb`](run_pipeline.ipynb) runs them all)

| File | Stage | What it does | Runtime |
|---|---|---|---|
| `../04_model_inputs/code/00_export_model_inputs.py` | prep | Model tables → CSV | ~1 min |
| `../04_model_inputs/code/rh_features.py` | shared | Feature engineering (52 candidates) | — |
| `v2_final/01_feature_study.py` | selection | Redundancy, PCA test, source ablation, elimination, tuning, blend weight → `final_features.json` | ~15 min |
| `v2_final/02_static_risk.py` | A · WHERE | Final road model, prospective test, bootstrap CIs, deployment scores, reasons | ~3 min |
| `v2_final/03_condition_likelihood.py` | B1 · WHEN | Conditional logistic regression on crash vs control hours | ~2 min |
| `v2_final/04_severity_multipliers.py` | B2 · WHEN | Fatal-vs-non-fatal model; picks logistic over trees (AUC 0.571 vs 0.573); darkness multipliers | <1 min |
| `v2_final/05_signal_bands.py` | C · signal | Bands 0–3, vehicle actions, validation, alert load, segment signal table | <1 min |

**Folder layout the scripts expect:** `<ROOT>/2_readable/10_model_inputs/` holding `where_segments.csv` (from `04_model_inputs/data`), `raw_layers/` (`osm_pois.csv`, `osm_road_nodes.csv`, `viirs_night_lights_2024.csv` from `02_data/public`), and `internal_do_not_share/` (`crash_events.csv`, `when_crash_hours.csv`, built from the cleaned iRAD release). Results go to `<ROOT>/2_readable/11_model_results/`.
