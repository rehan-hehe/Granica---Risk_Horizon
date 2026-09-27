# 05 · Model

## Architecture

```
WHERE  A1 negative-binomial safety performance function (20%) ┐
       A2 gradient-boosted Poisson trees (80%)                ┴▶ blend ▶ Empirical Bayes with the segment's own crash record
WHEN   B1 conditional logit (does rain/fog raise crash odds?)  B2 logistic severity (does darkness raise fatality?)
SIGNAL risk = WHERE × darkness multiplier (× weather, kept at 1.0) ▶ bands: 3 = top 1% of road, 2 = next 4%, 1 = next 15%, 0 = rest
NOW    live TomTom speed decides when to warn
```

## Time split and why

| Role | Window | Use |
|---|---|---|
| Selection-train | Jan 2023 – Jun 2024 | Fit candidates in the feature study |
| Validation | Jul 2024 – Jun 2025 | Choose features, parameters, blend weight |
| Final training | Jan 2023 – Jun 2025 (28 data months, ~72% of crashes) | Fit the final model |
| **Test (untouched)** | **Jul 2025 – May 2026** | Report results once |

**Why:**
- **Split by time, not at random:** a car uses the past to warn about the future, so a random split would leak future crashes into training.
- **Spatial folds:** all scores are out-of-fold over 0.2° spatial blocks, so a segment is never scored by a model that saw its own area.

## Code

| Folder | Scripts | Runtime |
|---|---|---|
| `v2_final/` | `01_feature_study.py` (selection, ~15 min), `02_static_risk.py` (final WHERE model + test + bootstrap, ~3 min), `03_condition_likelihood.py` (B1), `04_severity_multipliers.py` (B2), `final_features.json` | laptop CPU |
| `v1_signal_pipeline/` | The first complete pipeline including **signal bands and car actions** (`03_signal_bands.py`), report and `run_all.bat` | ~3 min |
| `v0_prototype/` | First road model used for the early headline test | — |

Scripts expect the project's dataset layout (`<root>/2_readable/10_model_inputs/...`); pass that root as the first argument.
