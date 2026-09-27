# 06 · Results and what they mean

**Held-out test (trained Jan 2023–Jun 2025, judged on Jul 2025–May 2026):**

| Model | Serious crashes in top 1% of road | top 5% | top 10% | New hotspots flagged (of 236) |
|---|---|---|---|---|
| Blackspot list (past crashes) | 15.2% | 39.5% | 53.9% | 124 |
| Previous model v1 | 16.5% | 49.3% | 67.4% | 185 |
| **Final (blend + Empirical Bayes)** | **17.2%** | **49.7%** | **68.2%** | **184** |

**What it means:**
- **Against the blackspot list:** on the same 4,060 km, the final model covers a quarter more future serious crashes (+10.6 pts, 95% CI 8.1–12.2).
- **Against v1:** the gain from feature engineering is small but statistically clear on overall ranking (+0.006, CI 0.003–0.009).
- **Bands:** band 3 (1% of road) holds 17.7% of future serious crashes, at 83× the rate of band 0.
- **Conditions:**
  - rain slightly lowers crash odds (0.955), so the signal never lowers a warning for rain
  - darkness raises the odds a crash is fatal ×1.39 (1.19–1.61), giving a darkness multiplier of ~1.18
  - after dark, the road flagged band 2+ grows from 5.0% to 5.9%

| Folder | Contents |
|---|---|
| `01_feature_study/` | `feature_catalog.csv` (every candidate: source, missing %, correlation, mutual information, validation importance, kept or dropped and why), `experiment_log.csv`, `group_ablation.csv`, PCA, tuning, blend weight, correlation matrix |
| `02_static_risk_final/` | Test results, bootstrap CIs, calibration, regression rate ratios, tree importance, band validation, settings |
| `03_conditions/` | B1 odds ratios (rain, fog, rain × main road, per period); B2 model comparison, odds ratios and darkness multipliers |
| `04_signal_bands/` | Band definitions with vehicle actions, band validation, alert load, **segment signal table** (2 parts, 246,789 segments) |
| `05_v1_baseline/` | v1 results, kept for comparison |
| `figures/` | Slide figures in story order |

**Found and fixed in our own code:**
- **Leak 1:** float residue from fast grid sums leaked each segment's own crash count into a neighbour feature.
- **Leak 2:** subtracting own crashes from a regional total made that total a copy of the segment's count.

Both were caught by rebuilding the previous model and demanding an exact match, then fixed before the final test.

**Open item:** the driver-facing reasons come from the regression (A1), which showed a convergence warning with 25 terms. A smaller term set is the next fix.
