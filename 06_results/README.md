# 06 · Results and what they mean

**Held-out test (trained Jan 2023–Jun 2025, judged on Jul 2025–May 2026):**

| Model | Serious crashes in top 1% of road | top 5% | top 10% | New hotspots flagged (of 236) |
|---|---|---|---|---|
| Blackspot list (past crashes) | 15.2% | 39.5% | 53.9% | 124 |
| Previous model v1 | 16.5% | 49.3% | 67.4% | 185 |
| **Final (blend + Empirical Bayes)** | **17.2%** | **49.7%** | **68.2%** | **184** |

**What the numbers mean:**
- **Against the blackspot list:** on the same 4,060 km, Risk Horizon covers a quarter more of future serious crashes (+10.6 pts, 95% CI 8.1–12.2).
- **Against v1:** the gain from feature engineering is small but statistically clear in overall ranking (+0.006, CI 0.003–0.009). Our inputs were already close to their limit; the next gain needs traffic volumes.
- **Bands:** band 3 (1% of road) holds 17.7% of future serious crashes, at **83×** the rate of band 0.
- **Darkness** raises fatal odds ×1.43 (1.27–1.60). Rain and fog do not raise crash odds.

| Folder | Contents |
|---|---|
| `01_feature_study/` | `feature_catalog.csv` (every candidate: source, missing %, correlation, mutual information, validation importance, kept or dropped and why), `experiment_log.csv` (every experiment), `group_ablation.csv` (information carried by each source), PCA variance and loadings, tuning, blend weight, correlation matrix |
| `02_static_risk_final/` | `A_test_results.csv`, `A_bootstrap_ci.csv`, calibration by decile, rate ratios, tree importance, band validation, settings |
| `03_conditions/` | Crash-likelihood odds ratios (rain, fog, rain × main road, per period), severity odds ratios, darkness multipliers |
| `04_signal_bands/` | Band definitions with vehicle actions, validation, alert load by condition |
| `05_v1_baseline/` | v1 results for comparison |
| `figures/` | Slide-ready figures (numbered in story order) and model figures |
| `MODEL_CARD.md`, `Risk_Horizon_Model_Results.xlsx` | Summary card and workbook |

**Found and fixed in our own code:**
- **Leak 1:** float residue from fast grid sums leaked each segment's own crash count into a neighbour feature.
- **Leak 2:** subtracting own crashes from a regional total made it a copy of the segment's count.

Both were caught by rebuilding the previous model and demanding an exact match. They were fixed before the final test.

**Open item:** the driver-facing "reasons" should use the v1 regression, because the v2 one showed a convergence warning.
