"""Headline test for Risk Horizon: using ONLY 2023-24 information, can we find the road segments that are dangerous in 2025-26
— including the ones that were not dangerous before — better than the usual approach (ranking by past crashes)?

Scores compared (all built from 2023-24 data only):
  B  history      : fatal+grievous (KSI) crashes on the segment in 2023-24 (what blackspot lists do)
  A  road-model   : Safety-performance model — expected KSI from road features, nearby crash density (excluding the segment itself),
                    night light, pedestrian places. Out-of-fold predictions (spatial blocks) so a segment never scores from its own crashes.
  EB combined     : Empirical-Bayes blend of A and the segment's own history (Highway Safety Manual method).
Evaluation on 2025-26: share of KSI crashes inside the top 1% / 5% of road length, and how many NEW blackspots are flagged.
"""
import os, sys, json, numpy as np, pandas as pd, shapely
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import BallTree

BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(BASE, 'model_ready')
S = pd.read_parquet(os.path.join(OUTD, 'where_segments.parquet'))

# nearby crash density: KSI crashes (2023-24) on OTHER segments within 1 km of the midpoint
xy = np.radians(S[['mid_lat', 'mid_lon']].values)
bt = BallTree(xy, metric='haversine')
nb = bt.query_radius(xy, r=1.0 / 6371.0)
ksi = S.ksi_train.values; cr = S.crashes_train.values
S['nearby_ksi_1km_train'] = [ksi[n].sum() - ksi[i] for i, n in enumerate(nb)]
S['nearby_crashes_1km_train'] = [cr[n].sum() - cr[i] for i, n in enumerate(nb)]
S['nearby_road_km_1km'] = [S.length_m.values[n].sum() / 1000 for n in nb]

hw = pd.get_dummies(S.highway, prefix='hw').astype(int)
FEATS = ['length_m', 'sinuosity', 'turn_deg_per_km', 'is_main_road', 'is_national_highway', 'lanes_num', 'is_bridge', 'junctions',
         'junctions_per_km', 'n_rail_crossing', 'n_speed_breaker', 'n_signal', 'n_ped_crossing', 'n_bus_stop',
         'poi200_market_shop', 'poi200_education', 'poi200_health', 'poi200_transit', 'poi200_eatery', 'poi200_worship', 'poi200_fuel',
         'poi200_total', 'night_light_nw', 'nearby_ksi_1km_train', 'nearby_crashes_1km_train', 'nearby_road_km_1km', 'mid_lat', 'mid_lon']
X = pd.concat([S[FEATS], hw], axis=1)
y = S.ksi_train.values

# spatial block CV: 0.2° blocks (~22 km) assigned to 5 folds
blk = (np.floor(S.mid_lat / 0.2).astype(int).astype(str) + '_' + np.floor(S.mid_lon / 0.2).astype(int).astype(str))
ub = blk.unique(); rng = np.random.default_rng(42); fold_of = dict(zip(ub, rng.integers(0, 5, len(ub))))
fold = blk.map(fold_of).values
oof = np.zeros(len(S))
for f in range(5):
    tr, te = fold != f, fold == f
    m = HistGradientBoostingRegressor(loss='poisson', max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100, random_state=0)
    m.fit(X[tr], y[tr]); oof[te] = m.predict(X[te])
S['spf_expected_ksi_2yr'] = np.round(oof, 4)
full = HistGradientBoostingRegressor(loss='poisson', max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100, random_state=0).fit(X, y)

# Empirical Bayes (HSM): w = 1 / (1 + k * mu); k (overdispersion) by method of moments
mu = np.maximum(oof, 1e-6)
k = max(((y - mu) ** 2 - mu).sum() / (mu ** 2).sum(), 1e-3)
w = 1 / (1 + k * mu)
S['eb_expected_ksi_2yr'] = np.round(w * mu + (1 - w) * y, 4)
S['eb_weight_on_model'] = np.round(w, 3)

def evaluate(score, name):
    d = S.assign(sc=score + 1e-9 * S.length_m).sort_values('sc', ascending=False)  # ties: longer first (neutral)
    cumkm = d.length_m.cumsum() / d.length_m.sum()
    out = dict(score=name)
    for q in (0.01, 0.05):
        top = cumkm <= q
        out[f'test_KSI_captured_top{int(q*100)}pct_km'] = round(d.loc[top, 'ksi_test'].sum() / S.ksi_test.sum(), 4)
        out[f'new_blackspots_flagged_top{int(q*100)}pct_km'] = int(d.loc[top, 'new_blackspot_test'].sum())
        out[f'all_test_blackspots_flagged_top{int(q*100)}pct_km'] = int(d.loc[top, 'blackspot_test'].sum())
    return out
res = pd.DataFrame([evaluate(S.ksi_train.values + 0.01 * S.crashes_train.values, 'B: past crashes (blackspot-list approach)'),
                    evaluate(S.spf_expected_ksi_2yr.values, 'A: road model (no own crash history)'),
                    evaluate(S.eb_expected_ksi_2yr.values, 'EB: road model + own history (Empirical Bayes)')])
res.insert(1, 'new_blackspots_total', int(S.new_blackspot_test.sum())); res.insert(2, 'test_blackspots_total', int(S.blackspot_test.sum()))
print(res.T.to_string())

# ranks and risk bands for the map / live layer (EB score per km, 2-year horizon)
S['risk_per_km'] = S.eb_expected_ksi_2yr / np.maximum(S.length_m / 1000, 0.1)
pct = S.risk_per_km.rank(pct=True)
S['risk_band'] = pd.cut(pct, [0, 0.80, 0.95, 0.99, 1.0], labels=['low', 'elevated', 'high', 'very_high'], include_lowest=True).astype(str)
S.to_parquet(os.path.join(OUTD, 'where_segments_scored.parquet'), index=False)
res.to_csv(os.path.join(OUTD, 'headline_test_results.csv'), index=False)
# feature importance (permutation on a sample, for explainability)
from sklearn.inspection import permutation_importance
smp = S.sample(40000, random_state=1).index
pi = permutation_importance(full, X.loc[smp], y[smp], n_repeats=3, random_state=0, scoring='neg_mean_poisson_deviance')
imp = pd.DataFrame({'feature': X.columns, 'importance': pi.importances_mean}).sort_values('importance', ascending=False)
imp.to_csv(os.path.join(OUTD, 'feature_importance.csv'), index=False)
print(imp.head(12).to_string())
json.dump(dict(overdispersion_k=round(k, 4), n_segments=len(S), cv='5-fold spatial blocks 0.2 deg'), open(os.path.join(OUTD, 'model_meta.json'), 'w'))
