"""Stage A — WHERE: static crash risk per road segment.

Design (what road-safety practice uses, kept small):
  A1  Negative-binomial safety performance function (SPF): log(expected KSI) = log(length) + b·x.
      The Highway Safety Manual standard: interpretable rate ratios, and it gives the per-segment "reasons".
  A2  Gradient-boosted Poisson trees: same target, picks up non-linear effects the SPF misses.
  ENS Weighted mix of A1 and A2. The weight is chosen on the 2023-24 data only (out-of-fold deviance).
  EB  Empirical Bayes: blends the model with the segment's own crash history (HSM network screening).
Honest test: everything is fit on 2023-24; scored against Jan 2025 - May 2026 crashes, next to the
"past crashes" ranking that blackspot lists use. Folds are 0.2° spatial blocks, so a segment is never
scored by a model that saw its neighbourhood.

Input : 2_readable/10_model_inputs/where_segments.csv
Output: 2_readable/11_model_results/A_static_risk/*
"""
import os, sys, json, time, numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
IN = os.path.join(ROOT, '2_readable', '10_model_inputs', 'where_segments.csv')
OUT = os.path.join(ROOT, '2_readable', '11_model_results', 'A_static_risk'); os.makedirs(OUT, exist_ok=True)
t0 = time.time()
S = pd.read_csv(IN, low_memory=False)
TRAIN_MONTHS, TEST_MONTHS = 24, 17          # 2023-24 ; Jan 2025 - May 2026 (iRAD has no Jan/Jul 2024 crashes: disclosed)
y = S.ksi_train.values.astype(float)
L_km = np.maximum(S.length_m.values / 1000, 0.01)

# ---------------- features
hw = S.highway.fillna('other')
top_hw = ['trunk', 'primary', 'secondary', 'tertiary', 'residential', 'unclassified']
S['hw_group'] = np.where(hw.str.replace('_link', '').isin(top_hw), hw.str.replace('_link', ''), 'minor_other')
# A1 SPF: few, transformed, readable terms (reference class = minor_other)
spf = pd.DataFrame({
    'trunk': (S.hw_group == 'trunk'), 'primary': (S.hw_group == 'primary'), 'secondary': (S.hw_group == 'secondary'),
    'tertiary': (S.hw_group == 'tertiary'), 'residential': (S.hw_group == 'residential'), 'unclassified': (S.hw_group == 'unclassified'),
    'national_highway': S.is_national_highway == 1,
    'log_turn_deg_per_km': np.log1p(S.turn_deg_per_km.clip(0, 5000)),
    'junctions_per_km': S.junctions_per_km.clip(0, 40),
    'rail_crossing': S.n_rail_crossing > 0,
    'log_ped_places_200m': np.log1p(S.poi200_total),
    'log_night_light': np.log1p(S.night_light_nw.fillna(0).clip(0, 200)),
    'log_nearby_ksi_1km': np.log1p(S.nearby_ksi_1km_train),
    'log_road_km_1km': np.log1p(S.nearby_road_km_1km),
}).astype(float)
Xspf = sm.add_constant(spf)
# A2 trees: raw features
gbm_cols = ['length_m', 'sinuosity', 'turn_deg_per_km', 'is_main_road', 'is_national_highway', 'lanes_num', 'is_bridge', 'junctions',
            'junctions_per_km', 'n_rail_crossing', 'n_speed_breaker', 'n_signal', 'n_ped_crossing', 'n_bus_stop', 'poi200_market_shop',
            'poi200_education', 'poi200_health', 'poi200_transit', 'poi200_eatery', 'poi200_worship', 'poi200_fuel', 'poi200_total',
            'night_light_nw', 'nearby_ksi_1km_train', 'nearby_crashes_1km_train', 'nearby_road_km_1km', 'mid_lat', 'mid_lon']
Xg = pd.concat([S[gbm_cols], pd.get_dummies(S.hw_group, prefix='hw').astype(int)], axis=1)

# spatial folds: 0.2° blocks -> 5 folds
blk = np.floor(S.mid_lat / 0.2).astype(int).astype(str) + '_' + np.floor(S.mid_lon / 0.2).astype(int).astype(str)
ub = blk.unique(); rng = np.random.default_rng(42); fold = blk.map(dict(zip(ub, rng.integers(0, 5, len(ub))))).values

# ---------------- A1: NB SPF. alpha estimated once on all train data, then GLM refits per fold (fast IRLS)
nb_full = sm.NegativeBinomial(y, Xspf, offset=np.log(L_km)).fit(disp=0, maxiter=200)
alpha = float(nb_full.params['alpha'])
fam = sm.families.NegativeBinomial(alpha=alpha)
oof_a1 = np.zeros(len(S)); oof_a2 = np.zeros(len(S))
gbm_kw = dict(loss='poisson', max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0, random_state=0)
for f in range(5):
    tr, te = fold != f, fold == f
    g = sm.GLM(y[tr], Xspf[tr], family=fam, offset=np.log(L_km[tr])).fit()
    oof_a1[te] = g.predict(Xspf[te], offset=np.log(L_km[te]))
    m = HistGradientBoostingRegressor(**gbm_kw).fit(Xg[tr], y[tr])
    oof_a2[te] = m.predict(Xg[te])
    print(f'fold {f} done  ({time.time() - t0:.0f}s)')
gbm_full = HistGradientBoostingRegressor(**gbm_kw).fit(Xg, y)

def pdev(yv, mu):
    mu = np.maximum(mu, 1e-9); t = np.where(yv > 0, yv * np.log(np.maximum(yv, 1e-9) / mu), 0)
    return float(2 * np.sum(t - (yv - mu)))
# ensemble weight picked on TRAIN out-of-fold deviance (never on the test period)
wgrid = np.round(np.arange(0, 1.01, 0.1), 1)
dev = {w: pdev(y, w * oof_a1 + (1 - w) * oof_a2) for w in wgrid}
w_spf = min(dev, key=dev.get)
ens = w_spf * oof_a1 + (1 - w_spf) * oof_a2

# ---------------- Empirical Bayes (HSM): w = 1/(1 + k*mu); k by method of moments on the ensemble
k = max(float(((y - ens) ** 2 - ens).sum() / (ens ** 2).sum()), 1e-3)
w_eb = 1 / (1 + k * ens)
eb = w_eb * ens + (1 - w_eb) * y

# ---------------- prospective test on Jan 2025 - May 2026
yt = S.ksi_test.values.astype(float); scale = TEST_MONTHS / TRAIN_MONTHS
def evaluate(score, name):
    d = pd.DataFrame({'sc': score + 1e-9 * S.length_m.values, 'L': S.length_m.values, 'yt': yt, 'nb': S.new_blackspot_test.values, 'bt': S.blackspot_test.values})
    d = d.sort_values('sc', ascending=False); cum = d.L.cumsum() / d.L.sum()
    r = dict(model=name)
    for q in (0.01, 0.05, 0.10):
        top = cum <= q
        r[f'test_KSI_in_top{int(q*100)}pct_km'] = round(d.yt[top].sum() / yt.sum(), 4)
        r[f'new_blackspots_in_top{int(q*100)}pct_km'] = int(d.nb[top].sum())
    return r
models = {'B  past crashes (blackspot-list approach)': y + 0.01 * S.crashes_train.values,
          'A1 NB safety performance function': oof_a1, 'A2 gradient-boosted Poisson trees': oof_a2,
          f'ENS {w_spf:.1f}*A1 + {1 - w_spf:.1f}*A2': ens, 'EB ensemble + own history (final)': eb}
ev = pd.DataFrame([evaluate(v, k_) for k_, v in models.items()])
ev['test_poisson_deviance'] = [round(pdev(yt, np.maximum(v, 0) * scale), 0) if not k_.startswith('B') else None for k_, v in models.items()]
ev.insert(1, 'new_blackspots_total', int(S.new_blackspot_test.sum()))
ev.to_csv(os.path.join(OUT, 'A_prospective_test.csv'), index=False)
print(ev.T.to_string())

# calibration: predicted (scaled to test months) vs observed test KSI by decile of the final score
cal = pd.DataFrame({'pred': eb * scale, 'obs': yt, 'L': S.length_m / 1000})
cal['decile'] = pd.qcut(pd.Series(eb).rank(method='first'), 10, labels=range(1, 11)).astype(int)
cal = cal.groupby('decile').agg(segments=('obs', 'size'), road_km=('L', 'sum'), predicted_test_KSI=('pred', 'sum'), observed_test_KSI=('obs', 'sum')).round(1)
cal['observed_over_predicted'] = (cal.observed_test_KSI / cal.predicted_test_KSI).round(2)
cal.to_csv(os.path.join(OUT, 'A_calibration_by_decile.csv'))
# note: KSI reported per month rose between periods, so absolute levels run high; the ranking (what the signal uses) is what is tested
level_shift = round((yt.sum() / TEST_MONTHS) / (y.sum() / TRAIN_MONTHS), 3)

# SPF coefficients as rate ratios (readable), and tree importance
ci = nb_full.conf_int()
coef = pd.DataFrame({'term': nb_full.params.index, 'coefficient': nb_full.params.values, 'rate_ratio': np.exp(nb_full.params.values),
                     'rr_ci95_low': np.exp(ci[0].values), 'rr_ci95_high': np.exp(ci[1].values), 'p_value': nb_full.pvalues.values}).round(4)
coef.loc[coef.term == 'alpha', ['rate_ratio', 'rr_ci95_low', 'rr_ci95_high']] = np.nan
coef.to_csv(os.path.join(OUT, 'A1_spf_rate_ratios.csv'), index=False)
from sklearn.inspection import permutation_importance
smp = np.random.default_rng(1).choice(len(S), 30000, replace=False)
pi = permutation_importance(gbm_full, Xg.iloc[smp], y[smp], n_repeats=3, random_state=0, scoring='neg_mean_poisson_deviance')
pd.DataFrame({'feature': Xg.columns, 'importance': pi.importances_mean}).sort_values('importance', ascending=False).round(6) \
  .to_csv(os.path.join(OUT, 'A2_tree_feature_importance.csv'), index=False)

# ---------------- deployment score: same models, EB updated with ALL crash history (2023 - May 2026)
mu_all = ens * (TRAIN_MONTHS + TEST_MONTHS) / TRAIN_MONTHS
w_all = 1 / (1 + k * mu_all)
eb_all = w_all * mu_all + (1 - w_all) * (S.ksi_train.values + S.ksi_test.values)
risk_km_yr = eb_all / L_km / ((TRAIN_MONTHS + TEST_MONTHS) / 12)       # expected KSI crashes per km per year

# reasons: the two SPF terms that push this segment's risk up most (vs the Assam average segment)
labels = {'trunk': 'trunk road', 'primary': 'primary road', 'secondary': 'secondary road', 'tertiary': 'tertiary road', 'residential': 'residential street',
          'unclassified': 'unclassified road', 'national_highway': 'national highway', 'log_turn_deg_per_km': 'bends', 'junctions_per_km': 'many junctions',
          'rail_crossing': 'rail level crossing', 'log_ped_places_200m': 'pedestrian places', 'log_night_light': 'busy lit area',
          'log_nearby_ksi_1km': 'serious crashes nearby', 'log_road_km_1km': 'dense road network'}
# only terms whose rate ratio is > 1 AND significant, and only where this segment is above average on them
b = nb_full.params[spf.columns].values; pv = nb_full.pvalues[spf.columns].values
use = (b > 0) & (pv < 0.05)
contrib = np.where(use, np.clip((spf.values - spf.values.mean(0)) * b, 0, None), 0)
top2 = np.argsort(-contrib, axis=1)[:, :2]
names = np.array([labels[c] for c in spf.columns])
reason = [' + '.join(names[i][contrib[r, i] > 0.1]) or 'no standout factor' for r, i in enumerate(top2)]

out = pd.DataFrame({'segment_id': S.segment_id, 'highway': S.highway, 'name': S.name, 'ref': S.ref, 'mid_lat': S.mid_lat, 'mid_lon': S.mid_lon,
                    'length_m': S.length_m, 'night_light_class': S.night_light_class, 'poi200_total': S.poi200_total,
                    'spf_expected_ksi_2yr': oof_a1.round(4), 'gbm_expected_ksi_2yr': oof_a2.round(4), 'ensemble_expected_ksi_2yr': ens.round(4),
                    'eb_expected_ksi_2yr_test_model': eb.round(4), 'deploy_expected_ksi_per_km_year': risk_km_yr.round(4), 'main_reasons': reason,
                    'ksi_train': S.ksi_train, 'ksi_test': S.ksi_test, 'blackspot_train': S.blackspot_train, 'new_blackspot_test': S.new_blackspot_test})
out.to_csv(os.path.join(OUT, 'A_static_risk_scores.csv'), index=False)
json.dump(dict(ksi_per_month_test_over_train=level_shift, nb_alpha=round(alpha, 4), eb_k=round(k, 4), ensemble_weight_on_spf=float(w_spf), train_oof_deviance_by_weight={str(a): round(v, 1) for a, v in dev.items()},
               folds='5 spatial folds of 0.2 degree blocks', gbm=gbm_kw, n_segments=len(S), runtime_s=round(time.time() - t0)),
          open(os.path.join(OUT, 'A_model_settings.json'), 'w'), indent=1)
print(f'Stage A done in {time.time() - t0:.0f}s | alpha={alpha:.3f} k={k:.3f} w_spf={w_spf}')
