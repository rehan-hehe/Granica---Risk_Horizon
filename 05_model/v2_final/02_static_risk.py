"""Stage A — WHERE: static crash risk per road segment (v2: engineered + selected features).

Architecture unchanged: A1 NB safety performance function + A2 Poisson boosted trees -> blend -> Empirical Bayes.
Features, SPF terms, tree settings and blend weight come from 01_feature_study.py (final_features.json), chosen on
Jan 2023-Jun 2024 -> Jul 2024-Jun 2025. This script never tunes anything.

Time split (more training data than v1):
  Final training window : Jan 2023 - Jun 2025 (28 months with data, ~72% of crashes)
  Test window (untouched): Jul 2025 - May 2026 (11 months)
Also re-scores the v1 split (2023-24 -> Jan 2025-May 2026, MoRTH blackspots) for continuity with earlier results.

Inputs : 2_readable/10_model_inputs/*.csv, internal crash_events.csv (aggregated to segment counts here)
Outputs: 2_readable/11_model_results/A_static_risk/*, 2_readable/10_model_inputs/where_model_input_final.csv
"""
import os, sys, json, time, warnings, numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rh_features as RF

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(HERE, '..', '..'))
INP = os.path.join(ROOT, '2_readable', '10_model_inputs')
OUT = os.path.join(ROOT, '2_readable', '11_model_results', 'A_static_risk'); os.makedirs(OUT, exist_ok=True)
CFG = json.load(open(os.path.join(HERE, 'final_features.json')))
t0 = time.time()
def log(m): print(f'[{time.time() - t0:4.0f}s] {m}', flush=True)

S, F, DEF, xyG = RF.static_features(INP)
E = pd.read_csv(os.path.join(INP, 'internal_do_not_share', 'crash_events.csv'))
GEO = dict(seg_lat=S.mid_lat.values, seg_lon=S.mid_lon.values, seg_len=S.length_m.values)
L = S.length_m.values; off = np.log(np.maximum(L / 1000, 0.01))
blk = np.floor(S.mid_lat / 0.2).astype(int).astype(str) + '_' + np.floor(S.mid_lon / 0.2).astype(int).astype(str)
ub = blk.unique(); fold = blk.map(dict(zip(ub, np.random.default_rng(42).integers(0, 5, len(ub))))).values

V1_COLS = ['length_m', 'sinuosity', 'turn_deg_per_km', 'is_main_road', 'is_national_highway', 'is_bridge', 'junctions', 'junctions_per_km', 'n_rail_crossing',
           'n_speed_breaker', 'n_signal', 'n_ped_crossing', 'n_bus_stop', 'poi200_market_shop', 'poi200_education', 'poi200_health', 'poi200_transit', 'poi200_eatery',
           'poi200_worship', 'poi200_fuel', 'poi200_total', 'night_light', 'nearby_ksi_1km', 'nearby_crashes_1km', 'road_km_1km', 'mid_lat', 'mid_lon', 'hw_rank']
V1_P = dict(loss='poisson', max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0, random_state=0)

def eb_blend(mu, yy):
    k = max(float(((yy - mu) ** 2 - mu).sum() / (mu ** 2).sum()), 1e-3); w = 1 / (1 + k * mu); return w * mu + (1 - w) * yy, k

def fit_window(train_months):
    """Build features on a training window, fit A1 + A2 out-of-fold, blend, EB. Returns scores and fitted objects."""
    H, own, DEFH = RF.history_features(E, S.segment_id.values, xyG, train_months, **GEO)
    X = pd.concat([F.drop(columns='segment_id'), H], axis=1); y = own['ksi']
    # A1 SPF
    Z = sm.add_constant(RF.spf_transform(X, CFG['a1_spf_terms']))
    nb = sm.NegativeBinomial(y, Z, offset=off).fit(disp=0, maxiter=300); alpha = float(nb.params['alpha'])
    a1 = np.zeros(len(S)); a2 = np.zeros(len(S)); v1 = np.zeros(len(S))
    for f in range(5):
        tr, te = fold != f, fold == f
        a1[te] = sm.GLM(y[tr], Z[tr], family=sm.families.NegativeBinomial(alpha=alpha), offset=off[tr]).fit().predict(Z[te], offset=off[te])
        a2[te] = HistGradientBoostingRegressor(**CFG['a2_params']).fit(X[CFG['a2_features']][tr], y[tr]).predict(X[CFG['a2_features']][te])
        v1[te] = HistGradientBoostingRegressor(**V1_P).fit(X[V1_COLS][tr], y[tr]).predict(X[V1_COLS][te])
    w = CFG['blend_weight_spf']; mu = w * a1 + (1 - w) * a2
    ebs, k = eb_blend(mu, y); v1eb, _ = eb_blend(v1, y)
    return dict(X=X, y=y, own=own, nb=nb, alpha=alpha, a1=a1, a2=a2, mu=mu, eb=ebs, k=k, v1eb=v1eb, H=H, DEFH=DEFH)

def evaluate(R, target, new_flag, label):
    cands = {'B  past crashes on the segment (blackspot-list approach)': R['y'] + 0.01 * R['own']['crashes'],
             'v1 previous model (old features & settings, same split)': R['v1eb'],
             'A1 NB safety performance function (v2 terms)': R['a1'], 'A2 boosted Poisson trees (v2 features)': R['a2'],
             f"Blend {CFG['blend_weight_spf']}*A1 + {1 - CFG['blend_weight_spf']:.1f}*A2": R['mu'], 'FINAL v2: blend + Empirical Bayes': R['eb']}
    rows = []
    for name, sc in cands.items():
        m = RF.capture_metrics(sc, target, L)
        o = np.argsort(-(sc + 1e-9 * L)); cum = np.cumsum(L[o]) / L.sum()
        rows.append(dict(split=label, model=name, **{k_: round(v, 4) for k_, v in m.items()},
                         new_hotspots_in_top5pct=int(new_flag[o][cum <= 0.05].sum()), new_hotspots_total=int(new_flag.sum())))
    return pd.DataFrame(rows)

def bootstrap(R, target, label, reps=300):
    """Spatial block bootstrap (0.2° blocks) of the capture@5% and AUCC20 gains."""
    rng = np.random.default_rng(3); ublk = pd.Series(blk).astype('category'); codes = ublk.cat.codes.values; nb_ = codes.max() + 1
    rows = []
    for other, sc_o in [('past crashes', R['y'] + 0.01 * R['own']['crashes']), ('v1 model', R['v1eb'])]:
        d5, da = [], []
        for _ in range(reps):
            wts = np.bincount(rng.integers(0, nb_, nb_), minlength=nb_)[codes].astype(float)
            keep = wts > 0; tt = target * wts
            m1 = RF.capture_metrics(R['eb'][keep], tt[keep], (L * wts)[keep]); m0 = RF.capture_metrics(sc_o[keep], tt[keep], (L * wts)[keep])
            d5.append(m1['cap5'] - m0['cap5']); da.append(m1['aucc20'] - m0['aucc20'])
        rows.append(dict(split=label, comparison=f'FINAL v2 minus {other}', cap5_gain=round(float(np.mean(d5)), 4), cap5_ci95=f'{np.percentile(d5, 2.5):.4f} .. {np.percentile(d5, 97.5):.4f}',
                         aucc20_gain=round(float(np.mean(da)), 4), aucc20_ci95=f'{np.percentile(da, 2.5):.4f} .. {np.percentile(da, 97.5):.4f}', reps=reps))
    return pd.DataFrame(rows)

# ================= main split: Jan 2023 - Jun 2025 -> Jul 2025 - May 2026
WF, WT = RF.month_list('2023-01', '2025-06'), RF.month_list('2025-07', '2026-05')
R = fit_window(WF); log('main split fitted')
_, ownT, _ = RF.history_features(E, S.segment_id.values, xyG, WT, **GEO)
yT = ownT['ksi']
new_hot = ((yT >= 3) & (R['y'] < 3)).astype(int)          # >=3 KSI crashes in the 11-month test, fewer than 3 in the 28 training months
res = evaluate(R, yT, new_hot, 'main: Jan23-Jun25 -> Jul25-May26'); boot = bootstrap(R, yT, 'main'); log('main split evaluated')

# ================= continuity split (v1): 2023-24 -> Jan 2025 - May 2026, MoRTH-style blackspots from the segment table
R1 = fit_window(RF.month_list('2023-01', '2024-12'))
res1 = evaluate(R1, S.ksi_test.values.astype(float), S.new_blackspot_test.values, 'continuity: 2023-24 -> Jan25-May26 (v1 split)')
log('continuity split evaluated')
pd.concat([res, res1]).to_csv(os.path.join(OUT, 'A_test_results.csv'), index=False)
boot.to_csv(os.path.join(OUT, 'A_bootstrap_ci.csv'), index=False)
print(pd.concat([res, res1])[['split', 'model', 'cap1', 'cap5', 'cap10', 'aucc20', 'new_hotspots_in_top5pct', 'new_hotspots_total']].to_string()); print(boot.to_string())

# ================= interpretation
nb = R['nb']; ci = nb.conf_int()
rr = pd.DataFrame({'term': nb.params.index, 'coefficient': nb.params.values, 'rate_ratio': np.exp(nb.params.values), 'rr_ci95_low': np.exp(ci[0].values),
                   'rr_ci95_high': np.exp(ci[1].values), 'p_value': nb.pvalues.values}).round(4)
rr.loc[rr.term == 'alpha', ['rate_ratio', 'rr_ci95_low', 'rr_ci95_high']] = np.nan
rr.to_csv(os.path.join(OUT, 'A1_spf_rate_ratios.csv'), index=False)
full = HistGradientBoostingRegressor(**CFG['a2_params']).fit(R['X'][CFG['a2_features']], R['y'])
smp = np.random.default_rng(1).choice(len(S), 40000, replace=False)
pi = permutation_importance(full, R['X'][CFG['a2_features']].iloc[smp], yT[smp] * R['y'].sum() / yT.sum(), n_repeats=3, random_state=0, scoring='neg_mean_poisson_deviance')
pd.DataFrame({'feature': CFG['a2_features'], 'test_permutation_importance': pi.importances_mean}).sort_values('test_permutation_importance', ascending=False).round(6) \
  .to_csv(os.path.join(OUT, 'A2_tree_feature_importance.csv'), index=False)
scale = len(set(E.month) & set(WT)) / R['own']['data_months']
cal = pd.DataFrame({'pred': R['eb'] * scale, 'obs': yT, 'km': L / 1000, 'decile': pd.qcut(pd.Series(R['eb']).rank(method='first'), 10, labels=range(1, 11)).astype(int)})
cal = cal.groupby('decile').agg(segments=('obs', 'size'), road_km=('km', 'sum'), predicted_test_KSI=('pred', 'sum'), observed_test_KSI=('obs', 'sum')).round(1)
cal['observed_over_predicted'] = (cal.observed_test_KSI / cal.predicted_test_KSI).round(2); cal.to_csv(os.path.join(OUT, 'A_calibration_by_decile.csv'))

# ================= the exact model input (training window) as a readable table
Xin = R['X'][sorted(set(CFG['a2_features']) | set(CFG['a1_spf_terms']))].round(5)
Xin.insert(0, 'segment_id', S.segment_id.values); Xin['target_ksi_train_Jan23_Jun25'] = R['y']; Xin['ksi_test_Jul25_May26'] = yT
Xin.to_csv(os.path.join(INP, 'where_model_input_final.csv'), index=False)
DEFall = pd.concat([DEF, R['DEFH']]).drop_duplicates('feature').set_index('feature')
pd.DataFrame([dict(feature=c, used_in_A2_trees=c in CFG['a2_features'], used_in_A1_spf=c in CFG['a1_spf_terms'], **DEFall.loc[c].to_dict()) for c in Xin.columns[1:-2]]) \
  .to_csv(os.path.join(INP, 'where_model_input_final_dictionary.csv'), index=False)

# ================= deployment score: model from the training window + Empirical Bayes with ALL history (Jan 2023 - May 2026)
Hall, ownAll, _ = RF.history_features(E, S.segment_id.values, xyG, RF.month_list('2023-01', '2026-05'), **GEO)
m_all = ownAll['data_months']; mu_all = R['mu'] * m_all / R['own']['data_months']
w_all = 1 / (1 + R['k'] * mu_all); eb_all = w_all * mu_all + (1 - w_all) * ownAll['ksi']
risk = eb_all / np.maximum(L / 1000, 0.01) / (m_all / 12)
terms = CFG['a1_spf_terms']; Zt = RF.spf_transform(R['X'], terms); b = nb.params[terms].values; pv = nb.pvalues[terms].values
use = (b > 0) & (pv < 0.05); contrib = np.where(use, np.clip((Zt.values - Zt.values.mean(0)) * b, 0, None), 0)
LAB = {'hw_rank': 'higher road class', 'is_main_road': 'main road', 'is_national_highway': 'national highway', 'turn_contrast': 'sharp bend after straight',
       'junctions_per_km': 'many junctions', 'junctions_1km': 'junction-dense area', 'n_rail_crossing': 'rail level crossing', 'poi_500m': 'pedestrian places',
       'poi_2km': 'large settlement', 'school_500m': 'school nearby', 'ped_places_x_main': 'pedestrians on a main road', 'night_light': 'busy lit area',
       'light_contrast': 'lit strip in dark area', 'dist_to_town_km': 'far from town', 'road_km_5km': 'dense road network', 'main_road_km_1km': 'main-road corridor',
       'nearby_ksi_1km_rel': 'serious crashes nearby', 'nearby_ksi_3km_rel': 'serious crashes in area', 'is_bridge': 'bridge', 'is_divided': 'divided highway',
       'turn_deg_per_km': 'bends', 'length_m': 'long segment', 'dist_to_main_road_km': 'far from main road', 'night_light_10km': 'developed region'}
names = np.array([LAB.get(t, t.replace('_', ' ')) for t in terms]); top2 = np.argsort(-contrib, axis=1)[:, :2]
reason = [' + '.join(names[i][contrib[r, i] > 0.1]) or 'no standout factor' for r, i in enumerate(top2)]
out = pd.DataFrame({'segment_id': S.segment_id, 'highway': S.highway, 'name': S.name, 'ref': S.ref, 'mid_lat': S.mid_lat, 'mid_lon': S.mid_lon, 'length_m': S.length_m,
                    'night_light_class': S.night_light_class, 'poi200_total': S.poi200_total, 'spf_expected_ksi': R['a1'].round(4), 'gbm_expected_ksi': R['a2'].round(4),
                    'blend_expected_ksi': R['mu'].round(4), 'eb_score_test_model': R['eb'].round(4), 'deploy_expected_ksi_per_km_year': risk.round(4), 'main_reasons': reason,
                    'ksi_train_window': R['y'].astype(int), 'ksi_test_window': yT.astype(int), 'new_hotspot_test': new_hot})
out.to_csv(os.path.join(OUT, 'A_static_risk_scores.csv'), index=False)
json.dump(dict(train_window=[WF[0], WF[-1]], test_window=[WT[0], WT[-1]], train_data_months=R['own']['data_months'], nb_alpha=round(R['alpha'], 4), eb_k=round(R['k'], 4),
               config=CFG, runtime_s=round(time.time() - t0)), open(os.path.join(OUT, 'A_model_settings.json'), 'w'), indent=1)
log('Stage A v2 done')
