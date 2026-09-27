"""Stage B2 — WHEN: do current conditions make a crash deadlier? (v2: engineered pre-crash features)

Only information a car could know BEFORE a crash is allowed: light (computed sun position), time of day, weekday,
season, weather, and the engineered road features of the segment. Nothing from the crash report itself.
Compared: B2-base logistic (v1 terms) | B2-eng logistic (+ engineered road context) | B2-trees (boosted classifier, same inputs).
Fit on 2023-24 crashes, judged on 2025-26 (AUC, log-loss). The darkness multiplier per lighting class comes from the
logistic model (interpretable); the trees only tell us whether we are leaving much signal on the table.

Inputs : 2_readable/10_model_inputs/internal_do_not_share/when_crash_hours.csv (crash-level, internal) + where_segments.csv (via rh_features)
Output : 2_readable/11_model_results/B_conditions/B2_*   (aggregates only)
"""
import os, sys, json, time, warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rh_features as RF
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(HERE, '..', '..'))
INP = os.path.join(ROOT, '2_readable', '10_model_inputs')
OUT = os.path.join(ROOT, '2_readable', '11_model_results', 'B_conditions'); os.makedirs(OUT, exist_ok=True)
t0 = time.time()
T = pd.read_csv(os.path.join(INP, 'internal_do_not_share', 'when_crash_hours.csv'), low_memory=False)
c = T[(T.is_case == 1) & T.police_severity.notna() & T.light_computed.notna()].copy()
_, F, _, _ = RF.static_features(INP)
eng = ['hw_rank', 'turn_contrast', 'junctions_per_km', 'poi_500m', 'ped_places_x_main', 'dist_to_town_km', 'light_contrast', 'road_km_5km', 'is_divided']
c = c.drop(columns=[e for e in eng if e in c.columns]).merge(F[['segment_id'] + eng], on='segment_id', how='left')
c['fatal'] = (c.police_severity == 'Fatal').astype(int)
c['period'] = np.where(pd.to_datetime(c.dt_ist).dt.year <= 2024, 'train', 'test')
c['light'] = pd.Categorical(c.light_computed.replace({'civil_twilight': 'twilight'}), ['daylight', 'twilight', 'dark'])
c['lighting'] = pd.Categorical(c.night_light_class.fillna('unknown').replace({'nan': 'unknown'}), ['bright_urban', 'lit_town', 'dim', 'dark_rural', 'unknown'])
c['road'] = np.where(c.is_national_highway == 1, 'national_highway', np.where(c.is_main_road == 1, 'other_main', 'minor_or_unknown'))
c['rain'] = c.cond_rain.fillna(0); c['fog'] = c.cond_fog_likely_era5.fillna(0)
h = pd.to_datetime(c.dt_ist).dt.hour
c['time_band'] = pd.Categorical(np.select([h.between(22, 23) | h.between(0, 3), h.between(4, 6), h.between(17, 21)], ['late_night', 'early_morning', 'evening'], 'day'),
                                ['day', 'evening', 'late_night', 'early_morning'])
c['weekend'] = pd.to_datetime(c.dt_ist).dt.dayofweek.isin([5, 6]).astype(int)
for e in eng:
    c[e] = c[e].fillna(c[e].median())
c['log_poi_500m'] = np.log1p(c.poi_500m); c['log_road_km_5km'] = np.log1p(c.road_km_5km); c['log_dist_town'] = np.log1p(c.dist_to_town_km)
tr, te = c[c.period == 'train'], c[c.period == 'test']
base = "fatal ~ C(light, Treatment('daylight')) + C(lighting, Treatment('bright_urban')) + C(road, Treatment('minor_or_unknown')) + rain + fog + C(season)"
engf = base + " + C(time_band, Treatment('day')) + weekend + hw_rank + turn_contrast + junctions_per_km + log_poi_500m + log_dist_town + light_contrast + log_road_km_5km + is_divided"
rows, models = [], {}
for name, f in [('B2-base logistic (v1 terms)', base), ('B2-eng logistic (+engineered context)', engf)]:
    m = smf.logit(f, tr).fit(disp=0); models[name] = m; p = m.predict(te)
    rows.append(dict(model=name, n_terms=len(m.params) - 1, auc_train=round(roc_auc_score(tr.fatal, m.predict(tr)), 4), auc_test=round(roc_auc_score(te.fatal, p), 4),
                     logloss_test=round(log_loss(te.fatal, np.clip(p, 1e-6, 1 - 1e-6)), 4)))
num = ['hw_rank', 'turn_contrast', 'junctions_per_km', 'poi_500m', 'ped_places_x_main', 'dist_to_town_km', 'light_contrast', 'road_km_5km', 'is_divided',
       'rain', 'fog', 'weekend', 'is_national_highway', 'is_main_road', 'night_light_nw', 'sun_elevation_deg']
Xg = lambda d: pd.concat([d[num].astype(float), pd.get_dummies(d[['light', 'time_band', 'season']].astype(str)).astype(float)], axis=1)
Xtr, Xte = Xg(tr), Xg(te).reindex(columns=Xg(tr).columns, fill_value=0)
gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0, random_state=0).fit(Xtr, tr.fatal)
pg = gb.predict_proba(Xte)[:, 1]
rows.append(dict(model='B2-trees boosted classifier (same inputs)', n_terms=Xtr.shape[1], auc_train=round(roc_auc_score(tr.fatal, gb.predict_proba(Xtr)[:, 1]), 4),
                 auc_test=round(roc_auc_score(te.fatal, pg), 4), logloss_test=round(log_loss(te.fatal, pg), 4)))
comp = pd.DataFrame(rows); comp['note'] = 'fit on 2023-24 crashes, judged on 2025-26'
comp.to_csv(os.path.join(OUT, 'B2_model_comparison.csv'), index=False); print(comp.to_string())

use = 'B2-eng logistic (+engineered context)' if comp.set_index('model').loc['B2-eng logistic (+engineered context)', 'auc_test'] >= \
      comp.set_index('model').loc['B2-base logistic (v1 terms)', 'auc_test'] else 'B2-base logistic (v1 terms)'
m = models[use]; ci = m.conf_int(); mt = smf.logit(m.model.formula, te).fit(disp=0)
b2 = pd.DataFrame({'term': m.params.index, 'odds_ratio_2023_24': np.exp(m.params), 'ci95_low': np.exp(ci[0]), 'ci95_high': np.exp(ci[1]), 'p_value': m.pvalues,
                   'odds_ratio_2025_26_refit': np.exp(mt.params.reindex(m.params.index))}).round(4)
b2.to_csv(os.path.join(OUT, 'B2_severity_odds_ratios.csv'), index=False)

# darkness multiplier per lighting class from the chosen logistic model refit on ALL crashes; others at their observed mix
m_all = smf.logit(m.model.formula, c).fit(disp=0); out = []
for lc in ['bright_urban', 'lit_town', 'dim', 'dark_rural']:
    p = {}
    for lt in ['daylight', 'twilight', 'dark']:
        x = c.copy(); x['lighting'] = pd.Categorical([lc] * len(x), c.lighting.cat.categories); x['light'] = pd.Categorical([lt] * len(x), c.light.cat.categories)
        p[lt] = float(m_all.predict(x).mean())
    o = c[c.lighting == lc]
    out.append(dict(lighting_class=lc, crashes=len(o), fatal_share_day_observed=round(o[o.light == 'daylight'].fatal.mean(), 4), fatal_share_dark_observed=round(o[o.light == 'dark'].fatal.mean(), 4),
                    p_fatal_daylight=round(p['daylight'], 4), p_fatal_twilight=round(p['twilight'], 4), p_fatal_dark=round(p['dark'], 4),
                    multiplier_twilight=round(max(p['twilight'] / p['daylight'], 1), 3), multiplier_dark=round(max(p['dark'] / p['daylight'], 1), 3)))
mult = pd.DataFrame(out); mult.to_csv(os.path.join(OUT, 'B2_darkness_multipliers.csv'), index=False); print(mult.to_string())
json.dump(dict(chosen_model=use, comparison=comp.to_dict('records'), severity_crashes_train=len(tr), severity_crashes_test=len(te), runtime_s=round(time.time() - t0)),
          open(os.path.join(OUT, 'B2_settings.json'), 'w'), indent=1)
print(f'Stage B2 done in {time.time() - t0:.0f}s ({use})')
