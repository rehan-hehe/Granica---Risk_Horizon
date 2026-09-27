"""Stage B1 — WHEN: do current conditions make a crash more likely? (severity is Stage B2, 04_severity_multipliers.py)

B1  Crash likelihood: time-stratified case-crossover + conditional logistic regression
    (the standard design for weather/crash studies). Each crash is compared with the same place at the
    same weekday and clock time on other weeks of the month, so the place and time of day cancel out.
    Adds rain x main-road interaction (other studies find weather effects differ by road type).
B2  Crash severity: logistic regression of fatal vs non-fatal crashes on light (computed sun position),
    street lighting (VIIRS night-light class), road class and weather. Fit on 2023-24, checked on 2025-26.
    Gives the darkness multiplier per lighting class used in the signal.
Safety rule for the signal: a condition can raise risk, never lower it (a multiplier below 1 is set to 1).

Input : 2_readable/10_model_inputs/internal_do_not_share/when_crash_hours.csv   (crash-level, internal)
Output: 2_readable/11_model_results/B_conditions/*   (aggregates only: odds ratios, multipliers)
"""
import os, sys, json, time, numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit
from sklearn.metrics import roc_auc_score

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
IN = os.path.join(ROOT, '2_readable', '10_model_inputs', 'internal_do_not_share', 'when_crash_hours.csv')
OUT = os.path.join(ROOT, '2_readable', '11_model_results', 'B_conditions'); os.makedirs(OUT, exist_ok=True)
t0 = time.time()
T = pd.read_csv(IN, low_memory=False)
T['period'] = np.where(pd.to_datetime(T.dt_ist).dt.year <= 2024, 'train_2023_24', 'test_2025_26')

# ---------------- B1 conditional logit (all complete strata; main-road flag comes from the crash segment)
d = T.dropna(subset=['cond_rain', 'cond_wet_road', 'cond_fog_likely_era5', 'cond_heavy_rain']).copy()
d['main_road'] = d.groupby('stratum_id').is_main_road.transform('max').fillna(0)
d['rain_x_main_road'] = d.cond_rain * d.main_road
def clogit(df, cols, tag):
    ok = df.groupby('stratum_id').is_case.transform('sum').eq(1) & df.groupby('stratum_id').is_case.transform('size').ge(2)
    df = df[ok]
    r = ConditionalLogit(df.is_case.values, df[cols].astype(float).values, groups=df.stratum_id.values).fit(disp=0)
    ci = r.conf_int()
    return pd.DataFrame({'model': tag, 'term': cols, 'odds_ratio': np.exp(r.params), 'ci95_low': np.exp(ci[:, 0]), 'ci95_high': np.exp(ci[:, 1]),
                         'p_value': r.pvalues, 'crash_sets': df.stratum_id.nunique()})
b1 = pd.concat([clogit(d, ['cond_rain', 'cond_fog_likely_era5'], 'B1a all periods'),
                clogit(d, ['cond_wet_road', 'cond_fog_likely_era5'], 'B1b wet road, all periods'),
                clogit(d, ['cond_rain', 'rain_x_main_road', 'cond_fog_likely_era5'], 'B1c rain x main road'),
                clogit(d[d.period == 'train_2023_24'], ['cond_rain', 'cond_fog_likely_era5'], 'B1a 2023-24 only'),
                clogit(d[d.period == 'test_2025_26'], ['cond_rain', 'cond_fog_likely_era5'], 'B1a 2025-26 only')]).round(4)
b1['reading'] = np.where(b1.ci95_high < 1, 'fewer crashes', np.where(b1.ci95_low > 1, 'more crashes', 'no clear change'))
b1.to_csv(os.path.join(OUT, 'B1_crash_likelihood_odds_ratios.csv'), index=False)
print(b1.to_string()); print(f'B1 done {time.time() - t0:.0f}s')

w = b1[b1.model == 'B1a all periods'].set_index('term')
wm = pd.DataFrame([dict(condition=t, odds_ratio=w.loc[t, 'odds_ratio'], ci95_low=w.loc[t, 'ci95_low'], ci95_high=w.loc[t, 'ci95_high'],
                        signal_multiplier=round(w.loc[t, 'odds_ratio'], 3) if w.loc[t, 'ci95_low'] > 1 else 1.0) for t in ['cond_rain', 'cond_fog_likely_era5']])
wm.to_csv(os.path.join(OUT, 'B1_weather_multipliers_for_signal.csv'), index=False)
print(f'Stage B1 done in {time.time() - t0:.0f}s')
