"""Readable companion for model_ready/when_case_crossover.parquet -> model_ready/WHEN_dataset_readable.xlsx"""
import os, sys, numpy as np, pandas as pd
BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
t = pd.read_parquet(os.path.join(BASE, 'model_ready', 'when_case_crossover.parquet'))
cs = t[t.is_case == 1]

def mh(col):
    d = t.dropna(subset=[col])
    g = d.groupby('stratum_id').agg(n=('is_case', 'size'), cases=('is_case', 'sum'), exp_total=(col, 'sum'))
    g = g.join(d[d.is_case == 1].set_index('stratum_id')[col].rename('a')).dropna()
    g = g[(g.cases == 1) & (g.n >= 2)]
    c = g.exp_total - g.a; m = g.n - 1
    R = g.a * (m - c) / g.n; S = (1 - g.a) * c / g.n
    orr = R.sum() / S.sum()
    # Robins-Breslow-Greenland variance for matched sets
    P = (g.a + (m - c)) / g.n; Q = ((1 - g.a) + c) / g.n
    var = (P * R).sum() / (2 * R.sum() ** 2) + ((P * S).sum() + (Q * R).sum()) / (2 * R.sum() * S.sum()) + (Q * S).sum() / (2 * S.sum() ** 2)
    lo, hi = np.exp(np.log(orr) - 1.96 * np.sqrt(var)), np.exp(np.log(orr) + 1.96 * np.sqrt(var))
    return round(orr, 3), round(lo, 3), round(hi, 3), int(len(g)), int(g.a.sum())

conds = [('cond_rain', 'Raining ≥0.5 mm in the hour (ERA5)'), ('cond_heavy_rain', 'Heavy rain ≥4 mm/h (ERA5)'),
         ('cond_wet_road', 'Rain in the last 4 h (ERA5)'), ('cond_fog_likely_era5', 'Fog likely: RH≥95%, wind<7 km/h, T−Td≤1°C (ERA5)'),
         ('metar_fog', 'Fog observed at airport (FG or visibility <1 km)'), ('metar_mist_haze', 'Mist/haze/dust observed at airport'),
         ('metar_rain_thunder', 'Rain/drizzle/thunder observed at airport')]
orr = pd.DataFrame([dict(condition=lbl, column=c, **dict(zip(['matched_odds_ratio', 'ci95_low', 'ci95_high', 'crash_sets_used', 'crashes_in_condition'], mh(c))))
                    for c, lbl in conds])
orr['reading'] = np.where(orr.ci95_high < 1, 'crashes LESS likely in this condition (same place & clock time)',
                  np.where(orr.ci95_low > 1, 'crashes MORE likely in this condition', 'no clear change'))
sev = []
for c, lbl in conds + [('cond_dark', 'Dark (sun below −6°, computed)')]:
    x = cs.dropna(subset=[c])
    for v in (0, 1):
        s = x[x[c] == v]
        sev.append(dict(condition=lbl, in_condition=bool(v), crashes=len(s), fatal_share=round((s.police_severity == 'Fatal').mean(), 4),
                        fatal_or_grievous_share=round(s.police_severity.isin(['Fatal', 'Grievous Injury']).mean(), 4)))
sev = pd.DataFrame(sev)
pw = cs.police_weather.fillna('')
agree = pd.DataFrame([
    dict(check='Police wrote "Rain" → ERA5 shows rain in last 4 h', police_yes=round(cs[pw.str.contains('Rain')].cond_wet_road.mean(), 3),
         police_no=round(cs[~pw.str.contains('Rain')].cond_wet_road.mean(), 3), n_police_yes=int(pw.str.contains('Rain').sum())),
    dict(check='Police wrote "Mist/Fog" → ERA5 fog-likely', police_yes=round(cs[pw.str.contains('Fog')].cond_fog_likely_era5.mean(), 3),
         police_no=round(cs[~pw.str.contains('Fog')].cond_fog_likely_era5.mean(), 3), n_police_yes=int(pw.str.contains('Fog').sum())),
    dict(check='Police wrote "Mist/Fog" → airport saw fog (Guwahati only)', police_yes=round(cs[pw.str.contains('Fog')].metar_fog.mean(), 3),
         police_no=round(cs[~pw.str.contains('Fog')].metar_fog.mean(), 3), n_police_yes=int((pw.str.contains('Fog') & cs.metar_fog.notna()).sum())),
    dict(check='Police light condition = computed sun position', police_yes=None, police_no=None, n_police_yes=None)])
m = {'Day': 'daylight', 'Night': 'dark', 'Darkness with No street light': 'dark', 'Darkness with street lights on': 'dark',
     'Darkness with Poor street light': 'dark', 'Twilight': 'civil_twilight', 'Dawn': 'civil_twilight'}
pl = cs.police_light.map(m); ok = pl.notna()
agree.loc[3, 'police_yes'] = round((pl[ok] == cs.light_computed[ok]).mean(), 3); agree.loc[3, 'n_police_yes'] = int(ok.sum())
cov = pd.DataFrame([('Rows (place-hours)', f'{len(t):,}'), ('Crashes (cases)', f'{int(t.is_case.sum()):,}'), ('Controls', f'{int((t.is_case == 0).sum()):,}'),
                    ('Controls per crash (avg)', round((t.is_case == 0).sum() / t.is_case.sum(), 2)), ('Rows with ERA5 weather', f'{t.has_weather.mean():.2%}'),
                    ('Rows with airport observation', f'{t.has_metar.mean():.2%} (only Guwahati VEGT downloaded so far; 5 more airports pending)'),
                    ('Crash time window', f'{cs.dt_ist.min()} → {cs.dt_ist.max()}'), ('Crash times rounded to :00/:30', f'{(cs.time_precision != "minute").mean():.1%}')],
                   columns=['item', 'value'])
caseref = t.groupby('is_case')[['cond_rain', 'cond_heavy_rain', 'cond_wet_road', 'cond_fog_likely_era5', 'cond_dark', 'metar_fog', 'metar_mist_haze',
                                'era5_rh_pct', 'era5_precip_mm', 'era5_temp_c', 'era5_wind_kmh']].mean().round(4).T.rename(columns={0: 'controls (same place, other days)', 1: 'crashes'})
season = cs.groupby('season').agg(crashes=('stratum_id', 'size'), rain_share=('cond_rain', 'mean'), fog_likely_share=('cond_fog_likely_era5', 'mean'),
                                  dark_share=('cond_dark', 'mean'), fatal_share=('police_severity', lambda s: (s == 'Fatal').mean())).round(3).reset_index()
DICT = pd.DataFrame([
    ('stratum_id', 'iRAD accident id. The crash row and its control rows share it.', 'iRAD'),
    ('is_case', '1 = the crash hour; 0 = same place, same weekday & clock time, other weeks of the same month (no crash)', 'design'),
    ('offset_days', 'Days from the crash (0, ±7, ±14, ±21, ±28 within the same month)', 'design'),
    ('dt_ist / dt_utc / utc_hour', 'Time in IST, UTC, and the UTC hour used to join weather', 'iRAD'),
    ('time_precision', 'minute, or rounded_to_30min when the police time ends in :00/:30', 'iRAD'),
    ('latitude / longitude / cell_id', 'Crash location; cell_id = the 25 km ERA5 weather cell', 'iRAD / design'),
    ('sun_elevation_deg, light_computed', 'Sun angle from time & place; dark < −6°, civil_twilight −6..0°, daylight > 0°', 'computed'),
    ('era5_*', 'Hourly reanalysis weather for the 25 km cell (temperature, humidity, dew point, precipitation, low cloud, wind km/h, WMO code). Inferred, not a station reading. era5_weather_code over-reports drizzle (51); use era5_precip_mm instead', 'Open-Meteo ERA5'),
    ('era5_precip_prev3h_mm / era5_precip_4h_mm / era5_rh_min_prev3h', 'Build-up: rain in the 3 h before, total over 4 h, lowest humidity in 3 h before', 'computed from ERA5'),
    ('cond_*', 'Yes/no flags with the thresholds written in their names/Summary; cond_dark from sun position', 'computed'),
    ('metar_*', 'Airport observation nearest in time (±30 min), only for crashes within 30 km of the airport', 'Iowa Mesonet METAR'),
    ('metar_fog / metar_mist_haze / metar_rain_thunder', 'From the METAR weather codes (FG; BR/HZ/DU; RA/DZ/TS) or visibility <1 km', 'computed from METAR'),
    ('police_*', 'What the police report recorded (crash rows only; blank on controls)', 'iRAD'),
    ('has_weather / has_metar', 'Coverage flags', 'computed')], columns=['column', 'meaning', 'source'])
README = pd.DataFrame({'WHEN dataset — how to read it': [
    'One row = one place-hour. Each crash (is_case=1) is paired with the SAME place at the SAME weekday and clock time on other weeks of the same month (is_case=0).',
    'Comparing the crash hour with its own controls cancels out everything fixed about the place and time of day — what is left is the effect of the conditions (rain, fog...).',
    'This table is the training data for the "condition multiplier" of Risk Horizon. The WHERE table (road segments) comes next.',
    'Sheets: Coverage · Crash_vs_controls · Condition_effects (matched odds ratios) · Severity_by_condition · Police_vs_measured · Season · Data_dictionary · Sample_rows (first 3,000).',
    'Full data: when_case_crossover.parquet (same folder).',
    'Honesty notes: weather is 25 km reanalysis; airport fog only near airports; police times often rounded; darkness cannot be tested this way (controls are at the same clock time), it is tested through severity instead.']})
out = os.path.join(BASE, 'model_ready', 'WHEN_dataset_readable.xlsx')
with pd.ExcelWriter(out) as w:
    for name, df in [('README', README), ('Coverage', cov), ('Crash_vs_controls', caseref.reset_index().rename(columns={'index': 'measure'})),
                     ('Condition_effects', orr), ('Severity_by_condition', sev), ('Police_vs_measured', agree), ('Season', season),
                     ('Data_dictionary', DICT), ('Sample_rows', t.head(3000))]:
        df.to_excel(w, sheet_name=name, index=False)
from openpyxl import load_workbook
wb = load_workbook(out)
for ws in wb.worksheets:
    ws.freeze_panes = 'A2'
    for col in ws.columns:
        L = max(len(str(c.value)) if c.value is not None else 0 for c in col[:300])
        ws.column_dimensions[col[0].column_letter].width = min(max(10, L + 2), 100)
wb.save(out)
print(orr.to_string()); print(sev.to_string()); print(agree.to_string()); print(season.to_string())
