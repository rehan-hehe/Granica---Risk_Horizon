"""Build the WHEN training table: every crash (case) and its same-weekday controls, with the weather,
light and fog conditions at that place and hour.  One row = one place-hour.

Inputs  (Risk_Horizon_Data/):  derived/case_crossover_skeleton.parquet, data/weather_era5_hourly.parquet,
                               data/metar_assam_airports.parquet, scope/metar_stations.csv
Output  (Risk_Horizon_Data/):  model_ready/when_case_crossover.parquet  (+ readable Excel summary)
Every value keeps its origin: police_* = iRAD (case rows only), era5_* = reanalysis (inferred, 25 km),
metar_* = airport observation (observed, only within 30 km of an airport), sun_*/light_computed = computed.
"""
import os, sys, numpy as np, pandas as pd

BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(BASE, 'model_ready'); os.makedirs(OUTD, exist_ok=True)
sk = pd.read_parquet(os.path.join(BASE, 'derived', 'case_crossover_skeleton.parquet'))
wx = pd.read_parquet(os.path.join(BASE, 'data', 'weather_era5_hourly.parquet'))
mt = pd.read_parquet(os.path.join(BASE, 'data', 'metar_assam_airports.parquet'))

# ---------------- ERA5: value at the hour + build-up over the previous 3 hours
wx = wx.sort_values(['cell_id', 'time_utc']).reset_index(drop=True)
g = wx.groupby('cell_id')
full_idx = wx.set_index(['cell_id', 'time_utc'])
def lagged(col, k):  # value k hours earlier (NaN if that hour was not downloaded)
    key = pd.MultiIndex.from_arrays([wx.cell_id, wx.time_utc - pd.Timedelta(hours=k)])
    return full_idx[col].reindex(key).values
wx['precip_prev3h_mm'] = np.nansum(np.vstack([lagged('precipitation', k) for k in (1, 2, 3)]), axis=0)
wx['precip_4h_mm'] = wx.precipitation + wx.precip_prev3h_mm
wx['rh_min_prev3h'] = np.nanmin(np.vstack([lagged('relative_humidity_2m', k) for k in (1, 2, 3)]), axis=0)
wx = wx.rename(columns={'temperature_2m': 'era5_temp_c', 'relative_humidity_2m': 'era5_rh_pct', 'dew_point_2m': 'era5_dewpoint_c',
                        'precipitation': 'era5_precip_mm', 'cloud_cover_low': 'era5_low_cloud_pct', 'wind_speed_10m': 'era5_wind_kmh',
                        'weather_code': 'era5_weather_code', 'precip_prev3h_mm': 'era5_precip_prev3h_mm', 'precip_4h_mm': 'era5_precip_4h_mm',
                        'rh_min_prev3h': 'era5_rh_min_prev3h', 'time_utc': 'utc_hour'})
wx['era5_dewpoint_depression_c'] = (wx.era5_temp_c - wx.era5_dewpoint_c).round(2)
t = sk.merge(wx.drop(columns=['grid_lat', 'grid_lon']), on=['cell_id', 'utc_hour'], how='left', validate='m:1')

# ---------------- METAR: nearest report within ±30 min, only for rows whose crash is within 30 km of that airport
mt = mt.dropna(subset=['time_utc']).sort_values('time_utc')
parts = []
for icao, gg in t[t.nearest_metar.notna()].groupby('nearest_metar'):
    m = mt[mt.icao == icao][['time_utc', 'visibility_m', 'wxcodes', 'relh', 'metar']].rename(columns={'time_utc': 'metar_time_utc'})
    if m.empty: continue
    gg = gg[['dt_utc']].reset_index().sort_values('dt_utc')
    j = pd.merge_asof(gg, m, left_on='dt_utc', right_on='metar_time_utc', direction='nearest', tolerance=pd.Timedelta(minutes=30))
    parts.append(j.set_index('index'))
if parts:
    mj = pd.concat(parts)
    t = t.join(mj[['metar_time_utc', 'visibility_m', 'wxcodes', 'relh']].rename(columns={
        'visibility_m': 'metar_visibility_m', 'wxcodes': 'metar_wx', 'relh': 'metar_rh_pct'}))
else:
    t['metar_time_utc'] = pd.NaT; t['metar_visibility_m'] = np.nan; t['metar_wx'] = None; t['metar_rh_pct'] = np.nan
t['metar_offset_min'] = ((t.metar_time_utc - t.dt_utc).dt.total_seconds() / 60).round(0)
wxs = t.metar_wx.fillna('')
t['metar_fog'] = np.where(t.metar_time_utc.isna(), np.nan, (wxs.str.contains('FG') | (t.metar_visibility_m < 1000)).astype(float))
t['metar_mist_haze'] = np.where(t.metar_time_utc.isna(), np.nan, wxs.str.contains('BR|HZ|DU').astype(float))
t['metar_rain_thunder'] = np.where(t.metar_time_utc.isna(), np.nan, wxs.str.contains('RA|DZ|TS').astype(float))

# ---------------- simple, explainable condition flags (thresholds stated, so a stranger can audit them)
t['cond_rain'] = (t.era5_precip_mm >= 0.5).astype(int)                        # ≥0.5 mm in the hour
t['cond_heavy_rain'] = (t.era5_precip_mm >= 4.0).astype(int)                  # ≥4 mm/h
t['cond_wet_road'] = (t.era5_precip_4h_mm >= 0.5).astype(int)                 # rained at some point in last 4 h
t['cond_fog_likely_era5'] = ((t.era5_rh_pct >= 95) & (t.era5_wind_kmh < 7) & (t.era5_dewpoint_depression_c <= 1)).astype(int)
t['cond_dark'] = (t.light_computed == 'dark').astype(int)
t['cond_twilight'] = (t.light_computed == 'civil_twilight').astype(int)
t['hour_ist'] = t.dt_ist.dt.hour
t['weekday'] = t.dt_ist.dt.day_name()
t['month'] = t.dt_ist.dt.month
t['season'] = t.month.map({12: 'winter', 1: 'winter', 2: 'winter', 3: 'pre-monsoon', 4: 'pre-monsoon', 5: 'pre-monsoon',
                           6: 'monsoon', 7: 'monsoon', 8: 'monsoon', 9: 'monsoon', 10: 'post-monsoon', 11: 'post-monsoon'})
t['has_weather'] = t.era5_temp_c.notna().astype(int)
t['has_metar'] = t.metar_time_utc.notna().astype(int)
t = t.rename(columns={'severity': 'police_severity', 'light_condition': 'police_light', 'weather_condition': 'police_weather',
                      'visibility_m': 'police_visibility_m', 'nearest_metar': 'metar_station', 'metar_km': 'metar_station_km'})
cols = ['stratum_id', 'is_case', 'offset_days', 'dt_ist', 'dt_utc', 'utc_hour', 'time_precision', 'latitude', 'longitude', 'cell_id',
        'hour_ist', 'weekday', 'month', 'season',
        'sun_elevation_deg', 'light_computed', 'cond_dark', 'cond_twilight',
        'era5_temp_c', 'era5_rh_pct', 'era5_dewpoint_c', 'era5_dewpoint_depression_c', 'era5_precip_mm', 'era5_precip_prev3h_mm',
        'era5_precip_4h_mm', 'era5_rh_min_prev3h', 'era5_low_cloud_pct', 'era5_wind_kmh', 'era5_weather_code',
        'cond_rain', 'cond_heavy_rain', 'cond_wet_road', 'cond_fog_likely_era5',
        'metar_station', 'metar_station_km', 'metar_time_utc', 'metar_offset_min', 'metar_visibility_m', 'metar_wx', 'metar_rh_pct',
        'metar_fog', 'metar_mist_haze', 'metar_rain_thunder',
        'police_severity', 'police_light', 'police_weather', 'police_visibility_m', 'has_weather', 'has_metar']
t = t[cols].sort_values(['stratum_id', 'offset_days']).reset_index(drop=True)
t.to_parquet(os.path.join(OUTD, 'when_case_crossover.parquet'), index=False)
print(f'rows {len(t):,} | strata {t.stratum_id.nunique():,} | weather coverage {t.has_weather.mean():.2%} | METAR coverage {t.has_metar.mean():.2%}')
