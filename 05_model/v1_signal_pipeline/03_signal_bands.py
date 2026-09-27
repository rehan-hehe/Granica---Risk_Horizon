"""Stage C — combine WHERE x WHEN into the signal a driver-assist car receives.

risk_now(segment, conditions) = static risk (Stage A, expected KSI crashes per km per year)
                                x darkness multiplier for the segment's lighting class (Stage B2)
                                x weather multiplier (Stage B1; 1.0 unless evidence says it raises risk)
Bands use fixed cut-offs set on the static risk by share of ROAD LENGTH:
    3 very high = top 1% of road km | 2 high = next 4% | 1 elevated = next 15% | 0 low = the rest
The same cut-offs apply at night, so darkness can move a segment up a band.
Band validation uses the test model only (fit on 2023-24, counted on 2025-26).

Inputs : 2_readable/11_model_results/A_static_risk/A_static_risk_scores.csv, B_conditions/B2_darkness_multipliers.csv,
         B_conditions/B1_weather_multipliers_for_signal.csv, 2_readable/10_model_inputs/where_segments.csv
Output : 2_readable/11_model_results/C_signal/*
"""
import os, sys, json, numpy as np, pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
R = os.path.join(ROOT, '2_readable', '11_model_results'); OUT = os.path.join(R, 'C_signal'); os.makedirs(OUT, exist_ok=True)
A = pd.read_csv(os.path.join(R, 'A_static_risk', 'A_static_risk_scores.csv'), low_memory=False)
F = pd.read_csv(os.path.join(ROOT, '2_readable', '10_model_inputs', 'where_segments.csv'), low_memory=False,
                usecols=['segment_id', 'is_main_road', 'poi200_education', 'poi200_market_shop', 'poi200_transit', 'n_bus_stop', 'n_ped_crossing'])
A = A.merge(F, on='segment_id', how='left')
M = pd.read_csv(os.path.join(R, 'B_conditions', 'B2_darkness_multipliers.csv')).set_index('lighting_class')
W = pd.read_csv(os.path.join(R, 'B_conditions', 'B1_weather_multipliers_for_signal.csv')).set_index('condition').signal_multiplier

def cutoffs(score, length):
    o = np.argsort(-score); cum = np.cumsum(length[o]) / length.sum()
    return {3: score[o][np.searchsorted(cum, 0.01)], 2: score[o][np.searchsorted(cum, 0.05)], 1: score[o][np.searchsorted(cum, 0.20)]}
def band(score, cut):
    return np.select([score >= cut[3], score >= cut[2], score >= cut[1]], [3, 2, 1], 0)
L = A.length_m.values

# ---------------- validation of the banding with the TEST model (2023-24 only -> 2025-26 crashes)
tm = A.eb_expected_ksi_2yr_test_model.values / np.maximum(L / 1000, 0.01)
bt = band(tm, cutoffs(tm, L))
val = pd.DataFrame({'band': bt, 'km': L / 1000, 'ksi_test': A.ksi_test, 'new_bs': A.new_blackspot_test}).groupby('band').agg(
    road_km=('km', 'sum'), test_KSI=('ksi_test', 'sum'), new_blackspots=('new_bs', 'sum'))
val['share_road_km'] = val.road_km / val.road_km.sum(); val['share_test_KSI'] = val.test_KSI / val.test_KSI.sum()
val['KSI_per_100km'] = 100 * val.test_KSI / val.road_km
val = val.round(4).sort_index(ascending=False); val.index = val.index.map({3: '3 very high', 2: '2 high', 1: '1 elevated', 0: '0 low'})
val.to_csv(os.path.join(OUT, 'C_band_validation_2025_26.csv'))
print(val.to_string())

# ---------------- deployment signal
r = A.deploy_expected_ksi_per_km_year.values
cut = cutoffs(r, L)
mdark = A.night_light_class.map(M.multiplier_dark).fillna(M.multiplier_dark.mean()).values
mtw = A.night_light_class.map(M.multiplier_twilight).fillna(M.multiplier_twilight.mean()).values
wx = max(W.get('cond_rain', 1.0), W.get('cond_fog_likely_era5', 1.0), 1.0)
ped = ((A.poi200_education + A.poi200_market_shop + A.poi200_transit + A.n_bus_stop + A.n_ped_crossing) >= 3)
has_history = (A.ksi_train + A.ksi_test) > 0
conf = np.where(has_history & (A.is_main_road == 1), 'high', np.where(has_history | (A.is_main_road == 1), 'medium', 'low'))
sig = pd.DataFrame({'segment_id': A.segment_id, 'name': A.name, 'ref': A.ref, 'highway': A.highway, 'mid_lat': A.mid_lat, 'mid_lon': A.mid_lon,
                    'length_m': A.length_m, 'static_risk_ksi_per_km_yr': r.round(4),
                    'static_band': band(r, cut), 'twilight_band': band(r * mtw, cut), 'dark_band': band(r * mdark, cut),
                    'rain_band': band(r * wx, cut), 'dark_multiplier': mdark.round(3), 'main_reasons': A.main_reasons,
                    'ped_flag': ped.astype(int), 'confidence': conf, 'night_light_class': A.night_light_class})
sig.to_csv(os.path.join(OUT, 'C_segment_signal_table.csv'), index=False)

# alert load: how much road is flagged high+ by day vs in the dark
load = pd.DataFrame([dict(condition=c, band_2plus_road_km=round(L[sig[col] >= 2].sum() / 1000), band_2plus_share_km=round(L[sig[col] >= 2].sum() / L.sum(), 4),
                          band_3_road_km=round(L[sig[col] == 3].sum() / 1000))
                     for c, col in [('daylight', 'static_band'), ('twilight', 'twilight_band'), ('dark', 'dark_band'), ('rain', 'rain_band')]])
load.to_csv(os.path.join(OUT, 'C_alert_load_by_condition.csv'), index=False)
print(load.to_string())

# band definitions / actions (what the car does)
pd.DataFrame([
    (3, 'very high', 'top 1% of road length', f'>= {cut[3]:.3f}', 'Chime + spoken warning ~500 m before; ACC slows before the segment; max pedestrian sensitivity'),
    (2, 'high', 'next 4% of road length', f'>= {cut[2]:.3f}', 'Chime + icon ~300 m before; cap speed near limit; larger following gap; pedestrian sensitivity up if ped_flag'),
    (1, 'elevated', 'next 15% of road length', f'>= {cut[1]:.3f}', 'No alert (logged); slightly larger following gap'),
    (0, 'low', 'remaining 80%', f'< {cut[1]:.3f}', 'Normal driving')],
    columns=['band', 'label', 'defined_as', 'risk_threshold_ksi_per_km_yr', 'suggested_vehicle_action']).to_csv(os.path.join(OUT, 'C_band_definitions.csv'), index=False)
json.dump(dict(cutoffs_ksi_per_km_yr={str(k): round(float(v), 4) for k, v in cut.items()}, weather_multiplier_used=wx,
               dark_multiplier_by_lighting=M.multiplier_dark.to_dict(), ped_flag_rule='>=3 of schools/markets/transit places within 200 m + bus stops + ped crossings within 30 m',
               confidence_rule='high = crash history and main road; medium = one of them; low = neither'),
          open(os.path.join(OUT, 'C_signal_settings.json'), 'w'), indent=1)
print('Stage C done; bands static', np.bincount(sig.static_band), 'dark', np.bincount(sig.dark_band))
