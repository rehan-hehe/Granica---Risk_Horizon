"""Step 0: one-time conversion of the model-ready tables into readable CSV.
From here on, every modelling script reads ONLY these CSVs (2_readable/10_model_inputs/).

Run from anywhere:  python 00_export_model_inputs.py  [path to RiskHorizon_Dataset]
Writes:
  2_readable/10_model_inputs/where_segments.csv              one row per road segment (features + crash counts). Shareable (aggregated).
  2_readable/10_model_inputs/where_segments_geometry.csv     segment_id + WKT line (lon/lat), for maps.
  2_readable/10_model_inputs/internal_do_not_share/when_crash_hours.csv
                                                             crash + control hours with weather/light, plus the road features of the
                                                             segment each crash happened on. Crash-level: INTERNAL ONLY.
  2_readable/10_model_inputs/model_inputs_dictionary.csv     column meanings.
"""
import os, sys, pandas as pd, shapely

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PQ = os.path.join(ROOT, '1_parquet'); OUT = os.path.join(ROOT, '2_readable', '10_model_inputs'); INT = os.path.join(OUT, 'internal_do_not_share')
os.makedirs(INT, exist_ok=True)

# ---- WHERE: segments. Old model scores are dropped: the new pipeline recomputes them.
S = pd.read_parquet(os.path.join(PQ, '08_WHERE_road_risk', 'where_segments_scored.parquet'))
S = S.drop(columns=['piece', 'pieces', 'spf_expected_ksi_2yr', 'eb_expected_ksi_2yr', 'eb_weight_on_model', 'risk_per_km', 'risk_band'])
first = ['segment_id', 'way_id', 'highway', 'name', 'ref', 'mid_lat', 'mid_lon', 'length_m']
S = S[first + [c for c in S.columns if c not in first]]
S.to_csv(os.path.join(OUT, 'where_segments.csv'), index=False)
print('where_segments.csv', S.shape)

G = pd.concat([pd.read_parquet(os.path.join(PQ, '08_WHERE_road_risk', f'where_segments_geometry_part{i}.parquet')) for i in (1, 2)])
G['wkt_lonlat'] = shapely.to_wkt(shapely.from_wkb(G.geometry_wkb.values), rounding_precision=6)
G[['segment_id', 'wkt_lonlat']].to_csv(os.path.join(OUT, 'where_segments_geometry.csv'), index=False)
print('where_segments_geometry.csv', len(G))

# ---- WHEN: crash + control hours, with the road features of the crash segment (internal link used only here)
T = pd.read_parquet(os.path.join(PQ, '07_WHEN_conditions', 'when_case_crossover.parquet'))
L = pd.read_parquet(os.path.join(PQ, '01_iRAD_CLASSIFIED_do_not_share', 'crash_to_segment_INTERNAL.parquet'), columns=['accident_id', 'segment_id'])
road = S[['segment_id', 'highway', 'is_main_road', 'is_national_highway', 'turn_deg_per_km', 'junctions_per_km', 'poi200_total', 'night_light_nw', 'night_light_class']]
T = T.merge(L.rename(columns={'accident_id': 'stratum_id'}), on='stratum_id', how='left').merge(road, on='segment_id', how='left')
T = T.drop(columns=['metar_station', 'metar_time_utc', 'metar_wx'])
T.to_csv(os.path.join(INT, 'when_crash_hours.csv'), index=False)
print('when_crash_hours.csv', T.shape, '| with segment', round(T.segment_id.notna().mean(), 3))

# ---- raw public layers used for feature engineering (readable copies)
RAW = os.path.join(OUT, 'raw_layers'); os.makedirs(RAW, exist_ok=True)
pd.read_parquet(os.path.join(PQ, '06_public_data', 'osm', 'osm_pois.parquet'))[['lat', 'lon', 'key', 'value', 'name']].to_csv(os.path.join(RAW, 'osm_pois.csv'), index=False)
pd.read_parquet(os.path.join(PQ, '06_public_data', 'osm', 'osm_road_nodes.parquet'))[['lat', 'lon', 'kind']].to_csv(os.path.join(RAW, 'osm_road_nodes.csv'), index=False)
V = pd.read_parquet(os.path.join(PQ, '06_public_data', 'viirs', 'viirs_vnp46a4_2024_assam.parquet'), columns=['lat', 'lon', 'radiance_nw'])
V.round({'lat': 5, 'lon': 5, 'radiance_nw': 2}).to_csv(os.path.join(RAW, 'viirs_night_lights_2024.csv'), index=False)
print('raw layers written', len(V))

# ---- crash events: one row per snapped crash (date, segment, severity). Crash-level: INTERNAL ONLY.
E = T[T.is_case == 1][['stratum_id', 'segment_id', 'dt_ist', 'police_severity', 'latitude', 'longitude']].rename(columns={'stratum_id': 'accident_id'})
E['month'] = pd.to_datetime(E.dt_ist).dt.strftime('%Y-%m')
E['ksi'] = E.police_severity.isin(['Fatal', 'Grievous Injury']).astype(int); E['fatal'] = (E.police_severity == 'Fatal').astype(int)
E.to_csv(os.path.join(INT, 'crash_events.csv'), index=False)
print('crash_events.csv', len(E), '| snapped', int(E.segment_id.notna().sum()))

D = [('segment_id', 'Road segment key (<=500 m piece of an OSM way)'),
     ('highway / is_main_road / is_national_highway', 'OSM road class; main = trunk/primary/secondary (+links); NH from ref'),
     ('length_m, sinuosity, turn_deg_per_km', 'Segment length; path length / straight distance; total turning per km'),
     ('junctions, junctions_per_km', 'Road-graph nodes with >=3 connections within 5 m'),
     ('n_rail_crossing, n_speed_breaker, n_signal, n_ped_crossing, n_bus_stop', 'OSM points within 30 m'),
     ('poi200_*', 'Places within 200 m that draw pedestrians (market/shop, education, health, transit, eatery, worship, fuel)'),
     ('night_light_nw, night_light_class', 'NASA VIIRS 2024 radiance (nW/cm2/sr), 3x3 mean; class dark_rural <0.5, dim <5, lit_town <20, bright_urban'),
     ('crashes_*, ksi_*, fatal_crashes_*, killed_*, ped_crashes_*', 'iRAD counts on the segment; _train = 2023-24, _test = Jan 2025-May 2026; ksi = fatal or grievous crashes'),
     ('blackspot_train / blackspot_test / new_blackspot_test', '>=5 KSI crashes or >=10 killed in the period; new = blackspot in test but not in train'),
     ('nearby_ksi_1km_train, nearby_crashes_1km_train, nearby_road_km_1km', 'KSI / crashes on OTHER segments within 1 km (2023-24); road km within 1 km'),
     ('when_crash_hours: stratum_id, is_case', 'One crash (is_case=1) and its control hours (same place, weekday, clock time, other weeks of the month)'),
     ('when_crash_hours: era5_*, cond_*, metar_*, sun_elevation_deg, light_computed', 'Weather (ERA5 25 km, inferred), condition flags, airport obs, computed sun position'),
     ('when_crash_hours: police_*', 'What the police recorded (crash rows only)')]
pd.DataFrame(D, columns=['column(s)', 'meaning']).to_csv(os.path.join(OUT, 'model_inputs_dictionary.csv'), index=False)
print('done ->', OUT)
