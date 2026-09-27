"""Step 1 — decide exactly what public data we need, from the iRAD crash footprint.
Outputs (scope/): what every downloader is allowed to fetch, and nothing more.
Also builds datasets that need no download: case-crossover skeleton + sun position.
"""
import json, math, os
import numpy as np, pandas as pd, pvlib, mercantile

IRAD = os.environ.get('IRAD_DIR', '/home/claude/eda/data') + '/iRAD_Assam_all__Accidents.parquet'
OUT = 'scope'; DER = 'derived'
os.makedirs(OUT, exist_ok=True); os.makedirs(DER, exist_ok=True)

a = pd.read_parquet(IRAD, columns=['accident_id', 'accident_datetime', 'latitude', 'longitude', 'severity', 'district_name',
                                   'station_name', 'light_condition', 'weather_condition', 'visibility_m', 'road_classification'])
n0 = len(a)
a = a[a.latitude.between(24.0, 28.3) & a.longitude.between(89.6, 96.2) & a.accident_datetime.notna()].copy()
a['dt_ist'] = pd.to_datetime(a.accident_datetime)
a['time_precision'] = np.where(a.dt_ist.dt.minute.isin([0, 30]), 'rounded_to_30min', 'minute')
print(f'crashes usable: {len(a)} / {n0}')

# ---------------- ERA5 0.25° cells (Open-Meteo archive, model=era5)
a['era5_lat'] = (np.round(a.latitude * 4) / 4).round(2)
a['era5_lon'] = (np.round(a.longitude * 4) / 4).round(2)
cells = (a.groupby(['era5_lat', 'era5_lon']).agg(n_crashes=('accident_id', 'size'), n_fatal=('severity', lambda s: (s == 'Fatal').sum()))
         .reset_index().sort_values('n_crashes', ascending=False))
cells['priority'] = range(1, len(cells) + 1)
cells['cum_share'] = (cells.n_crashes.cumsum() / cells.n_crashes.sum()).round(4)
cells['cell_id'] = 'E' + cells.priority.astype(str).str.zfill(3)
cells.to_csv(f'{OUT}/era5_cells.csv', index=False)
a = a.merge(cells[['era5_lat', 'era5_lon', 'cell_id']], on=['era5_lat', 'era5_lon'])
WSTART, WEND = '2022-12-01', '2026-06-30'  # crash window ± 4 weeks for controls
days = (pd.Timestamp(WEND) - pd.Timestamp(WSTART)).days + 1
calls_per_cell = days / 14 * 1.0  # 7 variables (<10) → weight = weeks/2
print(f'ERA5 cells: {len(cells)}; 95% of crashes in {int((cells.cum_share < 0.95).sum()) + 1}; est. Open-Meteo weight {calls_per_cell * len(cells):.0f} calls')

# ---------------- METAR stations: keep only those near crashes
AIRPORTS = {'VEGT': ('Guwahati (LGBI)', 26.1061, 91.5859), 'VEMN': ('Dibrugarh (Mohanbari)', 27.4839, 95.0169),
            'VEJT': ('Jorhat (Rowriah)', 26.7315, 94.1755), 'VEKU': ('Silchar (Kumbhirgram)', 24.9129, 92.9787),
            'VETZ': ('Tezpur (Salonibari)', 26.7091, 92.7847), 'VELR': ('North Lakhimpur (Lilabari)', 27.2955, 94.0976)}
def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    d = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(d))
dist = pd.DataFrame({k: hav(a.latitude.values, a.longitude.values, v[1], v[2]) for k, v in AIRPORTS.items()}, index=a.index)
a['nearest_metar'] = dist.idxmin(axis=1); a['metar_km'] = dist.min(axis=1).round(1)
a.loc[a.metar_km > 30, 'nearest_metar'] = None
st = []
for k, (nm, la, lo) in AIRPORTS.items():
    n = int((a.nearest_metar == k).sum())
    st.append(dict(icao=k, name=nm, lat=la, lon=lo, crashes_within_30km=n, share=round(n / len(a), 4), keep=n >= 200))
st = pd.DataFrame(st).sort_values('crashes_within_30km', ascending=False)
st.to_csv(f'{OUT}/metar_stations.csv', index=False)
print(st.to_string(index=False))

# ---------------- OSM / VIIRS extent = crash bounding box + 5 km
bbox = dict(south=round(a.latitude.min() - 0.05, 3), north=round(a.latitude.max() + 0.05, 3),
            west=round(a.longitude.min() - 0.05, 3), east=round(a.longitude.max() + 0.05, 3))
def bm_tile(lat, lon): return f'h{int((lon + 180) // 10):02d}v{int((90 - lat) // 10):02d}'
bm_tiles = sorted({bm_tile(la, lo) for la in (bbox['south'], bbox['north']) for lo in (bbox['west'], bbox['east'])})

# ---------------- Live corridor (TomTom): the three micro-sites from Methodology-and-Workflow
STATIONS = {  # micro-site: police stations whose hotspot cells define it
    'NH27_Jalukbari_Saraighat_Amingaon': ['JALUKBARI OP', 'JALUKBARI', 'NORTH GUWAHATI'],
    'GS_Road_Dispur_Khanapara': ['DISPUR P.S', 'BASISTHA', 'KHANAPARA OP', 'JORABAT OP', 'HATIGAON PS', 'GEETA NAGAR'],
    'NH27_Changsari_Baihata': ['CHANGSARI PS', 'BAIHATA CHARIALI']}
h = pd.read_csv('/home/claude/eda/region_outputs/corridor_hotspots_100m.csv')
rows = []
for site, stns in STATIONS.items():
    hs = h[h.station.isin(stns)]
    for r in hs.itertuples():
        t = mercantile.tile(r.longitude, r.latitude, 13)
        rows.append(dict(site=site, z=13, x=t.x, y=t.y, n=r.n))
tl = pd.DataFrame(rows).groupby(['z', 'x', 'y']).agg(n_crashes=('n', 'sum'), sites=('site', lambda s: ';'.join(sorted(set(s))))).reset_index()
tl = tl.sort_values('n_crashes', ascending=False); tl['cum_share'] = (tl.n_crashes.cumsum() / tl.n_crashes.sum()).round(3)
tl = tl[tl.cum_share.shift(fill_value=0) < 0.97]  # tiles holding 97% of micro-site hotspot crashes
tl.to_csv(f'{OUT}/tomtom_flow_tiles_z13.csv', index=False)
pts = []
for site, stns in STATIONS.items():
    hs = h[h.station.isin(stns)].sort_values('n', ascending=False).head(8)
    for r in hs.itertuples():
        pts.append(dict(site=site, point_id=f'{site[:6]}_{len(pts)+1:02d}', lat=r.latitude, lon=r.longitude, crashes_100m=r.n, fatal=r.fatal, station=r.station))
pts = pd.DataFrame(pts); pts.to_csv(f'{OUT}/tomtom_segment_points.csv', index=False)
polls = 48 * 4
budget = dict(tiles=len(tl), tile_requests_48h=len(tl) * polls, tile_free_per_month=200000,
              points=len(pts), point_requests_48h=len(pts) * 48, segment_free_per_month=20000)
print('TomTom budget', budget)

scope = dict(generated=pd.Timestamp.now().isoformat(timespec='seconds'), crashes_used=len(a), crash_window=[str(a.dt_ist.min()), str(a.dt_ist.max())],
             weather_window=[WSTART, WEND], era5_cells=len(cells), openmeteo_weight_est=round(calls_per_cell * len(cells)),
             openmeteo_variables=['temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation', 'cloud_cover_low', 'wind_speed_10m', 'weather_code'],
             metar_keep=st[st.keep].icao.tolist(), osm_bbox=bbox, osm_source='https://download.geofabrik.de/asia/india/north-eastern-zone-latest.osm.pbf (104 MB, filtered to Assam + needed tags)',
             viirs_product='VNP46A4 (annual, 15 arc-sec) year 2024', viirs_tiles=bm_tiles, tomtom=budget)
json.dump(scope, open(f'{OUT}/scope.json', 'w'), indent=2)

# ---------------- Case-crossover skeleton (no download): TIME-STRATIFIED design
# controls = the same weekday, same clock time, in the SAME calendar month as the crash (3-4 per crash, standard design)
base = a[['accident_id', 'dt_ist', 'latitude', 'longitude', 'cell_id', 'nearest_metar', 'metar_km', 'severity', 'light_condition',
          'weather_condition', 'visibility_m', 'time_precision']].copy()
parts = []
for wk in range(-4, 5):
    t = base.dt_ist + pd.Timedelta(days=7 * wk)
    keep = (t.dt.month == base.dt_ist.dt.month) & (t.dt.year == base.dt_ist.dt.year)
    parts.append(base[keep].assign(dt_ist=t[keep], offset_days=7 * wk, is_case=int(wk == 0)))
cc = pd.concat(parts, ignore_index=True).sort_values(['accident_id', 'offset_days']).reset_index(drop=True)
for c in ['severity', 'light_condition', 'weather_condition', 'visibility_m']:  # police-recorded fields only meaningful for the case row
    cc.loc[cc.is_case == 0, c] = None
cc['dt_utc'] = (cc.dt_ist - pd.Timedelta(hours=5, minutes=30))
cc['utc_hour'] = cc.dt_utc.dt.floor('h')
sp = pvlib.solarposition.get_solarposition(pd.DatetimeIndex(cc.dt_utc).tz_localize('UTC'), cc.latitude.values, cc.longitude.values, method='nrel_numpy')
cc['sun_elevation_deg'] = sp['apparent_elevation'].values.round(2)
cc['light_computed'] = pd.cut(cc.sun_elevation_deg, [-91, -6, 0, 91], labels=['dark', 'civil_twilight', 'daylight']).astype(str)
cc = cc.rename(columns={'accident_id': 'stratum_id'})
cc.to_parquet(f'{DER}/case_crossover_skeleton.parquet', index=False)
print('case-crossover rows', len(cc), 'strata', cc.stratum_id.nunique(), 'controls per crash', round((len(cc) / cc.stratum_id.nunique()) - 1, 2))

cs = cc[cc.is_case == 1]
m = {'Day': 'daylight', 'Night': 'dark', 'Darkness with No street light': 'dark', 'Darkness with street lights on': 'dark',
     'Darkness with Poor street light': 'dark', 'Twilight': 'civil_twilight', 'Dawn': 'civil_twilight'}
cs = cs.assign(police=cs.light_condition.map(m))
pd.crosstab(cs.police, cs.light_computed).to_csv(f'{DER}/light_police_vs_computed.csv')
ex = cs.dropna(subset=['police'])
print('police light vs computed agreement:', round((ex.police == ex.light_computed).mean(), 3))

# ---------------- Exactly which weather hours we need: each case/control hour + the 3 hours before it (rain/fog build-up)
LAG = 3
need = pd.concat([cc[['cell_id', 'utc_hour']].assign(utc_hour=cc.utc_hour - pd.Timedelta(hours=k)) for k in range(LAG + 1)]).drop_duplicates()
need.to_parquet(f'{OUT}/weather_needed_hours.parquet', index=False)
mh = cc.dropna(subset=['nearest_metar'])[['nearest_metar', 'utc_hour']].rename(columns={'nearest_metar': 'icao'})
mneed = pd.concat([mh.assign(utc_hour=mh.utc_hour - pd.Timedelta(hours=k)) for k in range(LAG + 1)]).drop_duplicates()
mneed.to_parquet(f'{OUT}/metar_needed_hours.parquet', index=False)

# ---------------- Request plan: the API only serves continuous date ranges and bills per location per started 2 weeks,
# so group each cell's needed DATES into runs; merge runs only when merging does not cost more calls.
plan = []
need['date'] = need.utc_hour.dt.normalize()
cellix = cells.set_index('cell_id')
for cid, g in need.groupby('cell_id'):
    ds = sorted(g.date.unique())
    runs = [[ds[0], ds[0]]]
    for d in ds[1:]:
        s0, e0 = runs[-1]
        merged_w = max(1, ((d - s0).days + 1) / 14); sep_w = max(1, ((e0 - s0).days + 1) / 14) + 1
        if merged_w <= sep_w: runs[-1][1] = d
        else: runs.append([d, d])
    c = cellix.loc[cid]
    for s0, e0 in runs:
        nd = (e0 - s0).days + 1
        plan.append(dict(cell_id=cid, lat=c.era5_lat, lon=c.era5_lon, start=pd.Timestamp(s0).date().isoformat(), end=pd.Timestamp(e0).date().isoformat(),
                         days=nd, est_weight=round(max(1.0, nd / 14), 2), priority=int(c.priority)))
plan = pd.DataFrame(plan).sort_values(['priority', 'start'])
plan.to_csv(f'{OUT}/weather_requests.csv', index=False)
print(f'weather: needed cell-hours {len(need):,} (of {len(cells) * days * 24:,} if we took every hour); '
      f'{len(plan)} requests, est. weight {plan.est_weight.sum():.0f}')
print(f'metar: needed station-hours {len(mneed):,}')
scope.update(case_crossover='time-stratified: same weekday & time, same calendar month', weather_lag_hours=LAG,
             weather_needed_cell_hours=len(need), metar_needed_station_hours=len(mneed),
             weather_requests=len(plan), openmeteo_weight_planned=round(plan.est_weight.sum()))
scope.pop('openmeteo_weight_est', None)
json.dump(scope, open(f'{OUT}/scope.json', 'w'), indent=2)
