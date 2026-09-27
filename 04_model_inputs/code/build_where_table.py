"""Build the WHERE table: Assam roads cut into ≤500 m segments (MoRTH blackspot unit), with road features,
night-light, nearby pedestrian places, and iRAD crash counts split into train (2023-24) and test (2025-May 2026).
Inputs: data/osm_roads|osm_road_nodes|osm_pois.parquet, data/assam_boundary.wkt, data/viirs_vnp46a4_2024_assam.parquet, iRAD Accidents.
Output: model_ready/where_segments.parquet (+ crash_to_segment.parquet, internal only)
"""
import os, sys, math, numpy as np, pandas as pd, shapely
from shapely.ops import substring
from pyproj import Transformer

BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
IRAD = os.environ.get('IRAD', '/home/claude/eda/data/iRAD_Assam_all__Accidents.parquet')
D = os.path.join(BASE, 'data'); OUTD = os.path.join(BASE, 'model_ready'); os.makedirs(OUTD, exist_ok=True)
fwd = Transformer.from_crs(4326, 32646, always_xy=True); inv = Transformer.from_crs(32646, 4326, always_xy=True)
def proj(g): return shapely.transform(g, lambda c: np.column_stack(fwd.transform(c[:, 0], c[:, 1])))

# ---------------- roads inside Assam (+1 km), projected to metres
roads = pd.read_parquet(os.path.join(D, 'osm_roads.parquet'))
bnd = shapely.from_wkt(open(os.path.join(D, 'assam_boundary.wkt')).read())
geo = shapely.from_wkt(roads.wkt.values)
keep = shapely.intersects(bnd.buffer(0.01), geo)
roads = roads[keep].reset_index(drop=True); geo = proj(geo[keep])
print('roads in Assam', len(roads))

# ---------------- junctions: graph degree ≥3 from OSM node ids (interior node = 2 edges, end node = 1)
deg = {}
for ids in roads.node_ids.values:
    ids = ids.split(',')
    for k, nid in enumerate(ids):
        deg[nid] = deg.get(nid, 0) + (1 if k in (0, len(ids) - 1) else 2)
jpts = []
for ids, g in zip(roads.node_ids.values, geo):
    ids = ids.split(','); cs = shapely.get_coordinates(g)
    if len(ids) != len(cs): continue
    for nid, c in zip(ids, cs):
        if deg.get(nid, 0) >= 3: jpts.append((nid, c[0], c[1]))
jdf = pd.DataFrame(jpts, columns=['nid', 'x', 'y']).drop_duplicates('nid')
print('junction nodes', len(jdf))

# ---------------- cut each way into equal pieces ≤500 m
segs, meta = [], []
for i, g in enumerate(geo):
    L = g.length
    if L <= 0: continue
    n = max(1, math.ceil(L / 500))
    for k in range(n):
        s = substring(g, k * L / n, (k + 1) * L / n) if n > 1 else g
        segs.append(s); meta.append((i, k, n))
segs = np.array(segs, dtype=object)
S = pd.DataFrame(meta, columns=['road_idx', 'piece', 'pieces'])
r = roads.iloc[S.road_idx.values].reset_index(drop=True)
for c in ['way_id', 'highway', 'name', 'ref', 'lanes', 'oneway', 'maxspeed', 'surface', 'bridge']:
    S[c] = r[c].values
S['segment_id'] = 'S' + S.index.astype(str).str.zfill(6)
S['length_m'] = shapely.length(segs).round(1)
mid = shapely.line_interpolate_point(segs, 0.5, normalized=True)
mx, my = shapely.get_x(mid), shapely.get_y(mid)
lon, lat = inv.transform(mx, my); S['mid_lat'] = np.round(lat, 6); S['mid_lon'] = np.round(lon, 6)
print('segments', len(S), 'total km', round(S.length_m.sum() / 1000))

# geometry shape: sinuosity and turning (degrees per km)
def turning(g):
    c = shapely.get_coordinates(g)
    if len(c) < 3: return 0.0
    v = np.diff(c, axis=0); ang = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    d = np.abs((np.diff(ang) + 180) % 360 - 180)
    return float(d.sum())
chord = shapely.distance(shapely.get_point(segs, 0), shapely.get_point(segs, -1))
S['sinuosity'] = np.round(S.length_m / np.maximum(chord, 1), 3)
S['turn_deg_per_km'] = np.round([turning(g) for g in segs] / np.maximum(S.length_m.values / 1000, 0.05), 1)
S['is_main_road'] = S.highway.isin(['trunk', 'primary', 'secondary', 'trunk_link', 'primary_link', 'secondary_link']).astype(int)
S['is_national_highway'] = S.ref.fillna('').str.contains(r'\bNH', case=False).astype(int)
S['lanes_num'] = pd.to_numeric(S.lanes, errors='coerce')
S['is_bridge'] = S.bridge.notna().astype(int)

tree = shapely.STRtree(segs)
def count_near(px, py, dist, labels=None):
    pts = shapely.points(px, py)
    pi, si = tree.query(pts, predicate='dwithin', distance=dist)
    if labels is None:
        return np.bincount(si, minlength=len(S))
    lab = np.asarray(labels)[pi]
    return {l: np.bincount(si[lab == l], minlength=len(S)) for l in np.unique(lab)}

S['junctions'] = count_near(jdf.x.values, jdf.y.values, 5)
S['junctions_per_km'] = np.round(S.junctions / np.maximum(S.length_m / 1000, 0.05), 2)
nodes = pd.read_parquet(os.path.join(D, 'osm_road_nodes.parquet'))
nx, ny = fwd.transform(nodes.lon.values, nodes.lat.values)
nk = nodes.kind.map({'railway=level_crossing': 'rail_crossing', 'traffic_calming': 'speed_breaker', 'highway=traffic_signals': 'signal',
                     'highway=crossing': 'ped_crossing', 'highway=bus_stop': 'bus_stop'}).fillna('other').values
cn = count_near(nx, ny, 30, nk)
for k in ['rail_crossing', 'speed_breaker', 'signal', 'ped_crossing', 'bus_stop']:
    S[f'n_{k}'] = cn.get(k, np.zeros(len(S), int))
pois = pd.read_parquet(os.path.join(D, 'osm_pois.parquet'))
def pcat(k, v):
    if k == 'shop' or v == 'marketplace': return 'market_shop'
    if v in ('school', 'college', 'university', 'kindergarten'): return 'education'
    if v in ('hospital', 'clinic', 'doctors'): return 'health'
    if v in ('bus_station',) or k in ('railway', 'public_transport'): return 'transit'
    if v in ('restaurant', 'fast_food', 'cafe'): return 'eatery'
    if v == 'place_of_worship': return 'worship'
    if v == 'fuel': return 'fuel'
    return 'other'
pc = np.array([pcat(k, v) for k, v in zip(pois.key, pois.value)])
px_, py_ = fwd.transform(pois.lon.values, pois.lat.values)
cp = count_near(px_, py_, 200, pc)
for k in ['market_shop', 'education', 'health', 'transit', 'eatery', 'worship', 'fuel']:
    S[f'poi200_{k}'] = cp.get(k, np.zeros(len(S), int))
S['poi200_total'] = S[[c for c in S if c.startswith('poi200_')]].sum(axis=1)

# ---------------- night lights (VIIRS 2024, 15 arc-sec): 3x3 mean around segment midpoint
v = pd.read_parquet(os.path.join(D, 'viirs_vnp46a4_2024_assam.parquet'))
step = 10 / 2400
v['i'] = np.round((90 - v.lat) / step - 0.5).astype(int); v['j'] = np.round((v.lon + 180) / step - 0.5).astype(int)
i0, j0 = v.i.min(), v.j.min(); A = np.full((v.i.max() - i0 + 3, v.j.max() - j0 + 3), np.nan)
A[v.i - i0 + 1, v.j - j0 + 1] = v.radiance_nw.values
si_ = np.round((90 - S.mid_lat.values) / step - 0.5).astype(int) - i0 + 1
sj_ = np.round((S.mid_lon.values + 180) / step - 0.5).astype(int) - j0 + 1
si_ = np.clip(si_, 1, A.shape[0] - 2); sj_ = np.clip(sj_, 1, A.shape[1] - 2)
win = np.stack([A[si_ + a, sj_ + b] for a in (-1, 0, 1) for b in (-1, 0, 1)])
S['night_light_nw'] = np.round(np.nanmean(win, axis=0), 3)
S['night_light_class'] = pd.cut(S.night_light_nw, [-1, 0.5, 5, 20, 1e9], labels=['dark_rural', 'dim', 'lit_town', 'bright_urban']).astype(str)

# ---------------- iRAD crashes → nearest segment within 100 m
a = pd.read_parquet(IRAD, columns=['accident_id', 'accident_datetime', 'latitude', 'longitude', 'severity', 'killed_total', 'grievous_total', 'collision_type'])
a = a.dropna(subset=['latitude', 'longitude', 'accident_datetime'])
a['dt'] = pd.to_datetime(a.accident_datetime)
ax, ay = fwd.transform(a.longitude.values, a.latitude.values)
idx, dist = tree.query_nearest(shapely.points(ax, ay), max_distance=100, return_distance=True, all_matches=False)
a['segment_id'] = None; a['snap_m'] = np.nan
a.iloc[idx[0], a.columns.get_loc('segment_id')] = S.segment_id.values[idx[1]]
a.iloc[idx[0], a.columns.get_loc('snap_m')] = np.round(dist, 1)
a['period'] = np.where(a.dt < '2025-01-01', 'train_2023_24', 'test_2025_26')
a['ksi'] = a.severity.isin(['Fatal', 'Grievous Injury']).astype(int)
a['fatal'] = (a.severity == 'Fatal').astype(int)
a['ped'] = a.collision_type.fillna('').str.contains('Pedestrian', case=False).astype(int)
print(f"crashes snapped within 100 m: {a.segment_id.notna().mean():.1%}  (median distance {a.snap_m.median():.0f} m)")
a[['accident_id', 'segment_id', 'snap_m', 'period', 'ksi', 'fatal']].to_parquet(os.path.join(OUTD, 'crash_to_segment_INTERNAL.parquet'), index=False)
m = a.dropna(subset=['segment_id'])
for p in ['train_2023_24', 'test_2025_26']:
    g = m[m.period == p].groupby('segment_id').agg(crashes=('accident_id', 'size'), ksi=('ksi', 'sum'), fatal_crashes=('fatal', 'sum'),
                                                    killed=('killed_total', 'sum'), ped_crashes=('ped', 'sum'))
    g.columns = [f'{c}_{p[:5]}' for c in g.columns]
    S = S.merge(g, left_on='segment_id', right_index=True, how='left')
cnt = [c for c in S if c.endswith(('_train', '_test_'))]
S[cnt] = S[cnt].fillna(0).astype(int)
S = S.rename(columns={c: c.replace('_test_', '_test') for c in S})
# MoRTH-style blackspot flag, adapted to our 2-year windows: ≥5 fatal/grievous crashes OR ≥10 killed on the ≤500 m segment
S['blackspot_train'] = ((S.ksi_train >= 5) | (S.killed_train >= 10)).astype(int)
S['blackspot_test'] = ((S.ksi_test >= 5) | (S.killed_test >= 10)).astype(int)
S['new_blackspot_test'] = ((S.blackspot_test == 1) & (S.blackspot_train == 0)).astype(int)
S['wkt_lonlat'] = shapely.to_wkt(shapely.transform(segs, lambda c: np.column_stack(inv.transform(c[:, 0], c[:, 1]))), rounding_precision=6)
S = S.drop(columns=['road_idx'])
S.to_parquet(os.path.join(OUTD, 'where_segments.parquet'), index=False)
print('segments', len(S), '| train KSI', S.ksi_train.sum(), '| test KSI', S.ksi_test.sum(), '| blackspots train/test/new',
      S.blackspot_train.sum(), S.blackspot_test.sum(), S.new_blackspot_test.sum())
