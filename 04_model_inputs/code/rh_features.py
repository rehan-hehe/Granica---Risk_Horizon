"""Shared feature engineering for Risk Horizon (reads only the readable CSVs).

Two kinds of features:
  static_features()   -> road design, junctions/points, pedestrian places, lighting, network context (do not change with time)
  history_features()  -> crash history around each segment for a chosen time window (rates per month, own segment excluded)
Neighbourhood sums use a 200 m grid in a local km projection and disk-shaped convolution (fast, low memory).
"""
import os, numpy as np, pandas as pd
from scipy.signal import fftconvolve
from scipy import ndimage

LAT0, LON0 = 26.2, 92.9
KX, KY = 111.32 * np.cos(np.radians(LAT0)), 110.57
CELL = 0.2                                                     # km

def to_km(lat, lon):
    return (np.asarray(lon) - LON0) * KX, (np.asarray(lat) - LAT0) * KY

class Grid:
    def __init__(self, x, y, pad=6.0):
        self.x0, self.y0 = x.min() - pad, y.min() - pad
        self.nx = int(np.ceil((x.max() + pad - self.x0) / CELL)) + 1; self.ny = int(np.ceil((y.max() + pad - self.y0) / CELL)) + 1
    def idx(self, x, y):
        return (np.clip(((y - self.y0) / CELL).astype(int), 0, self.ny - 1), np.clip(((x - self.x0) / CELL).astype(int), 0, self.nx - 1))
    def raster(self, x, y, w=None):
        r = np.zeros((self.ny, self.nx), np.float32); iy, ix = self.idx(x, y)
        np.add.at(r, (iy, ix), 1.0 if w is None else np.asarray(w, np.float32)); return r
    @staticmethod
    def disk(radius_km):
        n = int(np.ceil(radius_km / CELL)); yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
        return ((xx ** 2 + yy ** 2) * CELL ** 2 <= radius_km ** 2).astype(np.float32)
    def disk_sum(self, r, radius_km):
        return np.maximum(fftconvolve(r, self.disk(radius_km), mode='same'), 0)
    def sample(self, r, x, y):
        iy, ix = self.idx(x, y); return r[iy, ix]

HW_RANK = {'motorway': 7, 'trunk': 6, 'primary': 5, 'secondary': 4, 'tertiary': 3, 'unclassified': 2, 'residential': 1, 'living_street': 1, 'service': 0, 'track': 0}

def static_features(inp):
    """inp = path to 2_readable/10_model_inputs. Returns DataFrame (one row per segment) + a definitions table."""
    S = pd.read_csv(os.path.join(inp, 'where_segments.csv'), low_memory=False)
    x, y = to_km(S.mid_lat.values, S.mid_lon.values); G = Grid(x, y)
    hw = S.highway.fillna('other'); base = hw.str.replace('_link', '')
    F = pd.DataFrame({'segment_id': S.segment_id})
    D = []
    def add(name, val, group, source, definition):
        F[name] = val; D.append((name, group, source, definition))
    # --- road class & design (OSM)
    add('hw_rank', base.map(HW_RANK).fillna(1).values, 'road_class', 'OSM', 'Road hierarchy 0 (service/track) .. 6 trunk')
    add('is_main_road', S.is_main_road.values, 'road_class', 'OSM', 'trunk/primary/secondary incl. links')
    add('is_national_highway', S.is_national_highway.values, 'road_class', 'OSM', 'ref contains NH')
    add('is_link', hw.str.endswith('_link').astype(int).values, 'road_class', 'OSM', 'slip road / link')
    add('is_divided', ((S.oneway == 'yes') & (S.is_main_road == 1)).astype(int).values, 'road_class', 'OSM', 'oneway main road (dual carriageway proxy)')
    add('is_unpaved', S.surface.isin(['unpaved', 'gravel', 'dirt', 'ground', 'sand']).astype(int).values, 'road_class', 'OSM', 'surface tag unpaved (94% of tags missing)')
    add('is_bridge', S.is_bridge.values, 'geometry', 'OSM', 'bridge tag')
    add('length_m', S.length_m.values, 'geometry', 'OSM', 'segment length (m)')
    add('sinuosity', S.sinuosity.clip(1, 5).values, 'geometry', 'OSM', 'path length / straight-line length')
    add('turn_deg_per_km', S.turn_deg_per_km.clip(0, 5000).values, 'geometry', 'OSM', 'total turning per km')
    # curve surprise: this segment's turning vs its neighbours on the same way (a bend after a straight)
    t = S.turn_deg_per_km.clip(0, 5000).values; w = S.way_id.values
    prev_ = np.r_[np.nan, t[:-1]]; prev_[np.r_[True, w[1:] != w[:-1]]] = np.nan
    next_ = np.r_[t[1:], np.nan]; next_[np.r_[w[1:] != w[:-1], True]] = np.nan
    with np.errstate(all='ignore'):
        import warnings; warnings.simplefilter('ignore', RuntimeWarning); nb = np.nanmean(np.vstack([prev_, next_]), axis=0)
    add('turn_contrast', np.where(np.isnan(nb), 0, t - nb), 'geometry', 'OSM (engineered)', 'turning minus mean of neighbouring segments on the same way')
    # --- junctions and road points
    add('junctions', S.junctions.values, 'junction_points', 'OSM', 'junction nodes within 5 m')
    add('junctions_per_km', S.junctions_per_km.clip(0, 40).values, 'junction_points', 'OSM', 'junctions per km')
    for c in ['n_rail_crossing', 'n_signal', 'n_speed_breaker', 'n_ped_crossing', 'n_bus_stop']:
        add(c, S[c].values, 'junction_points', 'OSM', f'{c[2:]} within 30 m')
    rj = G.raster(x, y, S.junctions.values)
    add('junctions_1km', G.sample(G.disk_sum(rj, 1.0), x, y), 'junction_points', 'OSM (engineered)', 'junction nodes on segments within 1 km')
    # --- pedestrian places (POIs)
    for c in ['market_shop', 'education', 'health', 'transit', 'eatery', 'worship', 'fuel', 'total']:
        add(f'poi200_{c}', S[f'poi200_{c}'].values, 'pedestrian_places', 'OSM', f'{c} places within 200 m')
    P = pd.read_csv(os.path.join(inp, 'raw_layers', 'osm_pois.csv')); px, py = to_km(P.lat.values, P.lon.values); rp = G.raster(px, py)
    add('poi_500m', G.sample(G.disk_sum(rp, 0.5), x, y), 'pedestrian_places', 'OSM (engineered)', 'all places within 500 m')
    add('poi_2km', G.sample(G.disk_sum(rp, 2.0), x, y), 'pedestrian_places', 'OSM (engineered)', 'all places within 2 km (settlement size)')
    edu = P.value.isin(['school', 'college', 'university', 'kindergarten']).values
    add('school_500m', G.sample(G.disk_sum(G.raster(px[edu], py[edu]), 0.5), x, y), 'pedestrian_places', 'OSM (engineered)', 'schools/colleges within 500 m')
    add('ped_places_x_main', S.poi200_total.values * S.is_main_road.values, 'pedestrian_places', 'engineered', 'places within 200 m on a main road (fast traffic meets pedestrians)')
    # --- lighting (VIIRS)
    V = pd.read_csv(os.path.join(inp, 'raw_layers', 'viirs_night_lights_2024.csv')); vx, vy = to_km(V.lat.values, V.lon.values)
    cnt = G.raster(vx, vy); tot = G.raster(vx, vy, V.radiance_nw.values)
    def mean_r(rad):
        return G.sample(G.disk_sum(tot, rad), x, y) / np.maximum(G.sample(G.disk_sum(cnt, rad), x, y), 1)
    nl = S.night_light_nw.fillna(0).clip(0, 300).values
    add('night_light', nl, 'lighting', 'VIIRS', 'radiance around segment (3x3 pixels, ~1.4 km)')
    add('night_light_3km', mean_r(3.0), 'lighting', 'VIIRS (engineered)', 'mean radiance within 3 km')
    add('night_light_10km', mean_r(10.0), 'lighting', 'VIIRS (engineered)', 'mean radiance within 10 km (regional development)')
    add('light_contrast', np.log1p(nl) - np.log1p(F.night_light_3km.values), 'lighting', 'VIIRS (engineered)', 'local vs 3 km light: lit strip in a dark area > 0')
    bright = G.raster(vx[V.radiance_nw.values >= 20], vy[V.radiance_nw.values >= 20]) > 0
    dist = ndimage.distance_transform_edt(~bright) * CELL
    add('dist_to_town_km', np.minimum(G.sample(dist, x, y), 50), 'lighting', 'VIIRS (engineered)', 'distance to nearest bright-urban pixel (>=20 nW), capped 50 km')
    # --- network context
    add('road_km_1km', G.sample(G.disk_sum(G.raster(x, y, S.length_m.values / 1000), 1.0), x, y), 'network', 'OSM (engineered)', 'road km within 1 km')
    add('road_km_5km', G.sample(G.disk_sum(G.raster(x, y, S.length_m.values / 1000), 5.0), x, y), 'network', 'OSM (engineered)', 'road km within 5 km (urbanity)')
    mr = G.raster(x, y, S.length_m.values / 1000 * S.is_main_road.values)
    add('main_road_km_1km', G.sample(G.disk_sum(mr, 1.0), x, y), 'network', 'OSM (engineered)', 'main-road km within 1 km')
    dmain = ndimage.distance_transform_edt(~(mr > 0)) * CELL
    add('dist_to_main_road_km', np.minimum(G.sample(dmain, x, y), 20), 'network', 'OSM (engineered)', 'distance to nearest main road, capped 20 km')
    # --- location
    add('mid_lat', S.mid_lat.values, 'location', 'OSM', 'latitude of segment midpoint'); add('mid_lon', S.mid_lon.values, 'location', 'OSM', 'longitude of segment midpoint')
    return S, F, pd.DataFrame(D, columns=['feature', 'group', 'source', 'definition']), (x, y, G)

def month_list(start, end):
    return pd.period_range(start, end, freq='M').strftime('%Y-%m').tolist()

def history_features(E, seg_ids, xyG, months, recent_months=12, seg_lat=None, seg_lon=None, seg_len=None):
    assert seg_lat is not None, 'pass seg_lat/seg_lon/seg_len'
    """E = crash_events (snapped). Counts per segment in `months`, then per-month rates around each segment (own segment excluded)."""
    x, y, G = xyG
    data_months = sorted(set(E.month) & set(months)); n = max(len(data_months), 1)
    e = E[E.month.isin(months) & E.segment_id.notna()]
    idx = pd.Series(np.arange(len(seg_ids)), index=seg_ids)
    def per_seg(df, col):
        v = np.zeros(len(seg_ids)); g = df.groupby('segment_id')[col].sum(); v[idx[g.index].values] = g.values; return v
    own_ksi, own_cr, own_fat = per_seg(e, 'ksi'), per_seg(e.assign(one=1), 'one'), per_seg(e, 'fatal')
    rec = sorted(data_months)[-recent_months:]; er = e[e.month.isin(rec)]; own_ksi_rec = per_seg(er, 'ksi')
    H = pd.DataFrame(index=range(len(seg_ids)))
    D = []
    def add(name, val, definition): H[name] = val; D.append((name, 'crash_history', 'iRAD (engineered)', definition))
    # counts are whole numbers: round after removing the segment's own crashes. Without rounding, float residue from the
    # FFT (~1e-6, negative only when the segment itself had crashes) leaks the segment's own history into the feature.
    from sklearn.neighbors import BallTree
    ll = np.radians(np.column_stack([seg_lat, seg_lon]))
    def nearby(own_v, rad):
        # exact great-circle search over crash-bearing segments only (fast); own segment excluded
        nz = np.flatnonzero(own_v > 0)
        if len(nz) == 0: return np.zeros(len(own_v))
        ind = BallTree(ll[nz], metric='haversine').query_radius(ll, r=rad / 6371.0)
        w = own_v[nz]; tot = np.fromiter((w[i].sum() for i in ind), float, len(ind))
        return np.clip(tot - own_v, 0, None)
    for rad, lab in [(0.5, '500m'), (1.0, '1km'), (3.0, '3km')]:
        add(f'nearby_ksi_{lab}', nearby(own_ksi, rad) / n, f'KSI crashes per month on other segments within {lab}')
    add('nearby_crashes_1km', nearby(own_cr, 1.0) / n, 'all crashes per month on other segments within 1 km')
    add('nearby_fatal_1km', nearby(own_fat, 1.0) / n, 'fatal crashes per month on other segments within 1 km')
    add('nearby_ksi_1km_recent', nearby(own_ksi_rec, 1.0) / max(len(rec), 1), f'KSI per month within 1 km, last {len(rec)} data months')
    H['nearby_severity_ratio_1km'] = (H.nearby_ksi_1km + 0.01) / (H.nearby_crashes_1km + 0.02)
    D.append(('nearby_severity_ratio_1km', 'crash_history', 'iRAD (engineered)', 'share of nearby crashes that were KSI'))
    # --- drift-corrected history: iRAD reporting grew unevenly across Assam (x0.8 .. x3.8 by area, 2023 -> 2024/25).
    #     Divide nearby crash rates by what the 0.5-degree region reports per road-km in the same window -> 'observed / regional expected'.
    if True:
        lat = np.asarray(seg_lat); lon = np.asarray(seg_lon); Lkm = np.asarray(seg_len) / 1000
        blk = pd.Series(np.floor(lat / 0.5).astype(int) * 1000 + np.floor(lon / 0.5).astype(int))
        reg_rate = (pd.Series(own_ksi).groupby(blk).transform('sum') / pd.Series(Lkm).groupby(blk).transform('sum')).values / n   # same value for every segment in the region (never subtract own: that would leak it)
        road1 = G.sample(G.disk_sum(G.raster(x, y, Lkm), 1.0), x, y); road3 = G.sample(G.disk_sum(G.raster(x, y, Lkm), 3.0), x, y)
        H['nearby_ksi_1km_rel'] = (H.nearby_ksi_1km + 1e-4) / (reg_rate * np.maximum(road1 - Lkm, 0.05) + 1e-4)
        H['nearby_ksi_3km_rel'] = (H.nearby_ksi_3km + 1e-4) / (reg_rate * np.maximum(road3 - Lkm, 0.05) + 1e-4)
        H['regional_ksi_rate'] = reg_rate
        D += [('nearby_ksi_1km_rel', 'crash_history_rel', 'iRAD (engineered)', 'nearby KSI rate within 1 km / regional KSI rate per road-km (reporting-drift corrected)'),
              ('nearby_ksi_3km_rel', 'crash_history_rel', 'iRAD (engineered)', 'nearby KSI rate within 3 km / regional KSI rate per road-km (reporting-drift corrected)'),
              ('regional_ksi_rate', 'reporting_level', 'iRAD (engineered)', 'KSI per road-km per month reported in the 0.5-degree region (mostly reporting level)')]
    own = dict(ksi=own_ksi, crashes=own_cr, fatal=own_fat, data_months=n)
    return H, own, pd.DataFrame(D, columns=['feature', 'group', 'source', 'definition'])

def spf_transform(X, cols):
    """Same transform the feature study used for the NB SPF: log1p for non-negative counts/rates with a long tail."""
    Z = pd.DataFrame(index=X.index)
    for c in cols:
        v = X[c].values
        Z[c] = np.log1p(np.clip(v, 0, None)) if (v.min() >= 0 and v.max() > 5) else v
    return Z

def capture_metrics(score, target, L, qs=(0.01, 0.05, 0.10, 0.20)):
    o = np.argsort(-(score + 1e-9 * L)); cum = np.cumsum(L[o]) / L.sum(); cap = np.cumsum(target[o]) / target.sum()
    out = {f'cap{int(q * 100)}': float(np.interp(q, cum, cap)) for q in qs}
    out['aucc20'] = float(np.interp(np.linspace(0.001, 0.20, 200), cum, cap).mean())
    return out
