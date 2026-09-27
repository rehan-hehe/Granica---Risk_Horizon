"""Live traffic collector for the Guwahati corridor (runs during the 48 h and the pitch).
Polls ONLY the tiles/points in scope/ (built from iRAD corridor hotspots):
  - Traffic Flow vector tiles (relative speed per road piece), every 15 min
  - Flow Segment Data (current vs free-flow speed) at 24 hotspot points, every 60 min
Writes one Parquet part per hour to data/live/ and logs every call to manifest.jsonl.
Geometry is stored once per unique road piece (dedup), readings store only a hash -> small files.

Usage:  python tomtom_live_collector.py --key YOUR_TOMTOM_KEY --hours 50
Check TomTom's terms for storing traffic data before publishing raw readings; publish aggregates if in doubt.
"""
import argparse, datetime as dt, hashlib, json, os, time, urllib.parse, urllib.request
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.join(HERE, 'data', 'live'); os.makedirs(LIVE, exist_ok=True)
TILES = pd.read_csv(os.path.join(HERE, 'scope', 'tomtom_flow_tiles_z13.csv'))
POINTS = pd.read_csv(os.path.join(HERE, 'scope', 'tomtom_segment_points.csv'))


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'RiskHorizon-IITG/1.0'}), timeout=60) as r:
        return r.read()


def log(kind, n_ok, n_err, note=''):
    with open(os.path.join(HERE, 'manifest.jsonl'), 'a') as f:
        f.write(json.dumps(dict(ts=dt.datetime.now().isoformat(timespec='seconds'), source=f'tomtom-{kind}', ok=n_ok, errors=n_err, note=note)) + '\n')


def tile_to_lonlat(x, y, z, px, py, extent):
    import math
    n = 2 ** z
    lon = (x + px / extent) / n * 360 - 180
    yt = y + (1 - py / extent)  # MVT y axis points down; mapbox_vector_tile flips to up by default
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yt / n))))
    return round(lon, 6), round(lat, 6)


def decode_flow_tile(b, z, x, y):
    import mapbox_vector_tile
    d = mapbox_vector_tile.decode(b)
    out = []
    for lname, layer in d.items():
        ext = layer.get('extent', 4096)
        for f in layer['features']:
            g = f['geometry']; coords = g['coordinates']
            lines = coords if g['type'] == 'MultiLineString' else [coords]
            ll = [[tile_to_lonlat(x, y, z, px, py, ext) for px, py in line] for line in lines if line]
            geom = json.dumps(ll, separators=(',', ':'))
            gh = hashlib.sha1(f'{z}/{x}/{y}/{geom}'.encode()).hexdigest()[:16]
            p = f.get('properties', {})
            out.append(dict(layer=lname, geom_hash=gh, geom=geom, **{k: p.get(k) for k in
                            ['road_type', 'road_category', 'road_subcategory', 'traffic_level', 'traffic_road_coverage', 'road_closure', 'left_hand_traffic']}))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--key', required=True); ap.add_argument('--hours', type=float, default=50)
    ap.add_argument('--tile-every', type=int, default=15); ap.add_argument('--point-every', type=int, default=60)
    a = ap.parse_args()
    geoms_fn = os.path.join(LIVE, 'road_pieces.parquet')
    seen = set(pd.read_parquet(geoms_fn).geom_hash) if os.path.exists(geoms_fn) else set()
    t_end = time.time() + a.hours * 3600
    last_tile = last_point = 0
    buf_tiles, buf_pts, new_geoms = [], [], []
    hour_key = None
    while time.time() < t_end:
        now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
        if time.time() - last_tile >= a.tile_every * 60:
            last_tile = time.time(); ok = err = 0
            for t in TILES.itertuples():
                url = f'https://api.tomtom.com/traffic/map/4/tile/flow/relative/{t.z}/{t.x}/{t.y}.pbf?key={a.key}'
                try:
                    for r in decode_flow_tile(fetch(url), t.z, t.x, t.y):
                        if r['geom_hash'] not in seen:
                            seen.add(r['geom_hash']); new_geoms.append(dict(geom_hash=r['geom_hash'], tile=f'{t.z}/{t.x}/{t.y}', layer=r['layer'],
                                                                            road_type=r['road_type'], road_category=r['road_category'], geom=r['geom']))
                        buf_tiles.append(dict(time_utc=now, tile=f'{t.z}/{t.x}/{t.y}', geom_hash=r['geom_hash'], traffic_level=r['traffic_level'],
                                              traffic_road_coverage=r['traffic_road_coverage'], road_closure=r['road_closure']))
                    ok += 1
                except Exception as e:
                    err += 1
            log('flow-tiles', ok, err)
        if time.time() - last_point >= a.point_every * 60:
            last_point = time.time(); ok = err = 0
            for p in POINTS.itertuples():
                url = ('https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?' +
                       urllib.parse.urlencode(dict(point=f'{p.lat},{p.lon}', unit='KMPH', key=a.key)))
                try:
                    fs = json.loads(fetch(url))['flowSegmentData']
                    buf_pts.append(dict(time_utc=now, point_id=p.point_id, site=p.site, frc=fs.get('frc'), current_speed_kmh=fs.get('currentSpeed'),
                                        free_flow_speed_kmh=fs.get('freeFlowSpeed'), current_tt_s=fs.get('currentTravelTime'),
                                        free_flow_tt_s=fs.get('freeFlowTravelTime'), confidence=fs.get('confidence'), road_closure=fs.get('roadClosure')))
                    ok += 1
                except Exception:
                    err += 1
            log('segment-points', ok, err)
        hk = now.strftime('%Y%m%d_%H')
        if hour_key and hk != hour_key or time.time() >= t_end:
            flush(hour_key, buf_tiles, buf_pts, new_geoms, geoms_fn); buf_tiles, buf_pts, new_geoms = [], [], []
        hour_key = hk
        time.sleep(20)
    flush(hour_key, buf_tiles, buf_pts, new_geoms, geoms_fn)


def flush(hk, tiles, pts, geoms, geoms_fn):
    if tiles: pd.DataFrame(tiles).to_parquet(os.path.join(LIVE, f'flow_tiles_{hk}.parquet'), index=False)
    if pts: pd.DataFrame(pts).to_parquet(os.path.join(LIVE, f'segment_points_{hk}.parquet'), index=False)
    if geoms:
        g = pd.DataFrame(geoms)
        if os.path.exists(geoms_fn): g = pd.concat([pd.read_parquet(geoms_fn), g]).drop_duplicates('geom_hash')
        g.to_parquet(geoms_fn, index=False)
    print(dt.datetime.now().strftime('%H:%M'), f'flushed hour {hk}: {len(tiles)} tile readings, {len(pts)} point readings, {len(geoms)} new road pieces')


if __name__ == '__main__':
    main()
