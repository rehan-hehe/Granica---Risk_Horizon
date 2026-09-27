"""OpenStreetMap roads, road-feature nodes and pedestrian POIs for Assam (needed tags only). Needs: pip install osmium
Standalone: needs only this file + the scope files listed below, so it can run on any PC.
Run from the Risk_Horizon_Data folder:  python fetch_osm.py [--keep-pbf]
Resumable: re-running skips anything already saved. Log: manifest_osm.jsonl
Needs scope/: scope.json
"""
import argparse, datetime as dt, gzip, hashlib, io, json, os, sys, time, urllib.error, urllib.parse, urllib.request
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCOPE = os.path.join(HERE, 'scope'); RAW = os.path.join(HERE, 'raw'); OUT = os.path.join(HERE, 'data')
for d in (RAW, OUT): os.makedirs(d, exist_ok=True)
UA = {'User-Agent': 'RiskHorizon-IITG-hackathon/1.0 (student research)'}
SCOPEJ = json.load(open(os.path.join(SCOPE, 'scope.json')))
MANIFEST = os.path.join(HERE, 'manifest_osm.jsonl')


def log(source, url, nbytes, sha, rows=None, note=''):
    with open(MANIFEST, 'a') as f:
        f.write(json.dumps(dict(ts=dt.datetime.now().isoformat(timespec='seconds'), source=source, url=url.split('key=')[0],
                                bytes=nbytes, sha256=sha, rows=rows, note=note)) + '\n')

def get(url, headers=None, tries=4, timeout=120):
    h = dict(UA); h.update(headers or {})
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
                b = r.read()
                if r.headers.get('Content-Encoding') == 'gzip': b = gzip.decompress(b)
                return b
        except urllib.error.HTTPError as e:
            if e.code == 429: raise
            if e.code in (400, 401, 403, 404): raise
            time.sleep(10 * (i + 1))
        except Exception:
            time.sleep(10 * (i + 1))
    raise RuntimeError('failed: ' + url.split('key=')[0])


# ------------------------------------------------------------------ OSM
ROAD_CLASSES = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'living_street', 'road',
                'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link'}
ROAD_TAGS = ['highway', 'name', 'ref', 'lanes', 'oneway', 'maxspeed', 'surface', 'lit', 'bridge', 'junction', 'sidewalk', 'layer', 'divider', 'dual_carriageway']
NODE_FEATURES = {('highway', 'traffic_signals'), ('highway', 'crossing'), ('highway', 'stop'), ('highway', 'give_way'), ('highway', 'speed_camera'),
                 ('highway', 'bus_stop'), ('highway', 'street_lamp'), ('railway', 'level_crossing')}
POI = {'amenity': {'school', 'college', 'university', 'kindergarten', 'hospital', 'clinic', 'doctors', 'marketplace', 'bus_station',
                   'place_of_worship', 'fuel', 'bank', 'restaurant', 'fast_food', 'cafe', 'police'},
       'shop': None, 'railway': {'station', 'halt'}, 'public_transport': {'station'}}


def osm(args):
    import osmium
    bb = SCOPEJ['osm_bbox']
    if os.path.exists(os.path.join(OUT, 'osm_roads.parquet')) and not args.keep_pbf:
        print('osm: already extracted (data/osm_roads.parquet) - skipping'); return
    pbf = os.path.join(RAW, 'north-eastern-zone-latest.osm.pbf')
    url = 'https://download.geofabrik.de/asia/india/north-eastern-zone-latest.osm.pbf'
    if not os.path.exists(pbf):
        print('downloading', url, '(~104 MB, deleted after extraction unless --keep-pbf)')
        b = get(url, timeout=1800); open(pbf, 'wb').write(b)
        log('geofabrik', url, len(b), hashlib.sha256(b).hexdigest(), None, 'NE India PBF')
    inb = lambda lat, lon: bb['south'] <= lat <= bb['north'] and bb['west'] <= lon <= bb['east']
    wkt = osmium.geom.WKTFactory()

    class H(osmium.SimpleHandler):
        def __init__(s):
            super().__init__(); s.roads = []; s.nodes = []; s.pois = []; s.boundary = None; s.ways_seen = 0

        def node(s, n):
            if not n.tags or not n.location.valid() or not inb(n.location.lat, n.location.lon): return
            t = n.tags
            if 'traffic_calming' in t:
                s.nodes.append(dict(node_id=n.id, lat=n.location.lat, lon=n.location.lon, kind='traffic_calming', value=t['traffic_calming']))
            for k, v in NODE_FEATURES:
                if t.get(k) == v: s.nodes.append(dict(node_id=n.id, lat=n.location.lat, lon=n.location.lon, kind=f'{k}={v}', value=t.get('crossing', '')))
            s._poi('n', n.id, t, n.location.lat, n.location.lon)

        def _poi(s, typ, oid, t, lat, lon):
            for k, vals in POI.items():
                if k in t and (vals is None or t[k] in vals):
                    s.pois.append(dict(osm_type=typ, osm_id=oid, lat=lat, lon=lon, key=k, value=t[k], name=t.get('name', ''))); return

        def way(s, w):
            t = w.tags
            try:
                locs = [(nd.lat, nd.lon) for nd in w.nodes]
            except osmium.InvalidLocationError:
                return
            if not any(inb(la, lo) for la, lo in locs): return
            if t.get('highway') in ROAD_CLASSES:
                s.roads.append(dict(way_id=w.id, **{k: t.get(k) for k in ROAD_TAGS}, n_nodes=len(locs),
                                    node_ids=','.join(str(nd.ref) for nd in w.nodes), wkt=wkt.create_linestring(w)))
            elif w.is_closed() and len(locs) > 2:
                s._poi('w', w.id, t, sum(x[0] for x in locs) / len(locs), sum(x[1] for x in locs) / len(locs))

        def area(s, a):
            if a.tags.get('boundary') == 'administrative' and a.tags.get('admin_level') == '4' and a.tags.get('name') in ('Assam', 'Asom'):
                s.boundary = wkt.create_multipolygon(a)

    h = H(); h.apply_file(pbf, locations=True, idx='flex_mem')
    roads = pd.DataFrame(h.roads); nodes = pd.DataFrame(h.nodes); pois = pd.DataFrame(h.pois)
    for df, name in [(roads, 'osm_roads.parquet'), (nodes, 'osm_road_nodes.parquet'), (pois, 'osm_pois.parquet')]:
        df.to_parquet(os.path.join(OUT, name), index=False)
    if h.boundary: open(os.path.join(OUT, 'assam_boundary.wkt'), 'w').write(h.boundary)
    cov = {k: round(roads[k].notna().mean(), 4) for k in ['lanes', 'maxspeed', 'oneway', 'lit', 'surface', 'bridge']}
    log('osm-extract', pbf, os.path.getsize(pbf), None, len(roads), json.dumps(dict(nodes=len(nodes), pois=len(pois), tag_coverage=cov, boundary=bool(h.boundary))))
    print(f'roads {len(roads)}, feature nodes {len(nodes)}, POIs {len(pois)}, Assam boundary: {bool(h.boundary)}; tag coverage {cov}')
    print(nodes.kind.value_counts().to_string() if len(nodes) else 'no feature nodes')
    if not args.keep_pbf: os.remove(pbf)


def combine(sub, outname, dedupe):
    d = os.path.join(RAW, sub)
    fs = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith('.parquet')]
    if not fs: return
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).drop_duplicates(dedupe)
    df.to_parquet(os.path.join(OUT, outname), index=False)
    print(f'combined -> data/{outname}: {len(df):,} rows')



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--keep-pbf', action='store_true')
    args = p.parse_args()
    osm(args)
