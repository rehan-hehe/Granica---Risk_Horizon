"""Scoped downloader for the Risk Horizon public datasets.
Fetches ONLY what scope/*.csv allows (built from the iRAD crash footprint by build_scope.py).

Usage (from the Public_Data folder):
    python fetch_public_data.py weather            # Open-Meteo ERA5 hourly, planned cells/date ranges only (resumable)
    python fetch_public_data.py metar              # METAR for the 6 Assam airports near crashes (Iowa Mesonet)
    python fetch_public_data.py osm                # Geofabrik NE-India PBF -> Assam roads/nodes/POIs (needed tags only)
    python fetch_public_data.py viirs --token XXX  # NASA Black Marble VNP46A4 2024, 2 tiles, cropped to Assam
    python fetch_public_data.py status             # what has been fetched so far
Every download is logged to manifest.jsonl (url, time, bytes, sha256, rows).
Only the standard library + pandas/pyarrow are needed, except: osm -> `pip install osmium`, viirs -> `pip install h5py`.
"""
import argparse, csv, datetime as dt, gzip, hashlib, io, json, os, sys, time, urllib.error, urllib.parse, urllib.request
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCOPE = os.path.join(HERE, 'scope'); RAW = os.path.join(HERE, 'raw'); OUT = os.path.join(HERE, 'data')
for d in (RAW, OUT): os.makedirs(d, exist_ok=True)
UA = {'User-Agent': 'RiskHorizon-IITG-hackathon/1.0 (student research; contact via project)'}
SCOPEJ = json.load(open(os.path.join(SCOPE, 'scope.json')))


def log(source, url, nbytes, sha, rows=None, note=''):
    with open(os.path.join(HERE, 'manifest.jsonl'), 'a') as f:
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


# ------------------------------------------------------------------ weather
def parse_openmeteo(js, cell_id):
    hdat = js['hourly']
    df = pd.DataFrame(hdat).rename(columns={'time': 'time_utc'})
    df['time_utc'] = pd.to_datetime(df['time_utc'])
    df.insert(0, 'cell_id', cell_id)
    df['grid_lat'] = js.get('latitude'); df['grid_lon'] = js.get('longitude')
    return df


def weather(args):
    plan = pd.read_csv(os.path.join(SCOPE, 'weather_requests.csv'))
    d = os.path.join(RAW, 'openmeteo'); os.makedirs(d, exist_ok=True)
    vars_ = ','.join(SCOPEJ['openmeteo_variables'])
    need = pd.read_parquet(os.path.join(SCOPE, 'weather_needed_hours.parquet'))  # only case/control hours + 3 h before
    need = {c: set(g.utc_hour) for c, g in need.groupby('cell_id')}
    done = 0
    for r in plan.itertuples():
        fn = os.path.join(d, f'{r.cell_id}_{r.start}_{r.end}.parquet')
        if os.path.exists(fn): continue
        url = ('https://archive-api.open-meteo.com/v1/archive?' + urllib.parse.urlencode(dict(
            latitude=r.lat, longitude=r.lon, start_date=r.start, end_date=r.end, hourly=vars_, models='era5', timezone='GMT')))
        try:
            b = get(url)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f'Open-Meteo quota reached after {done} requests this run. Re-run later (hourly limit resets each hour, daily at 00:00 UTC).')
                return
            raise
        js = json.loads(b)
        df = parse_openmeteo(js, r.cell_id)
        n_all = len(df)
        df = df[df.time_utc.isin(need[r.cell_id])]  # keep ONLY the hours the case-crossover table uses
        df.to_parquet(fn, index=False)
        log('open-meteo-era5', url, len(b), hashlib.sha256(b).hexdigest(), len(df), f'{r.cell_id} weight~{r.est_weight}; kept {len(df)}/{n_all} hours')
        done += 1
        print(f'[{done}] {r.cell_id} {r.start}..{r.end} rows={len(df)}')
        time.sleep(max(1.0, r.est_weight * 3600 / 4800))  # stay under 5,000 weighted calls / hour
    print('weather: all planned requests fetched')
    combine('openmeteo', 'weather_era5_hourly.parquet', dedupe=['cell_id', 'time_utc'])


# ------------------------------------------------------------------ METAR
METAR_FIELDS = ['tmpf', 'dwpf', 'relh', 'sknt', 'vsby', 'wxcodes', 'skyc1', 'skyl1', 'metar']


def parse_iem(text, icao):
    rows = [l for l in text.splitlines() if l and not l.startswith('#')]
    df = pd.read_csv(io.StringIO('\n'.join(rows)), na_values=['M'], low_memory=False)
    df = df.rename(columns={'valid': 'time_utc', 'station': 'station_raw'})
    df['time_utc'] = pd.to_datetime(df['time_utc'])
    df.insert(0, 'icao', icao)
    if 'vsby' in df: df['visibility_m'] = (pd.to_numeric(df.vsby, errors='coerce') * 1609.34).round(0)
    return df


def metar(args):
    st = pd.read_csv(os.path.join(SCOPE, 'metar_stations.csv'))
    st = st[st.keep]
    need = pd.read_parquet(os.path.join(SCOPE, 'metar_needed_hours.parquet'))
    need = {k: set(g.utc_hour) for k, g in need.groupby('icao')}
    d = os.path.join(RAW, 'metar'); os.makedirs(d, exist_ok=True)
    s0, e0 = SCOPEJ['weather_window']
    for icao in st.icao:
        for yr in range(int(s0[:4]), int(e0[:4]) + 1):
            fn = os.path.join(d, f'{icao}_{yr}.parquet')
            if os.path.exists(fn): continue
            a = max(pd.Timestamp(s0), pd.Timestamp(f'{yr}-01-01')); b_ = min(pd.Timestamp(e0) + pd.Timedelta(days=1), pd.Timestamp(f'{yr + 1}-01-01'))
            q = [('station', icao)] + [('data', f) for f in METAR_FIELDS] + [
                ('year1', a.year), ('month1', a.month), ('day1', a.day), ('year2', b_.year), ('month2', b_.month), ('day2', b_.day),
                ('tz', 'Etc/UTC'), ('format', 'onlycomma'), ('latlon', 'no'), ('missing', 'M'), ('trace', 'T'), ('direct', 'no'),
                ('report_type', '3'), ('report_type', '4')]
            url = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?' + urllib.parse.urlencode(q)
            b = get(url, timeout=300)
            df = parse_iem(b.decode('utf-8', 'replace'), icao)
            n_all = len(df)
            df = df[df.time_utc.dt.floor('h').isin(need.get(icao, set()))]  # keep only reports in needed hours
            df.to_parquet(fn, index=False)
            log('iem-metar', url, len(b), hashlib.sha256(b).hexdigest(), len(df), f'{icao} {yr}; kept {len(df)}/{n_all} reports')
            print(f'{icao} {yr}: {len(df)} reports')
            time.sleep(5)  # IEM asks for polite, sequential requests
    combine('metar', 'metar_assam_airports.parquet', dedupe=['icao', 'time_utc', 'metar'])


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


# ------------------------------------------------------------------ VIIRS Black Marble (street-light proxy)
def viirs(args):
    import h5py, numpy as np
    if not args.token: sys.exit('Need a NASA Earthdata token: https://urs.earthdata.nasa.gov -> Generate Token; pass --token')
    bb = SCOPEJ['osm_bbox']; hdr = {'Authorization': f'Bearer {args.token}'}
    d = os.path.join(RAW, 'viirs'); os.makedirs(d, exist_ok=True)
    year = 2024
    base = f'https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/VNP46A4/{year}/001/'
    try:
        listing = json.loads(get(base + '.json', hdr))
    except Exception:
        base = base.replace('/5200/', '/5000/'); listing = json.loads(get(base + '.json', hdr))
    items = listing['content'] if isinstance(listing, dict) else listing
    parts = []
    for tile in SCOPEJ['viirs_tiles']:
        name = next(i['name'] for i in items if f'.{tile}.' in i['name'] and i['name'].endswith('.h5'))
        fn = os.path.join(d, name)
        if not os.path.exists(fn):
            b = get(base + name, hdr, timeout=1800); open(fn, 'wb').write(b)
            log('nasa-blackmarble', base + name, len(b), hashlib.sha256(b).hexdigest(), None, tile)
        h_, v_ = int(tile[1:3]), int(tile[4:6])
        top, left, step = 90 - v_ * 10, -180 + h_ * 10, 10 / 2400
        with h5py.File(fn, 'r') as f:
            grp = f['HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields']
            key = 'AllAngle_Composite_Snow_Free'
            ds = grp[key]; arr = ds[()].astype('float64')
            fill = ds.attrs.get('_FillValue', [65535])[0]; sc = ds.attrs.get('scale_factor', [1.0])[0]; off = ds.attrs.get('add_offset', [0.0])[0]
            qn = 'AllAngle_Composite_Snow_Free_Num'
            num = grp[qn][()] if qn in grp else None
        lat = top - (np.arange(2400) + 0.5) * step; lon = left + (np.arange(2400) + 0.5) * step
        ri = np.where((lat >= bb['south']) & (lat <= bb['north']))[0]; ci = np.where((lon >= bb['west']) & (lon <= bb['east']))[0]
        if len(ri) == 0 or len(ci) == 0: continue
        sub = arr[np.ix_(ri, ci)]; valid = sub != fill
        LA, LO = np.meshgrid(lat[ri], lon[ci], indexing='ij')
        df = pd.DataFrame(dict(lat=LA[valid].round(5), lon=LO[valid].round(5), radiance_nw=(sub[valid] * sc + off).astype('float32')))
        if num is not None: df['n_obs'] = num[np.ix_(ri, ci)][valid]
        df['tile'] = tile; parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(os.path.join(OUT, f'viirs_vnp46a4_{year}_assam.parquet'), index=False)
    log('viirs-crop', base, None, None, len(out), f'{year} bbox crop')
    print('VIIRS pixels kept', len(out))


# ------------------------------------------------------------------ helpers
def combine(sub, outname, dedupe):
    d = os.path.join(RAW, sub)
    fs = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith('.parquet')]
    if not fs: return
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).drop_duplicates(dedupe)
    df.to_parquet(os.path.join(OUT, outname), index=False)
    print(f'combined -> data/{outname}: {len(df):,} rows')


def status(args):
    plan = pd.read_csv(os.path.join(SCOPE, 'weather_requests.csv'))
    got = set(os.listdir(os.path.join(RAW, 'openmeteo'))) if os.path.isdir(os.path.join(RAW, 'openmeteo')) else set()
    plan['done'] = [f'{r.cell_id}_{r.start}_{r.end}.parquet' in got for r in plan.itertuples()]
    cells = pd.read_csv(os.path.join(SCOPE, 'era5_cells.csv'))
    donecells = plan.groupby('cell_id').done.all()
    share = cells[cells.cell_id.isin(donecells[donecells].index)].n_crashes.sum() / cells.n_crashes.sum()
    print(f'weather: {plan.done.sum()}/{len(plan)} requests, cells complete cover {share:.1%} of crashes')
    for f in sorted(os.listdir(OUT)): print('data/', f, os.path.getsize(os.path.join(OUT, f)) // 1024, 'KB')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('what', choices=['weather', 'metar', 'osm', 'viirs', 'status'])
    p.add_argument('--token'); p.add_argument('--keep-pbf', action='store_true')
    a = p.parse_args(); globals()[a.what](a)
