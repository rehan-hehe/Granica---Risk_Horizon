"""METAR airport weather (visibility/fog) — 6 Assam airports, only needed hours kept.
Standalone: needs only this file + the scope files listed below, so it can run on any PC.
Run from the Risk_Horizon_Data folder:  python fetch_metar.py 
Resumable: re-running skips anything already saved. Log: manifest_metar.jsonl
Needs scope/: scope.json, metar_stations.csv, metar_needed_hours.parquet
"""
import argparse, datetime as dt, gzip, hashlib, io, json, os, sys, time, urllib.error, urllib.parse, urllib.request
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCOPE = os.path.join(HERE, 'scope'); RAW = os.path.join(HERE, 'raw'); OUT = os.path.join(HERE, 'data')
for d in (RAW, OUT): os.makedirs(d, exist_ok=True)
UA = {'User-Agent': 'RiskHorizon-IITG-hackathon/1.0 (student research)'}
SCOPEJ = json.load(open(os.path.join(SCOPE, 'scope.json')))
MANIFEST = os.path.join(HERE, 'manifest_metar.jsonl')


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


def combine(sub, outname, dedupe):
    d = os.path.join(RAW, sub)
    fs = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith('.parquet')]
    if not fs: return
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).drop_duplicates(dedupe)
    df.to_parquet(os.path.join(OUT, outname), index=False)
    print(f'combined -> data/{outname}: {len(df):,} rows')



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    
    args = p.parse_args()
    metar(args)
