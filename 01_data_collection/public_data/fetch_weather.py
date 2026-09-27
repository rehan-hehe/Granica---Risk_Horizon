"""Open-Meteo ERA5 hourly weather — only the crash/control hours (+3 h before) in the 150 crash cells.
Standalone: needs only this file + the scope files listed below, so it can run on any PC.
Run from the Risk_Horizon_Data folder:  python fetch_weather.py 
Resumable: re-running skips anything already saved. Log: manifest_weather.jsonl
Needs scope/: scope.json, era5_cells.csv, weather_requests.csv, weather_needed_hours.parquet
"""
import argparse, datetime as dt, gzip, hashlib, io, json, os, sys, time, urllib.error, urllib.parse, urllib.request
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCOPE = os.path.join(HERE, 'scope'); RAW = os.path.join(HERE, 'raw'); OUT = os.path.join(HERE, 'data')
for d in (RAW, OUT): os.makedirs(d, exist_ok=True)
UA = {'User-Agent': 'RiskHorizon-IITG-hackathon/1.0 (student research)'}
SCOPEJ = json.load(open(os.path.join(SCOPE, 'scope.json')))
MANIFEST = os.path.join(HERE, 'manifest_weather.jsonl')


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
        b = None
        for wait_round in range(9):  # on quota: pause 15 min and retry (hourly limit resets), give up after ~2 h
            try:
                b = get(url); break
            except urllib.error.HTTPError as e:
                if e.code != 429: raise
                print(f'  quota hit after {done} requests this run - pausing 15 min (attempt {wait_round + 1}/9) ...', flush=True)
                time.sleep(900)
        if b is None:
            print('Open-Meteo daily quota reached. Re-run later: python fetch_public_data.py weather (it resumes).'); return
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


def combine(sub, outname, dedupe):
    d = os.path.join(RAW, sub)
    fs = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith('.parquet')]
    if not fs: return
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).drop_duplicates(dedupe)
    df.to_parquet(os.path.join(OUT, outname), index=False)
    print(f'combined -> data/{outname}: {len(df):,} rows')



def status():
    plan = pd.read_csv(os.path.join(SCOPE, 'weather_requests.csv'))
    d = os.path.join(RAW, 'openmeteo'); got = set(os.listdir(d)) if os.path.isdir(d) else set()
    plan['done'] = [f'{r.cell_id}_{r.start}_{r.end}.parquet' in got for r in plan.itertuples()]
    cells = pd.read_csv(os.path.join(SCOPE, 'era5_cells.csv'))
    dc = plan.groupby('cell_id').done.all(); share = cells[cells.cell_id.isin(dc[dc].index)].n_crashes.sum() / cells.n_crashes.sum()
    print(f'weather: {plan.done.sum()}/{len(plan)} requests done; complete cells cover {share:.1%} of crashes')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--status', action='store_true')
    args = p.parse_args()
    status() if args.status else weather(args)
