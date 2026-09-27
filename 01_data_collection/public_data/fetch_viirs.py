"""NASA Black Marble VNP46A4 2024 night lights, cropped to Assam (street-lighting proxy). Needs: pip install h5py
Standalone: needs only this file + the scope files listed below, so it can run on any PC.
Run from the Risk_Horizon_Data folder:  python fetch_viirs.py --token EARTHDATA_TOKEN
Resumable: re-running skips anything already saved. Log: manifest_viirs.jsonl
Needs scope/: scope.json
"""
import argparse, datetime as dt, gzip, hashlib, io, json, os, sys, time, urllib.error, urllib.parse, urllib.request
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCOPE = os.path.join(HERE, 'scope'); RAW = os.path.join(HERE, 'raw'); OUT = os.path.join(HERE, 'data')
for d in (RAW, OUT): os.makedirs(d, exist_ok=True)
UA = {'User-Agent': 'RiskHorizon-IITG-hackathon/1.0 (student research)'}
SCOPEJ = json.load(open(os.path.join(SCOPE, 'scope.json')))
MANIFEST = os.path.join(HERE, 'manifest_viirs.jsonl')


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




EXTRA_TRUSTED_HOSTS = set()  # tests only


def h5_ok(fn):
    try:
        import h5py
        with h5py.File(fn, 'r') as f:
            f['HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields']
        return True
    except Exception:
        return False


def download_earthdata(url, fn, token):
    """NASA downloads redirect through Earthdata Login and back. Keep the token on every NASA hop
    and keep cookies across redirects (plain urllib drops both, which gives 401)."""
    import requests
    class S(requests.Session):
        def rebuild_auth(self, prep, resp):  # keep Authorization while we stay on *.nasa.gov
            host = urllib.parse.urlparse(prep.url).hostname or ''
            if host.endswith('nasa.gov') or host in EXTRA_TRUSTED_HOSTS:
                prep.headers['Authorization'] = f'Bearer {token}'
            elif 'Authorization' in prep.headers:
                del prep.headers['Authorization']
    part = fn + '.part'
    with S() as s:
        s.headers.update({'Authorization': f'Bearer {token}', **UA})
        total = None
        for attempt in range(8):  # NASA connections sometimes drop mid-file: resume from where it stopped
            have = os.path.getsize(part) if os.path.exists(part) else 0
            hdr = {'Range': f'bytes={have}-'} if have else {}
            try:
                r = s.get(url, stream=True, timeout=300, allow_redirects=True, headers=hdr)
                if r.status_code in (401, 403):
                    raise PermissionError(f'NASA refused the download (HTTP {r.status_code}). Check: token copied fully and not expired; '
                                          'you have logged in once at https://ladsweb.modaps.eosdis.nasa.gov with the same account.')
                if 'text/html' in r.headers.get('Content-Type', ''):
                    raise PermissionError('NASA returned a login page instead of the file (token not accepted).')
                r.raise_for_status()
                if have and r.status_code != 206:  # server ignored Range: start again
                    have = 0; open(part, 'wb').close()
                cl = r.headers.get('Content-Length')
                if cl is not None: total = have + int(cl)
                with open(part, 'ab' if have else 'wb') as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        got = f.tell()
                        print(f'\r  {got / 1e6:,.0f} / {total / 1e6 if total else 0:,.0f} MB', end='', flush=True)
            except PermissionError:
                raise
            except Exception as e:
                print(f'\n  connection dropped ({type(e).__name__}); resuming (attempt {attempt + 2}/8) ...')
                time.sleep(5); continue
            got = os.path.getsize(part)
            if total is None or got >= total:
                break
            print(f'\n  only {got:,} of {total:,} bytes arrived; resuming (attempt {attempt + 2}/8) ...')
        got = os.path.getsize(part)
        if total is not None and got < total:
            raise RuntimeError(f'download incomplete after 8 attempts ({got:,}/{total:,} bytes). Run run_viirs.bat again to continue.')
        print()
        h = hashlib.sha256()
        with open(part, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
        os.replace(part, fn)
        return got, h.hexdigest()

# ------------------------------------------------------------------ VIIRS Black Marble (street-light proxy)
def viirs(args):
    import h5py, numpy as np
    if not args.token: sys.exit('Need a NASA Earthdata token: https://urs.earthdata.nasa.gov -> Generate Token; pass --token')
    bb = SCOPEJ['osm_bbox']; hdr = {'Authorization': f'Bearer {args.token}'}
    d = os.path.join(RAW, 'viirs'); os.makedirs(d, exist_ok=True)
    year = 2024
    if os.path.exists(os.path.join(OUT, f'viirs_vnp46a4_{year}_assam.parquet')):
        print('viirs: already done - skipping'); return
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
        if os.path.exists(fn) and not h5_ok(fn):
            print(f'{name}: existing file is incomplete/corrupt - deleting and downloading again')
            os.remove(fn)
        if not os.path.exists(fn):
            print(f'downloading {name} ...', flush=True)
            try:
                nbytes, sha = download_earthdata(base + name, fn, args.token)
            except PermissionError as e:
                print(f'\n{e}\nFallback: while logged in to Earthdata in your browser, open\n  {base + name}\n'
                      f'save the file into  {d}\nthen run run_viirs.bat again (it will use the saved file).')
                sys.exit(1)
            log('nasa-blackmarble', base + name, nbytes, sha, None, tile)
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
    if not getattr(args, 'keep_raw', False):  # the 150 MB tiles are no longer needed once Assam is cropped out
        for f_ in os.listdir(d):
            if f_.endswith('.h5'): os.remove(os.path.join(d, f_))
    log('viirs-crop', base, None, None, len(out), f'{year} bbox crop')
    print('VIIRS pixels kept', len(out))


def combine(sub, outname, dedupe):
    d = os.path.join(RAW, sub)
    fs = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith('.parquet')]
    if not fs: return
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).drop_duplicates(dedupe)
    df.to_parquet(os.path.join(OUT, outname), index=False)
    print(f'combined -> data/{outname}: {len(df):,} rows')



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--token'); p.add_argument('--keep-raw', action='store_true')
    args = p.parse_args()
    viirs(args)
