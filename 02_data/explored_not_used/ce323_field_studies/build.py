"""Build the consolidated CE323 lab dataset from the Lab Report folder.
Run: python3 build.py  -> out/CE323_Lab_Dataset.xlsx + out/parquet/*.parquet
Principles: raw values kept verbatim; recomputed values in separate columns; every
row carries source_file + source_location + extraction_method; issues logged, never silently fixed.
"""
import csv, hashlib, os, re, datetime as dt
import numpy as np, pandas as pd, openpyxl, pdfplumber
import data_manual as M

SRC = os.environ.get('LAB_DIR', '../Lab Report') + '/'
OUT = 'out'
os.makedirs(OUT + '/parquet', exist_ok=True)

G5, G12, AK = 'Group 5', 'Group 12', 'Individual (Ayush Kumar)'
CLASS_STD = {'car': 'Car', 'lcv': 'LCV', 'truck': 'Truck', 'two-wheeler': 'Two Wheeler', 'two wheeler': 'Two Wheeler',
             'bus': 'Bus', 'buses': 'Bus', 'auto': 'Auto', 'ma': 'Multi Axle', 'multi axle': 'Multi Axle',
             'bike': 'Two Wheeler', 'scooty': 'Two Wheeler', '2 wheeler': 'Two Wheeler', 'cycle': 'Cycle',
             'cycli': 'Cycle', 'e- rickshaw': 'E-Rickshaw', 'trucks': 'Truck', 'cars': 'Car'}
def std(c):
    return CLASS_STD.get(str(c).strip().lower(), str(c).strip()) if c is not None else None

ISSUES = []
def issue(sev, grp, exp, where, desc, evidence='', action='Kept as reported; flagged'):
    ISSUES.append(dict(issue_id=f'V{len(ISSUES)+1:03d}', severity=sev, group=grp, experiment=exp, location=where,
                       description=desc, evidence=evidence, action=action))

REPORTED = []
def rep(grp, exp, param, value, unit='', subgroup='', src='', loc='', method='', recomputed=None, note=''):
    match = ''
    if recomputed is not None and isinstance(value, (int, float)) and value is not None:
        tol = max(0.02, abs(value) * 0.005)
        match = 'MATCH' if abs(value - recomputed) <= tol else 'MISMATCH'
    REPORTED.append(dict(group=grp, experiment=exp, parameter=param, subgroup=subgroup, reported_value=value, unit=unit,
                         recomputed_value=None if recomputed is None else round(float(recomputed), 4), check=match,
                         source_file=src, source_location=loc, extraction_method=method, note=note))

def pctl(x, q, method='linear'):  # 'linear' = Excel PERCENTILE.INC; 'weibull' = PERCENTILE.EXC
    return float(np.percentile(np.asarray(x, float), q, method=method))

# ============================================================== Exp 3/4  Group 5 per-vehicle video data
F3 = 'Bhavesh_Traffic volume video_analysis_datasheet_1.xlsx'
def blk(sheet, hdr, c0):
    d = pd.read_excel(SRC + F3, sheet, header=hdr).iloc[:, c0:c0 + 8]
    d.columns = ['vehicle_id', 'lane', 'class_raw', 't_in_s', 't_out_s', 'headway_reported_s', 'free_flow_reported', 'speed_reported_kmh']
    d = d.dropna(subset=['vehicle_id']); d['sheet'] = sheet; d['cols'] = f'{openpyxl.utils.get_column_letter(c0+1)}-{openpyxl.utils.get_column_letter(c0+8)}'
    return d
lanes = pd.concat([blk('Shoulder lane', 2, 0), blk('Shoulder lane', 2, 9), blk('Median lane', 1, 0), blk('Median lane', 1, 9)], ignore_index=True)
raw = pd.read_excel(SRC + F3, 'Speed (free & non free)', header=1).iloc[:, 0:5]
raw.columns = ['vehicle_id', 'lane_rawsheet', 'class_rawsheet', 't_in_rawsheet', 't_out_rawsheet']
raw = raw.dropna(subset=['vehicle_id'])
raw['order_in_rawsheet'] = range(1, len(raw) + 1)
v = raw.merge(lanes, on='vehicle_id', how='left', validate='1:1')
assert v.sheet.notna().all() and len(v) == 321
v['lane'] = v['lane'].str.strip()
v['class_std'] = v['class_raw'].map(std)
v['section_length_m'] = 50.0
v['travel_time_s'] = (v.t_out_s - v.t_in_s).round(4)
v['speed_kmh_recomputed'] = (50 / v.travel_time_s * 3.6).round(3)
v['speed_matches_sheet'] = (v.speed_kmh_recomputed - v.speed_reported_kmh).abs() < 0.01
v['t_out_edited_between_sheets'] = (v.t_out_rawsheet - v.t_out_s).abs() > 1e-6
v['t_in_edited_between_sheets'] = (v.t_in_rawsheet - v.t_in_s).abs() > 1e-6
v['speed_from_rawsheet_kmh'] = (50 / (v.t_out_rawsheet - v.t_in_rawsheet) * 3.6).round(3)
# recompute lane headways (in-time difference to the leading vehicle in the same lane)
v = v.sort_values(['lane', 't_in_s', 'order_in_rawsheet']).reset_index(drop=True)
v['leader_id'] = v.groupby('lane').vehicle_id.shift(1)
v['leader_class'] = v.groupby('lane').class_std.shift(1)
v['headway_to_leader_s'] = (v.t_in_s - v.groupby('lane').t_in_s.shift(1)).round(3)
v['headway_to_follower_s'] = (v.groupby('lane').t_in_s.shift(-1) - v.t_in_s).round(3)
v['pair_follower_leader'] = np.where(v.leader_class.notna(), v.class_std + '–' + v.leader_class.fillna(''), None)
v['free_flow_recomputed_gt5s'] = v.headway_to_follower_s > 5  # sheet convention: headway column = gap to NEXT vehicle
def plaus(r):
    if r.speed_kmh_recomputed > 120: return 'IMPLAUSIBLE (>120 km/h on urban mid-block)'
    if r.speed_kmh_recomputed > 90: return 'CHECK (90-120 km/h)'
    if r.speed_kmh_recomputed < 5: return 'IMPLAUSIBLE (<5 km/h)'
    return 'OK'
v['speed_plausibility'] = v.apply(plaus, axis=1)
v.insert(0, 'group', G5); v.insert(1, 'experiments', 'Exp3 (speed/volume) + Exp4 (headway)')
v['source_file'] = F3
v['source_location'] = v.sheet + ' ' + v.cols + ' (final) | Speed (free & non free) A-E (raw entry)'
v['extraction_method'] = 'xlsx cell values (openpyxl, data_only)'
E3V = v[['group', 'experiments', 'vehicle_id', 'lane', 'class_raw', 'class_std', 't_in_s', 't_out_s', 't_in_rawsheet', 't_out_rawsheet',
         't_in_edited_between_sheets', 't_out_edited_between_sheets', 'section_length_m', 'travel_time_s', 'speed_reported_kmh',
         'speed_kmh_recomputed', 'speed_from_rawsheet_kmh', 'speed_plausibility', 'headway_reported_s', 'free_flow_reported',
         'headway_to_leader_s', 'headway_to_follower_s', 'free_flow_recomputed_gt5s', 'leader_id', 'pair_follower_leader',
         'order_in_rawsheet', 'source_file', 'source_location', 'extraction_method']].copy()
n_edit = int(v.t_out_edited_between_sheets.sum())
issue('HIGH', G5, 'Exp3/4', F3 + ': Speed (free & non free) vs lane sheets',
      f'{n_edit} of 321 vehicles have a different OUT time in the raw-entry sheet than in the analysis sheets (mostly two-wheelers, typically +1 to +5 s). Raw-entry values give speeds up to '
      f'{v.speed_from_rawsheet_kmh.max():.0f} km/h; analysis-sheet values give max {v.speed_kmh_recomputed.max():.0f} km/h.',
      'e.g. vehicle 9582: out 102.54 (raw) vs 107.54 (analysis) -> 2571 vs 35.5 km/h',
      'Analysis-sheet times used as primary (t_out_s); raw-entry time kept in t_out_rawsheet; both speeds given.')
bad = v[v.speed_plausibility != 'OK']
issue('MEDIUM', G5, 'Exp3', F3, f'{len(bad)} vehicles have speeds outside 5-90 km/h even after the sheet corrections (50 m trap).',
      '; '.join(f'{int(r.vehicle_id)} {r.class_std} {r.speed_kmh_recomputed:.1f}' for r in bad.itertuples()), 'Flagged in speed_plausibility; not removed')
hw_mis = (v.headway_to_follower_s.fillna(-1) - v.headway_reported_s.fillna(-1)).abs() > 0.02
issue('MEDIUM', G5, 'Exp3', F3 + ' lane sheets col F/O', f'Reported "time headway" disagrees with recomputed in-time gap for {int(hw_mis.sum())} of 321 vehicles; the reported free-flow flag therefore does not always follow the >5 s rule.',
      'Headway recomputed from sorted in-times per lane (headway_to_follower_s, headway_to_leader_s)', 'Both kept')
ff_mis = (v.free_flow_reported.astype(str).str.strip().str.lower().isin(['yes', 'true']) != v.free_flow_recomputed_gt5s)
if ff_mis.sum(): issue('LOW', G5, 'Exp3', F3, f'Free-flow flag differs from recomputed (>5 s to next vehicle) for {int(ff_mis.sum())} vehicles.', '', 'Both kept')

# Group 5 traffic-volume sheet (2-min classified counts)
wb3 = openpyxl.load_workbook(SRC + F3, data_only=True); ws = wb3['Traffic Volume']
vol_rows = []
cls5 = ['Multi Axle', 'Truck', 'LCV', 'Bus', 'Car', 'Auto', 'Two Wheeler']
PCU5 = {'Multi Axle': 4.5, 'Truck': 3.5, 'LCV': 2.2, 'Bus': 3.5, 'Car': 1.0, 'Auto': 2.0, 'Two Wheeler': 0.5, 'Cycle': None}
for lane, c0 in [('Shoulder (Lane 1)', 1), ('Median (Lane 2)', 13)]:
    for r in range(4, 9):
        f, t = ws.cell(r, c0).value, ws.cell(r, c0 + 1).value
        pcu_rep = ws.cell(r, c0 + 9).value
        cnts = [ws.cell(r, c0 + 2 + k).value for k in range(7)]
        pcu_re = sum(c * PCU5[k] for c, k in zip(cnts, cls5))
        for k, c in zip(cls5, cnts):
            vol_rows.append(dict(group=G5, experiment='Exp3', site='Mid-block (video), site not named in report', lane=lane,
                                 interval_min=f'{f}-{t}', vehicle_class=k, count=c, pcu_factor=PCU5[k], pcu=c * PCU5[k],
                                 interval_pcu_reported=pcu_rep, interval_pcu_recomputed=round(pcu_re, 2),
                                 source_file=F3, source_location=f'Traffic Volume r{r}', extraction_method='xlsx'))
E3VOL = pd.DataFrame(vol_rows)
for lane, g in E3VOL.groupby('lane'):
    rep(G5, 'Exp3', 'Total vehicles in 10 min', int(g['count'].sum()), 'veh', lane, F3, 'Traffic Volume row 9', 'xlsx')
    rep(G5, 'Exp3', 'Total equivalent flow in 10 min (report text says "178 PCU"/"150 PCU")', 178 if 'Shoulder' in lane else 150, 'PCU', lane,
        'Transport Lab 3.docx', '§4.1', 'docx text', recomputed=g.pcu.sum(),
        note='Report value equals the VEHICLE count; PCU sum is larger')
issue('HIGH', G5, 'Exp3', 'Transport Lab 3.docx §4.1 / xlsx Traffic Volume row 9',
      'Report states total equivalent flow 178 PCU (shoulder) and 150 PCU (median); these are vehicle counts. PCU totals from the same sheet are 206.2 and 250.3. Composition shares in row 10 (e.g. 44.94% 2W) are vehicle shares, not "PCU flow" shares as the text says.',
      'J9=178 but SUM(J4:J8)=206.2; V9=150 but SUM(V4:V8)=250.3', 'Recomputed PCU given alongside')
rep(G5, 'Exp3', 'PHF (2-min basis)', 0.8118, '-', 'Shoulder (Lane 1)', F3, 'Traffic Volume C37', 'xlsx', recomputed=206.2 / (5 * 50.8))
rep(G5, 'Exp3', 'PHF (2-min basis)', 0.934, '-', 'Median (Lane 2)', F3, 'Traffic Volume F37', 'xlsx', recomputed=250.3 / (5 * 53.6))
rep(G5, 'Exp3', 'PHF (report text)', 0.811, '-', 'Shoulder', 'Transport Lab 3.docx', '§4.1', 'docx text')
rep(G5, 'Exp3', 'PHF (report text)', 0.933, '-', 'Median', 'Transport Lab 3.docx', '§4.1', 'docx text')
cnt_speed = E3V.class_std.value_counts(); cnt_vol = E3VOL.groupby('vehicle_class')['count'].sum()
issue('LOW', G5, 'Exp3', F3, f'Per-vehicle speed list has 321 vehicles; 2-min volume sheet counts 328 ({ {k:int(x) for k,x in cnt_vol.items()} }) vs speed list ({ {k:int(x) for k,x in cnt_speed.items()} }). Counts come from different passes of the same video.', '', 'Both kept')
# reported free-flow / stream stats from lane sheets
for lane, sheet, cells in [('Shoulder', 'Shoulder lane', [('Car', 'U54', 'U55', 'U56'), ('Truck', 'X54', 'X55', 'X56'), ('LCV', 'U59', 'U60', 'U61'), ('Two Wheeler', 'X59', 'X60', 'X61'), ('Auto', 'V64', 'V65', 'V66')])]:
    ws = wb3[sheet]
    for cls, a, b, c in cells:
        ff = E3V[(E3V.lane == lane) & (E3V.class_std == cls) & E3V.free_flow_reported.astype(str).str.lower().isin(['yes', 'true'])].speed_kmh_recomputed
        # sheets' free-flow blocks
        rep(G5, 'Exp3', 'Free-flow speed mean', round(ws[a].value, 4), 'km/h', f'{lane} {cls}', F3, f'{sheet}!{a}', 'xlsx', recomputed=ff.mean() if len(ff) else None, note=f'n_free(flag)={len(ff)}')
        rep(G5, 'Exp3', 'Free-flow speed p85', round(ws[b].value, 4), 'km/h', f'{lane} {cls}', F3, f'{sheet}!{b}', 'xlsx', recomputed=pctl(ff, 85) if len(ff) else None)
        rep(G5, 'Exp3', 'Free-flow speed p50', round(ws[c].value, 4), 'km/h', f'{lane} {cls}', F3, f'{sheet}!{c}', 'xlsx', recomputed=pctl(ff, 50) if len(ff) else None)
for sheet, cell_mean, cell_sd, cell_chi, lbl in [('Median lane', 'AL10', 'AL11', 'AQ15', 'Median ? (block AK)'), ('Median lane', 'AC18', 'AC19', 'AH24', 'Median ? (block AB)'), ('Median lane', 'T25', 'T26', 'Y34', 'Median Car?')]:
    pass  # median-lane blocks are unlabeled in-sheet; captured via chi-square list below
chis = [('Shoulder', 'Car', 'free', 'Shoulder lane', 'X25'), ('Shoulder', 'Truck', 'free', 'Shoulder lane', 'AG18'), ('Shoulder', 'LCV', 'free', 'Shoulder lane', 'AP19'),
        ('Shoulder', 'Two Wheeler', 'free', 'Shoulder lane', 'AY17'), ('Shoulder', 'Car', 'stream', 'Speed Stream Ana', 'Z73'), ('Median', 'Car', 'stream', 'Speed Stream Analysis', 'Y88')]
for lane, cls, kind, sh, cell in chis:
    rep(G5, 'Exp3', f'Chi-square (normal fit), {kind}', round(wb3[sh][cell].value, 4), '-', f'{lane} {cls}', F3, f'{sh}!{cell}', 'xlsx')
rep(G5, 'Exp3', 'Chi-square calc (report, "free-flow" car normality)', 11.864, '-', 'Shoulder Car', 'Transport Lab 3.docx', '§4.2 B', 'docx text', note='dof 6, crit 12.592, accepted')
rep(G5, 'Exp3', 'Chi-square calc (report, "free-flow" car normality)', 15.4109, '-', 'Median Car', 'Transport Lab 3.docx', '§4.2 B', 'docx text', note='dof 4, crit 9.488, rejected; value equals the STREAM sheet (Y88)')
issue('LOW', G5, 'Exp3', 'Transport Lab 3.docx §4.2B', 'Chi-square values quoted as "free-flow" normality (11.864, 15.4109) sit in the stream-speed sheets of the workbook (Speed Stream Ana!Z73 = 11.8645, Speed Stream Analysis!Y88 = 15.4109).', '', 'Recorded with note')

# ============================================================== Exp 4 Group 5 (headway) -- same vehicles; reported lane stats
F4 = 'Expt 4.xlsx'
w4 = openpyxl.load_workbook(SRC + F4, data_only=True)['General']
for lane, cm, n_cell in [('Shoulder', 'Z36', 'AC34'), ('Median', 'BA35', 'AT43')]:
    pass
sh = E3V[E3V.lane == 'Shoulder'].headway_to_leader_s.dropna(); md = E3V[E3V.lane == 'Median'].headway_to_leader_s.dropna()
rep(G5, 'Exp4', 'Mean time headway', round(w4['Z36'].value, 4), 's', 'Shoulder', F4, 'General!Z36', 'xlsx', recomputed=sh.mean(), note='recomputed from Exp3 in-times, gap to leader')
rep(G5, 'Exp4', 'Mean time headway', round(w4['BA35'].value, 4), 's', 'Median', F4, 'General!BA35', 'xlsx', recomputed=md.mean(), note='small difference: Expt 4 sheet uses raw-entry times / its own headway column (one median headway differs)')
rep(G5, 'Exp4', 'SD time headway', 3.584609, 's', 'Shoulder', 'transportation exp 4.docx', 'image84', 'image (read)', recomputed=sh.std())
rep(G5, 'Exp4', 'SD time headway', 4.401448274, 's', 'Median', 'transportation exp 4.docx', 'image85', 'image (read)', recomputed=md.std())
rep(G5, 'Exp4', 'Neg-exp lambda', 0.328532, '1/s', 'Shoulder', 'transportation exp 4.docx', 'image84', 'image (read)', recomputed=1 / sh.mean())
rep(G5, 'Exp4', 'Neg-exp lambda', 0.226392921, '1/s', 'Median', 'transportation exp 4.docx', 'image85', 'image (read)', recomputed=1 / md.mean())
rep(G5, 'Exp4', 'Total observations', w4['AC34'].value, 'veh', 'Shoulder', F4, 'General!AC34', 'xlsx', recomputed=(E3V.lane == 'Shoulder').sum())
rep(G5, 'Exp4', 'Total observations', w4['AT43'].value, 'veh', 'Median', F4, 'General!AT43', 'xlsx', recomputed=(E3V.lane == 'Median').sum())
rep(G5, 'Exp4', 'Best-fit headway distribution', 'Negative exponential (both lanes)', '', '', 'transportation exp 4.docx', '§5.5', 'docx text')
# check Expt4 per-vehicle list identical to Exp3 raw-entry sheet
g4 = pd.read_excel(SRC + F4, 'General', header=1).iloc[:, 0:5].dropna(subset=['Vehicle ID'])
same = (g4['Vehicle ID'].values == raw.vehicle_id.values).all() and np.allclose(g4['Out Time (s)'].values, raw.t_out_rawsheet.values)
issue('INFO', G5, 'Exp4', F4 + ' General A-E', f'Exp4 uses the same 321-vehicle video list as Exp3 (identical IDs and raw-entry times: {same}). Headway in Exp4 sheet = gap to leader; in Exp3 sheet = gap to follower.', '', 'Single per-vehicle table E3E4_G5_Vehicles serves both')
pairs = E3V.dropna(subset=['headway_to_leader_s']).groupby(['lane', 'pair_follower_leader']).headway_to_leader_s.agg(['count', 'mean', 'std', 'min', 'max']).round(3).reset_index()
pairs.insert(0, 'group', G5); pairs['note'] = 'Recomputed from per-vehicle in-times (not a reported table)'
E4PAIRS = pairs

# ============================================================== Exp 4 Group 12 (binned only)
F12_4 = 'Group12_exp4.pdf'
rows = []
for lane, freqs in M.G12_E4_BINS.items():
    n = sum(freqs)
    for i, f in enumerate(freqs):
        rows.append(dict(group=G12, experiment='Exp4', lane=lane, bin_lo_s=i * 0.5, bin_hi_s=i * 0.5 + 0.5, frequency=f,
                         rel_freq_recomputed=round(f / n, 6), source_file=F12_4, source_location='embedded image p.3-4',
                         extraction_method='image, read manually at native resolution'))
    exp_n = M.G12_E4_SUMMARY[lane]['n'] - 1
    if n != exp_n:
        issue('LOW', G12, 'Exp4', F12_4 + ' frequency table ' + lane, f'Binned frequencies sum to {n}; vehicle count {exp_n + 1} implies {exp_n} headways (table shown only to 16 s / truncated).', '', 'Kept')
E4BINS = pd.DataFrame(rows)
for lane, d in M.G12_E4_SUMMARY.items():
    for k, u in [('n', 'veh'), ('mean', 's'), ('sd', 's'), ('min', 's')]:
        rep(G12, 'Exp4', {'n': 'Vehicle count', 'mean': 'Mean headway', 'sd': 'SD headway', 'min': 'Min headway'}[k], d[k], u, lane, F12_4, 'p.3 table', 'pdf text')
for q, val in M.G12_E4_PCTL.items():
    rep(G12, 'Exp4', f'Headway p{q}', val, 's', 'Inner lane CDF', F12_4, 'p.4', 'pdf text')
issue('HIGH', G12, 'Exp4', F12_4 + ' p.4 percentiles', 'Reported 85th/95th percentile headways of 402.15 s and 527.83 s are impossible for a 10-min count with mean headway ~5 s (they look like cumulative arrival times); median 7.50 s also exceeds the lane maximum implied by the frequency table. Do not use.', '', 'Kept verbatim, flagged')
for r in M.G12_E4_GOF:
    rep(G12, 'Exp4', f'Chi-square {r[1]} (stat/crit/p/df)', r[4], '-', r[0], F12_4, 'image p.8 Result Summary', 'image (read)', note=f'crit {r[5]}, p {r[6]}, df {r[7]}, {r[8]}')
issue('MEDIUM', G12, 'Exp4', F12_4 + ' conclusion', 'Group 12 concludes the combined stream is Normal and rejects negative-exponential in all cases; Group 5 (same course, possibly same video type) concludes negative-exponential fits both lanes. Conflicting conclusions.', '', 'Both recorded')

# ============================================================== Exp 5 spot speed
F5 = 'Expt 5.xlsx'
w5 = openpyxl.load_workbook(SRC + F5, data_only=True).active
rows = []
for cls, c_nf, c_ff in [('Truck', 1, 2), ('Car', 5, 6), ('Auto', 9, 10), ('LCV', 13, 14), ('Bus', 17, 18), ('Two Wheeler', 21, 22)]:
    for c, state in [(c_nf, 'Non-free (car-following)'), (c_ff, 'Free flow')]:
        for r in range(3, 80):
            val = w5.cell(r, c).value
            if val is None or str(val).strip() == '': continue
            rows.append(dict(group=G5, experiment='Exp5', site='Amingaon, Guwahati (4-lane divided)', date='2026-02-21', vehicle_class=cls,
                             flow_state=state, speed_kmh=float(val), source_file=F5,
                             source_location=f'Sheet1!{openpyxl.utils.get_column_letter(c)}{r}', extraction_method='xlsx'))
# Group 12
p = pdfplumber.open(SRC + 'CE323_EXP5.pdf')
t = p.pages[3].extract_tables()[0]
cols = [('Truck', 'Free'), ('Truck', 'Forced'), ('Bus', 'Free'), ('Bus', 'Forced'), ('LCV', 'Free'), ('LCV', 'Forced'),
        ('Car', 'Free'), ('Car', 'Forced'), ('Auto', 'Free'), ('Auto', 'Forced'), ('Two Wheeler', 'Free'), ('Two Wheeler', 'Forced')]
g12_5 = {c: [] for c in cols}
for row in t[2:]:
    vals = [x for i, x in enumerate(row) if i != 6]
    for c, x in zip(cols, vals):
        if x not in (None, ''): g12_5[c].append(float(x))
for c, lst in M.G12_E5_P5.items(): g12_5[c] += [float(x) for x in lst]
for (cls, st), lst in g12_5.items():
    for i, x in enumerate(lst):
        rows.append(dict(group=G12, experiment='Exp5', site='Mid-block section (not named)', date='', vehicle_class=cls,
                         flow_state='Free flow' if st == 'Free' else 'Forced (car-following)', speed_kmh=x, source_file='CE323_EXP5.pdf',
                         source_location=f'Observation table p.4-5, col {cls}/{st}, entry {i+1}',
                         extraction_method='pdf table (pdfplumber) p.4; word x-positions p.5; verified against page image'))
E5 = pd.DataFrame(rows)
# Group 5 checks vs sheet stats
stat_cells = {'Truck': 'AA', 'Car': 'AH', 'Auto': 'AP', 'LCV': 'AX', 'Bus': 'BF', 'Two Wheeler': 'BN'}
for cls, col in stat_cells.items():
    ff = E5[(E5.group == G5) & (E5.vehicle_class == cls) & (E5.flow_state == 'Free flow')].speed_kmh
    al = E5[(E5.group == G5) & (E5.vehicle_class == cls)].speed_kmh
    for lab, rr, series in [('Free-flow', 14, ff), ('Combined (stream)', 49, al)]:
        rep(G5, 'Exp5', f'{lab} mean speed', round(w5[f'{col}{rr}'].value, 3), 'km/h', cls, F5, f'Sheet1!{col}{rr}', 'xlsx', recomputed=series.mean(), note=f'n={len(series)}')
        rep(G5, 'Exp5', f'{lab} SD', round(w5[f'{col}{rr+1}'].value, 3), 'km/h', cls, F5, f'Sheet1!{col}{rr+1}', 'xlsx', recomputed=series.std())
        rep(G5, 'Exp5', f'{lab} p85', w5[f'{col}{rr+2}'].value, 'km/h', cls, F5, f'Sheet1!{col}{rr+2}', 'xlsx', recomputed=pctl(series, 85, 'weibull'), note=f'group used PERCENTILE.EXC; INC would give {pctl(series,85):.2f}')
        rep(G5, 'Exp5', f'{lab} p50', w5[f'{col}{rr+3}'].value, 'km/h', cls, F5, f'Sheet1!{col}{rr+3}', 'xlsx', recomputed=pctl(series, 50, 'weibull'))
issue('MEDIUM', G5, 'Exp5', 'Experiment5_SpotSpeed_Report.docx §4', 'Report gives "Road Width: 3.75 m" for a 4-lane divided road (3.75 m is a lane width). Free-flow criterion stated as headway > 50 m (distance) in text; sample of free-flow vehicles is small: ' + ', '.join(f'{k} {int(x)}' for k,x in E5[(E5.group==G5)&(E5.flow_state=='Free flow')].vehicle_class.value_counts().items()) + '.', '', 'Kept')
small = E5[(E5.group == G5) & (E5.speed_kmh < 10)]
if len(small): issue('MEDIUM', G5, 'Exp5', F5, 'Radar readings below 10 km/h on a free-flowing arterial (likely mis-reads/typos): ' + ', '.join(f'{r.vehicle_class} {r.speed_kmh:g} ({r.source_location})' for r in small.itertuples()), '', 'Kept; flag')
small12 = E5[(E5.group == G12) & (E5.speed_kmh < 10)]
if len(small12): issue('MEDIUM', G12, 'Exp5', 'CE323_EXP5.pdf', 'Readings below 10 km/h: ' + ', '.join(f'{r.vehicle_class} {r.flow_state} {r.speed_kmh:g}' for r in small12.itertuples()), '', 'Kept; flag')
big12 = E5[(E5.group == G12) & (E5.speed_kmh > 90)]
if len(big12): issue('LOW', G12, 'Exp5', 'CE323_EXP5.pdf', 'Reading above 90 km/h: ' + ', '.join(f'{r.vehicle_class} {r.flow_state} {r.speed_kmh:g}' for r in big12.itertuples()), '', 'Kept; flag')
# Group 12 reported vs recomputed
for cls, vals in M.G12_E5_REPORTED.items():
    ff = E5[(E5.group == G12) & (E5.vehicle_class == cls) & (E5.flow_state == 'Free flow')].speed_kmh
    al = E5[(E5.group == G12) & (E5.vehicle_class == cls)].speed_kmh
    names = ['N free', 'N combined', 'Mean free', 'Mean combined', 'p85 free', 'p85 combined', 'p50 free', 'p50 combined', 'SD free', 'SD combined']
    rec = [len(ff), len(al), ff.mean(), al.mean(), pctl(ff, 85), pctl(al, 85), pctl(ff, 50), pctl(al, 50), ff.std(), al.std()]
    for nm, rv, rc in zip(names, vals, rec):
        rep(G12, 'Exp5', nm, rv, 'km/h' if 'N ' not in nm else 'veh', cls, 'CE323_EXP5.pdf', 'p.9 comparison table', 'pdf text', recomputed=rc)
issue('HIGH', G12, 'Exp5', 'CE323_EXP5.pdf p.7-9', 'Group 12 comparison table has the "free" and "combined" labels SWAPPED: their "free" N and mean equal the free+forced totals (Truck N 49, mean 35.61) and their "combined" mean equals the free-flow-only mean (Truck 37.00, Car 57.40). p50/p85 values also do not follow from the table (e.g. car "free" p50 55 with mean 44.81). Earlier per-class free-flow table (p.7) inherits the same error. Use Recomputed_Stats.', 'see Reported_Results check=MISMATCH rows', 'Raw table digitised; stats recomputed')
# Individual (Ayush) Exp5
for cls, n in M.AK_E5_COUNTS.items():
    rep(AK, 'Exp5', 'Number of observations', n, 'veh', cls, 'Spot Speed Study using Radar speed gun.pdf', 'p.4 Data Table', 'pdf text')
for nm, val in M.AK_E5_RESULTS:
    rep(AK, 'Exp5', nm, val, 'km/h', '', 'Spot Speed Study using Radar speed gun.pdf', 'p.4 Results', 'pdf text')
issue('HIGH', AK, 'Exp5', 'Spot Speed Study using Radar speed gun.pdf p.6-7', 'Raw handwritten datasheets (2 pages, ~345 readings, dated Sat 21/02/26) NOT digitised: column headers are re-labelled by hand ("car" over the Trucks column, "Bike" under LCV), circled totals (41/51/39/35) disagree with the report counts (38/45/37/34), and several digits are overwritten. Transcribing without the author would introduce errors.', '', 'Owner to type values into template sheet AK_E5_Template')
issue('MEDIUM', AK, 'Exp5', 'Radar report TOC', 'Date of experiment in report table = 08/02/2026, but datasheets are dated Sat 21/02/26 (same day as Group 5 Amingaon session).', '', 'Datasheet date preferred in Catalog')

# ============================================================== Exp 6 moving observer
F6 = 'Expt 6.xlsx'
w6 = openpyxl.load_workbook(SRC + F6, data_only=True).active
cls6 = ['Multi Axle', 'Truck', 'LCV', 'Bus', 'Car', 'Auto', 'Two Wheeler']
def read_block(r0, c0, n):  # rows r0.., type col c0, count c0+1, pcu c0+2
    out = {}
    for r in range(r0, r0 + n):
        t_ = w6.cell(r, c0).value
        if t_ is None: continue
        out[std(t_)] = (w6.cell(r, c0 + 1).value, w6.cell(r, c0 + 2).value, f'{openpyxl.utils.get_column_letter(c0)}{r}')
    return out
blocks = [  # (run_id, direction, count_type, r0, c0, n) -- mapping follows the group's own calculation table (Sheet1 R4:W9)
    ('NB-1', 'Northbound (Lane 1)', 'Overtaking test vehicle', 4, 9, 6), ('SB-1', 'Southbound (Lane 2)', 'Overtaking test vehicle', 4, 13, 6),
    ('NB-2', 'Northbound (Lane 1)', 'Overtaking test vehicle', 13, 9, 6), ('SB-2', 'Southbound (Lane 2)', 'Overtaking test vehicle', 13, 13, 6),
    ('NB-1', 'Northbound (Lane 1)', 'Overtaken by test vehicle', 24, 9, 6), ('SB-1', 'Southbound (Lane 2)', 'Overtaken by test vehicle', 24, 13, 6),
    ('NB-2', 'Northbound (Lane 1)', 'Overtaken by test vehicle', 33, 9, 6), ('SB-2', 'Southbound (Lane 2)', 'Overtaken by test vehicle', 33, 13, 6),
    ('run-1 (block A3:C10)', 'opposite stream: "from lane 1 to observing lane 2"', 'Opposite-direction count', 3, 1, 7),
    ('run-1 (block E3:G10)', 'opposite stream: "from lane 2 to observing lane 1"', 'Opposite-direction count', 3, 5, 7),
    ('run-2 (block A13:C20)', 'opposite stream: "from lane 1 to observing lane 2"', 'Opposite-direction count', 13, 1, 7),
    ('run-2 (block E13:G20)', 'opposite stream: "from lane 2 to observing lane 1"', 'Opposite-direction count', 13, 5, 7)]
rows = []
for run, d, ct, r0, c0, n in blocks:
    for k, (cnt, pcu, cell) in read_block(r0, c0, n).items():
        rows.append(dict(group=G5, experiment='Exp6', site='Amingaon, Guwahati; L = 3 km', run=run, direction=d, count_type=ct, vehicle_class=k,
                         count=cnt, pcu=pcu, source_file=F6, source_location=f'Sheet1!{cell}', extraction_method='xlsx'))
# Group 12 laps
for i, (fr, to, tot, ot, on, po, pn) in enumerate(M.G12_E6_LAPS):
    for k, a, b in zip(G12_E6c := M.G12_E6_CLASSES, ot, on):
        rows.append(dict(group=G12, experiment='Exp6', site='Road stretch L = 1.9 km (not named)', run=f'Lap {i+1} ({fr}-{to} min)', direction='not stated',
                         count_type='Overtaking test vehicle', vehicle_class=k, count=a, pcu=None, source_file='CE323_EXP6.pdf', source_location='p.4 table 1', extraction_method='pdf table'))
        rows.append(dict(group=G12, experiment='Exp6', site='Road stretch L = 1.9 km (not named)', run=f'Lap {i+1} ({fr}-{to} min)', direction='not stated',
                         count_type='Overtaken by test vehicle', vehicle_class=k, count=b, pcu=None, source_file='CE323_EXP6.pdf', source_location='p.4 table 1', extraction_method='pdf table'))
    for k, a in zip(M.G12_E6_CLASSES + ['Cycle'], M.G12_E6_OPPOSITE[i][:8]):
        rows.append(dict(group=G12, experiment='Exp6', site='Road stretch L = 1.9 km (not named)', run=f'Lap {i+1} ({fr}-{to} min)', direction='not stated',
                         count_type='Opposite-direction count', vehicle_class=k, count=a, pcu=None, source_file='CE323_EXP6.pdf', source_location='p.4 table 2', extraction_method='pdf table'))
E6C = pd.DataFrame(rows)
# run-level
runs = []
for r in (5, 6, 8, 9):
    runs.append(dict(group=G5, run=('NB-' if r < 7 else 'SB-') + ('1' if r in (5, 8) else '2'), direction='Northbound (Lane 1)' if r < 7 else 'Southbound (Lane 2)',
                     section_length_km=3.0, travel_time_s=w6.cell(r, 19).value, opposite_pcu_used=w6.cell(r, 21).value,
                     overtaking_pcu=w6.cell(r, 22).value, overtaken_pcu=w6.cell(r, 23).value,
                     q_reported=w6.cell(r, 24).value, t_stream_reported_s=w6.cell(r, 25).value, v_reported_kmh=w6.cell(r, 26).value,
                     k_reported=w6.cell(r, 27).value, source_file=F6, source_location=f'Sheet1!R{r}:AA{r}'))
R6 = pd.DataFrame(runs)
# recompute standard formula q = (Na + Nw)/(ta + tw) with Na = opposite count seen on the opposite-direction trip of the same run index
for i, r in R6.iterrows():
    idx = r.run[-1]; opp = R6[(R6.run.str[-1] == idx) & (R6.run != r.run)].iloc[0]
    q = (opp.opposite_pcu_used + r.overtaking_pcu - r.overtaken_pcu) / (opp.travel_time_s + r.travel_time_s) * 3600
    tbar = r.travel_time_s - (r.overtaking_pcu - r.overtaken_pcu) / q * 3600
    R6.loc[i, 'q_recomputed_pcu_hr'] = round(q, 3); R6.loc[i, 't_stream_recomputed_s'] = round(tbar, 3)
    R6.loc[i, 'v_recomputed_kmh'] = round(3.0 / tbar * 3600, 3); R6.loc[i, 'k_recomputed'] = round(q / (3.0 / tbar * 3600), 3)
for r in R6.itertuples():
    rep(G5, 'Exp6', 'Flow q (PCU/h)', round(r.q_reported, 3), 'PCU/h', r.run, F6, r.source_location, 'xlsx', recomputed=r.q_recomputed_pcu_hr,
        note='Recomputed with opposite count from the paired opposite-direction trip (the Nsb/Nnb labels in the sheet are swapped but values are used correctly)')
    rep(G5, 'Exp6', 'Stream speed v (km/h)', round(r.v_reported_kmh, 3), 'km/h', r.run, F6, r.source_location, 'xlsx', recomputed=r.v_recomputed_kmh)
    rep(G5, 'Exp6', 'Density k (PCU/km)', round(r.k_reported, 3), 'PCU/km', r.run, F6, r.source_location, 'xlsx', recomputed=r.k_recomputed)
for lane, r in [('Lane 1', 19), ('Lane 2', 20)]:
    rep(G5, 'Exp6', 'Average flow / speed / density', f'{w6.cell(r,21).value:.2f} / {w6.cell(r,22).value:.2f} / {w6.cell(r,23).value:.2f}', 'PCU/h / km/h / PCU/km', lane, F6, f'Sheet1!U{r}:W{r}', 'xlsx')
issue('LOW', G5, 'Exp6', 'Experiment 6_Moving observer method.docx', 'Report gives L as known section length; sheet states L = 3 km; road width 3.75 m (lane width) as in Exp5. Counts are in PCU, so "veh/hr" labels actually mean PCU/h.', '', 'Units recorded as PCU/h')
issue('MEDIUM', G5, 'Exp6', 'TOC across reports', 'Exp6 date listed as 21/02/2026 in all TOCs; submission date is 21/02/2026 in Exp6 report but 08/03/2026 in later reports.', '', 'Kept')
for i, (q, t_, vv) in enumerate(M.G12_E6_RESULT):
    fr, to, tot, ot, on, po, pn = M.G12_E6_LAPS[i]
    opp = M.G12_E6_OPPOSITE[i][8]
    rep(G12, 'Exp6', 'Flow (as reported, "veh/hr")', q, 'PCU/h', f'Lap {i+1}', 'CE323_EXP6.pdf', 'p.5 results', 'pdf table',
        recomputed=(opp - po + pn) / (2 * tot / 60), note='Reproduces only as (opposite − overtaking + overtaken)/(2·t): sign of overtaking terms reversed vs theory eq.(9)')
    rep(G12, 'Exp6', 'Average speed (as reported)', vv, 'km/h', f'Lap {i+1}', 'CE323_EXP6.pdf', 'p.5 results', 'pdf table', recomputed=1.9 / (tot / 60), note='= L / test-vehicle lap time (test vehicle journey speed, not stream space-mean speed)')
rep(G12, 'Exp6', 'Average flow / travel time / speed', f'{M.G12_E6_AVG[0]} / {M.G12_E6_AVG[1]} h / {M.G12_E6_AVG[2]}', '', 'All laps', 'CE323_EXP6.pdf', 'p.5', 'pdf table')
issue('HIGH', G12, 'Exp6', 'CE323_EXP6.pdf p.5', 'Flow uses (opposite − overtaking + overtaken)/(2t) — opposite sign to eq.(9) in their own theory — and the same lap time for both directions; "average speed" is the test vehicle speed L/T, not the stream speed. PCU factors are not stated (cannot be reproduced from IRC values). Lap directions are not recorded.', '', 'Raw counts digitised; reported results kept but flagged')

# ============================================================== Exp 7 TMC
rows = []
with open(SRC + 'Expt 7(Sheet1).csv', newline='') as f:
    rr = list(csv.reader(f))
PCU7 = {'Car': 1, 'Bus': 3.5, 'Two wheeler': 0.5, 'LCV': 2.2, 'Auto': 2, 'Truck': 3.5}
cur = {0: None, 7: None, 14: None}
for i, row in enumerate(rr[2:32], start=3):
    for c0, appr, movs in [(0, 'A (→)', ('Through', 'Left')), (7, 'B (←)', ('Through', 'Right')), (14, 'C (↓)', ('Left', 'Right'))]:
        if row[c0 + 1].strip(): cur[c0] = row[c0 + 1].strip()
        k = cur[c0]
        for j, mv in enumerate(movs):
            cnt = row[c0 + 2 + j]; pcu = row[c0 + 4 + j]
            rows.append(dict(group=G5, experiment='Exp7', site='T-intersection (not named); 3 approaches', approach=appr, interval_min=row[c0],
                             movement=mv, vehicle_class=std(k), count=int(float(cnt)), pcu_reported=float(pcu),
                             pcu_recomputed=int(float(cnt)) * PCU7[k], source_file='Expt 7(Sheet1).csv', source_location=f'row {i}, cols {c0+1}-{c0+6}', extraction_method='csv'))
for (route, mv), cnts in M.G12_E7_COUNTS.items():
    for k, c in zip(['Truck', 'LCV', 'Bus', 'Car', 'Auto', 'Two Wheeler'], cnts):
        rows.append(dict(group=G12, experiment='Exp7', site='T-intersection (not named)', approach=route, interval_min='whole count (duration not stated; ×3 used for hourly)',
                         movement=mv, vehicle_class=k, count=c, pcu_reported=None, pcu_recomputed=c * M.G12_E7_PCU[k],
                         source_file='CE323_EXP7 (1).pdf', source_location='p.2 observation tables', extraction_method='pdf text'))
E7 = pd.DataFrame(rows)
pm = E7[(E7.group == G5) & ((E7.pcu_reported - E7.pcu_recomputed).abs() > 1e-6)]
if len(pm): issue('LOW', G5, 'Exp7', 'Expt 7(Sheet1).csv', f'{len(pm)} cells where PCU ≠ count × factor', '', 'Kept')
issue('HIGH', G5, 'Exp7', 'Expt 7(Sheet1).csv / Experiment 7.pdf §7', 'Saturation headways 0.53 s / 1.18 s / 2.92 s give 6792 / 3051 / 1233 PCU/h/lane; 0.53 s is physically implausible. V/S for all three phases uses approach A saturation flow (6792.45). Phase-1 critical volume 4696.5 = A-through + B-through + A-left hourly PCU summed (not max per lane). Count covers 22 min in 4-min bins with a final 6-min bin (16-22) but hourly rates use ×15.', '', 'Raw counts digitised; design values flagged')
issue('MEDIUM', G5, 'Exp7', 'Experiment 7.pdf', 'Methodology says 2-h video with 15-min intervals; data are 22 min in 4-6 min bins. Date 21/3/2026 is after the Exp8 date (10/03) in the TOC.', '', 'Kept')
rep(G5, 'Exp7', 'Critical volume Phase 1/2/3', '4696.5 / 198 / 418.5', 'PCU/h', '', 'Expt 7(Sheet1).csv', 'rows 36-38', 'csv')
rep(G5, 'Exp7', 'Saturation headway A/B/C', '0.53 / 1.18 / 2.92', 's', '', 'Expt 7(Sheet1).csv', 'rows 26-28', 'csv')
rep(G5, 'Exp7', 'Sum V/S', 0.782191667, '-', '', 'Expt 7(Sheet1).csv', 'row 39', 'csv', recomputed=(4696.5 + 198 + 418.5) / 6792.45283)
rep(G5, 'Exp7', 'Optimum cycle C0', 65, 's', '', 'Experiment 7.pdf', '§6', 'pdf text', recomputed=(1.5 * 6 + 5) / (1 - 0.782191667), note='64.28 rounded to 65')
rep(G5, 'Exp7', 'Actual green Phase 1/2/3', '54.15 / 4.2 / 6.6', 's', '', 'Experiment 7.pdf', '§6', 'pdf text')
issue('LOW', G5, 'Exp7', 'Experiment 7.pdf §6', 'Green split fractions quoted as 0.8839/0.0373/0.0788 but V/S ratios are 0.6914/0.029/0.0616 (fractions = V/S ÷ 0.7822). Phase-2 green of 2.2 s effective is below any practical minimum.', '', 'Kept')
for (route, mv), val in M.G12_E7_REPORTED['lane_pcu'].items():
    rec = E7[(E7.group == G12) & (E7.approach == route) & (E7.movement == mv)].pcu_recomputed.sum()
    rep(G12, 'Exp7', 'Movement PCU', val, 'PCU', f'{route} {mv}', 'CE323_EXP7 (1).pdf', 'p.2', 'pdf text', recomputed=rec)
RR = M.G12_E7_REPORTED
for i in range(3):
    rep(G12, 'Exp7', f'V/S Phase {i+1}', RR['vs'][i], '-', '', 'CE323_EXP7 (1).pdf', 'p.3', 'pdf text', recomputed=RR['crit_hr'][i] / 1800)
rep(G12, 'Exp7', 'Sum V/S (computed)', RR['sum_vs'], '-', '', 'CE323_EXP7 (1).pdf', 'p.3', 'pdf text', recomputed=sum(RR['crit_hr']) / 1800)
rep(G12, 'Exp7', 'Cycle length C0 (with ΣV/S forced to 0.9)', RR['C0'], 's', '', 'CE323_EXP7 (1).pdf', 'p.3', 'pdf text', recomputed=(1.5 * 6 + 5) / (1 - 0.9))
rep(G12, 'Exp7', 'Green / amber / red Phase 1-3', '58/2/80, 70/2/68, 8/2/130', 's', '', 'CE323_EXP7 (1).pdf', 'p.4', 'pdf text')
issue('HIGH', G12, 'Exp7', 'CE323_EXP7 (1).pdf p.3', 'ΣV/S = 1.023 (>1, intersection over capacity with assumed S = 1800); the group replaced it with 0.9 to obtain C0 = 140 s. Saturation flow assumed, not measured. Count duration not stated (hourly = ×3 implies 20 min). Movement labels: Route 1 has "Right turn" and Route 2 "Left turn" only.', '', 'Kept, flagged')

# ============================================================== Exp 8 parking
F8 = 'Expt 8.xlsx'
w8 = openpyxl.load_workbook(SRC + F8, data_only=True).active
def tok(prefix, s, table={}):
    key = (prefix, str(s))
    if key not in table: table[key] = f'{prefix}{sum(1 for k in table if k[0]==prefix)+1:03d}'
    return table[key]
rows = []
for r in range(3, w8.max_row + 1):
    typ = w8.cell(r, 2).value
    if typ is None: continue
    plate = w8.cell(r, 3).value; tin = w8.cell(r, 4).value; tout = w8.cell(r, 5).value; dur = w8.cell(r, 6).value
    def mins(x): return x.hour * 60 + x.minute if isinstance(x, dt.time) else None
    rec = mins(tout) - mins(tin) if mins(tin) is not None and mins(tout) is not None else None
    rows.append(dict(group=G5, experiment='Exp8', site='Road-side parking near KV School, IIT Guwahati', vehicle_token=tok('G5P', plate),
                     plate_recorded_as='(pseudonymised)', vehicle_type_raw=typ, vehicle_type=std(typ), time_in=tin.strftime('%H:%M') if tin else None,
                     time_out=tout.strftime('%H:%M') if tout else None, duration_reported_min=dur, duration_recomputed_min=rec,
                     duration_check='OK' if rec == dur else 'MISMATCH', source_file=F8, source_location=f'Sheet1!B{r}:F{r}', extraction_method='xlsx'))
E8V = pd.DataFrame(rows)
mm = E8V[E8V.duration_check != 'OK']
if len(mm): issue('MEDIUM', G5, 'Exp8', F8, f'{len(mm)} rows where recorded duration ≠ out − in: ' + '; '.join(f'{r.source_location} {r.time_in}-{r.time_out} recorded {"(blank)" if pd.isna(r.duration_reported_min) else int(r.duration_reported_min)} vs {r.duration_recomputed_min}' for r in mm.itertuples()), '', 'Recomputed column added')
dup = E8V[E8V.duplicated('vehicle_token', keep=False)]
if len(dup):
    issue('LOW', G5, 'Exp8', F8, f'{dup.vehicle_token.nunique()} plate numbers (last 4 digits) appear more than once; some with overlapping stays, implying different vehicles sharing last-4 digits or entry errors: ' +
          '; '.join(f'{t}: ' + ', '.join(f'{a}-{b}' for a, b in zip(g.time_in, g.time_out)) for t, g in dup.groupby('vehicle_token')), '', 'Kept as separate records')
rep(G5, 'Exp8', 'Parking volume (90 min)', 114, 'veh', '', F8, 'U4', 'xlsx', recomputed=len(E8V))
rep(G5, 'Exp8', 'Average parking duration', 29.2719, 'min', '', F8, 'U6', 'xlsx', recomputed=E8V.duration_reported_min.mean(), note='recomputed = mean of recorded durations')
rep(G5, 'Exp8', 'Parking load', 60.6667, 'veh-h', '', F8, 'U3 / L12', 'xlsx', recomputed=sum(w8.cell(r, 11).value for r in range(3, 12)) * 10 / 60)
rep(G5, 'Exp8', 'Average arrival rate', 54, 'veh/h', '', F8, 'U7', 'xlsx')
acc = [(w8.cell(r, 9).value, w8.cell(r, 11).value) for r in range(3, 12)]
def occ_at(tm):
    t0 = int(tm[:2]) * 60 + int(tm[3:5])
    return int(((E8V.time_in.map(lambda x: int(x[:2]) * 60 + int(x[3:]))) <= t0).mul(E8V.time_out.map(lambda x: int(x[:2]) * 60 + int(x[3:])) > t0).sum())
E8ACC = pd.DataFrame([dict(group=G5, interval=a, accumulation_reported=b, occupancy_at_interval_start_recomputed=occ_at(a[:5]),
                           source_file=F8, source_location=f'Sheet1!I{3+i}:K{3+i}') for i, (a, b) in enumerate(acc)])
issue('MEDIUM', G5, 'Exp8', 'Expt 8_Parking Study.docx / Expt 8.xlsx', 'Date of survey written as 21/02/2026 in the report body but 10/03/2026 in the TOC; time "10:20 am to 11:50 pm" (data 10:20-11:45). Last accumulation interval is 11:40-11:45 (5 min) but load uses 10 min for it. Only 115 vehicle rows vs reported volume 114. Arrival rate 54 veh/h not derivable from stated formula.', '', 'Kept')
# Group 12 bays
rows = []
tbl = {}
for b, (typ, snaps, dur) in enumerate(M.G12_E8_ROWS, start=1):
    occ = sum(1 for s in snaps if s != '0')
    for tm, s in zip(M.G12_E8_TIMES, snaps):
        rows.append(dict(group=G12, experiment='Exp8', site='Parking area (not named in report)', bay_row=b, vehicle_type=std(typ), snapshot_time=tm,
                         occupied=s != '0', vehicle_token=None if s == '0' else tok('G12P', s, tbl), row_duration_reported_min=dur,
                         row_duration_recomputed_min=occ * 10, source_file='CE323_EXP8.pdf', source_location='p.3-4 observation table', extraction_method='pdf text'))
E8B = pd.DataFrame(rows)
tot = E8B.groupby('snapshot_time').occupied.sum().reindex(M.G12_E8_TIMES).tolist()
issue('HIGH', G12, 'Exp8', 'CE323_EXP8.pdf', f'Row totals printed {M.G12_E8_ROW_TOTALS}; recomputed from the table {tot}; accumulation table used in calculations {M.G12_E8_REPORTED["accum_table"]}. Max accumulation stated as 46 and 44; parking index uses capacity 49 in text and 52 in the formula. Rows are bays (plate changes within a row), so "parking volume = 52" counts bays, not vehicles; durations are bay-occupancy durations. Arrivals (14) and departures (14) listed do not match accumulation changes.', '', 'Bay table digitised with pseudonymous tokens; results flagged')
n_veh12 = E8B.dropna(subset=['vehicle_token']).vehicle_token.nunique()
rep(G12, 'Exp8', 'Parking volume', 52, 'veh', '', 'CE323_EXP8.pdf', 'p.5', 'pdf text', recomputed=n_veh12, note='recomputed = distinct plate tokens in table')
rep(G12, 'Exp8', 'Parking load', 3690, 'veh-min', '', 'CE323_EXP8.pdf', 'p.5', 'pdf text', recomputed=sum(tot) * 10)
rep(G12, 'Exp8', 'Average duration (row-based)', 70.38, 'min', '', 'CE323_EXP8.pdf', 'p.6', 'pdf text', recomputed=np.mean([d for _, _, d in M.G12_E8_ROWS]))
rep(G12, 'Exp8', 'Average duration cars', 91.42, 'min', '', 'CE323_EXP8.pdf', 'p.6', 'pdf text', recomputed=np.mean([d for t_, _, d in M.G12_E8_ROWS if t_ == 'car']))
rep(G12, 'Exp8', 'Average duration bikes', 64.47, 'min', '', 'CE323_EXP8.pdf', 'p.6', 'pdf text', recomputed=np.mean([d for t_, _, d in M.G12_E8_ROWS if t_ == 'bike']))
rep(G12, 'Exp8', 'Parking index', 78.85, '%', '', 'CE323_EXP8.pdf', 'p.5', 'pdf text', recomputed=3690 / (52 * 90) * 100)
issue('MEDIUM', G12, 'Exp8', 'CE323_EXP8.pdf p.5-6', f'Arithmetic check: parking load from the printed accumulation table is 3690 veh-min, but the bay table itself gives {sum(tot)*10}; car durations in the table sum to {sum(d for t_,_,d in M.G12_E8_ROWS if t_=="car")} min (report says 1280 → 91.42 min); distinct plates in the table = {n_veh12} (report volume 52 = bay rows).', '', 'Recomputed values in Reported_Results')
rep(G12, 'Exp8', 'Arrival / departure rate', '10 / 10', 'veh/h', '', 'CE323_EXP8.pdf', 'p.7', 'pdf text')

# ============================================================== Exp 9 O-D
F9 = 'Expt 9.xlsx'
w9 = openpyxl.load_workbook(SRC + F9, data_only=True).active
MODE = {1: 'Walk', 2: 'Cycle', 3: 'Two Wheeler', 4: 'Car'}
PURP = {1: 'Class', 2: 'Office/work', 3: 'Hostel return', 4: 'Medical', 5: 'Shopping', 6: 'Visitor', 7: 'Other'}
rows = []
for r in range(2, 32):
    if w9.cell(r, 1).value is None: continue
    rows.append(dict(group=G5, experiment='Exp9', site='Market Complex, IIT Guwahati', date='2026-03-17', stream='Outgoing (col A-H)', serial=w9.cell(r, 1).value,
                     interview_time=w9.cell(r, 2).value.strftime('%H:%M'), mode_text=w9.cell(r, 3).value, mode_code=w9.cell(r, 4).value,
                     mode=MODE.get(w9.cell(r, 4).value), origin_zone=w9.cell(r, 5).value, destination_zone=w9.cell(r, 6).value,
                     purpose_code=w9.cell(r, 7).value, purpose=PURP.get(w9.cell(r, 7).value), occupancy=w9.cell(r, 8).value,
                     source_file=F9, source_location=f'Sheet1!A{r}:H{r}', extraction_method='xlsx'))
for r in range(4, 33):
    if w9.cell(r, 11).value is None: continue
    rows.append(dict(group=G5, experiment='Exp9', site='Market Complex, IIT Guwahati', date='2026-03-17', stream='Incoming (col K-Q)', serial=w9.cell(r, 11).value,
                     interview_time=w9.cell(r, 12).value.strftime('%H:%M'), mode_text=None, mode_code=w9.cell(r, 13).value,
                     mode=MODE.get(w9.cell(r, 13).value), origin_zone=w9.cell(r, 14).value, destination_zone=w9.cell(r, 15).value,
                     purpose_code=w9.cell(r, 16).value, purpose=PURP.get(w9.cell(r, 16).value), occupancy=w9.cell(r, 17).value,
                     source_file=F9, source_location=f'Sheet1!K{r}:Q{r}', extraction_method='xlsx'))
E9I = pd.DataFrame(rows)
odd = E9I[E9I.interview_time.str[:2].astype(int) >= 13]
issue('LOW', G5, 'Exp9', F9, f'{len(odd)} interview times outside the 10:30-12:00 survey window (likely 11:xx mistyped): ' + ', '.join(f'{r.source_location} {r.interview_time}' for r in odd.itertuples()), '', 'Kept verbatim')
mt = E9I[(E9I.mode_text.notna()) & (E9I.mode_text.map(std) != E9I['mode'])]
if len(mt): issue('LOW', G5, 'Exp9', F9, 'Mode text vs mode code disagree: ' + ', '.join(f'{r.source_location} "{r.mode_text}"→{r.mode}' for r in mt.itertuples()), '', 'Kept')
no_occ = E9I[E9I.occupancy.isna()]
if len(no_occ): issue('LOW', G5, 'Exp9', F9, f'{len(no_occ)} interview(s) without occupancy: ' + ', '.join(no_occ.source_location), '', 'Left blank')
issue('INFO', G5, 'Exp9', F9, 'Incoming list: every destination = zone 5 (survey point); outgoing list: every origin = zone 5. Zone 5 = New D, E Type, Market Complex.', '', '')
vol9 = [('Car', 47, 30, 1.5667), ('Two Wheeler', 31, 18, 1.7222), ('Cycle', 16, 9, 1.7778), ('Walk', 12, 2, 6)]
E9V = [dict(group=G5, experiment='Exp9', mode=m, interval='whole survey 10:30-12:00', counted=c, interviewed_reported=i_, interviewed_in_raw=int((E9I['mode'] == m).sum()),
            expansion_factor_reported=ef, source_file=F9, source_location='Sheet1!P44:S47') for m, c, i_, ef in vol9]
for m, lst in M.G12_E9_VOLUME.items():
    for k, x in enumerate(lst):
        E9V.append(dict(group=G12, experiment='Exp9', mode=m, interval=f'{15*k}-{15*k+15} min', counted=x, interviewed_reported=None, interviewed_in_raw=None,
                        expansion_factor_reported=None, source_file='Experiment9_grp 12.pdf', source_location='p.9 handwritten Traffic Volume Count sheet'))
E9V = pd.DataFrame(E9V)
for m, (cnt, n, ef) in M.G12_E9_EF.items():
    s = sum(M.G12_E9_VOLUME[m])
    rep(G12, 'Exp9', 'Total counted (EF numerator)', cnt, 'trips', m, 'Experiment9_grp 12.pdf', 'p.2 §5.1', 'pdf text', recomputed=s, note='recomputed = sum of 15-min handwritten counts')
    rep(G12, 'Exp9', 'Interviewed', n, 'trips', m, 'Experiment9_grp 12.pdf', 'p.2 §5.1', 'pdf text', recomputed=sum(M.G12_E9_PARTIAL[m].values()))
    rep(G12, 'Exp9', 'Expansion factor', ef, '-', m, 'Experiment9_grp 12.pdf', 'p.2 §5.1', 'pdf text', recomputed=cnt / n)
for m, c, i_, ef in vol9:
    rep(G5, 'Exp9', 'Expansion factor', ef, '-', m, F9, 'Sheet1!S44:S47', 'xlsx', recomputed=c / i_)
    rep(G5, 'Exp9', 'Interviewed', i_, 'trips', m, F9, 'Sheet1!R44:R47', 'xlsx', recomputed=int((E9I['mode'] == m).sum()))
pur = E9I.purpose.value_counts()
for code, name in PURP.items():
    rep(G5, 'Exp9', 'Trips by purpose (sample)', {1: 0, 2: 12, 3: 1, 4: 1, 5: 36, 6: 2, 7: 7}[code], 'trips', name, F9, 'Sheet1!D40:D46', 'xlsx', recomputed=int(pur.get(name, 0)))
rows = []
for m, d in M.G12_E9_PARTIAL.items():
    for (o, de), n in d.items():
        rows.append(dict(group=G12, experiment='Exp9', mode=m, origin_zone=o, destination_zone=de, sampled_trips=n,
                         expansion_factor=M.G12_E9_EF[m][2], expanded_trips=round(n * M.G12_E9_EF[m][2], 2),
                         source_file='Experiment9_grp 12.pdf', source_location='Partial O-D matrix images p.3-4', extraction_method='image, read at 170 dpi; totals reconcile'))
    assert sum(d.values()) == M.G12_E9_PARTIAL_TOTALS[m]
E9P = pd.DataFrame(rows)
issue('MEDIUM', G12, 'Exp9', 'Experiment9_grp 12.pdf', 'Survey date: photos geotagged 23/03/2026 10:33-11:46 IST at 26.18748 N, 91.68987 E; handwritten count sheet date is overwritten (reads 17/03/2026?). Raw interview sheets only appear as a small photo (not legible); only the partial O-D matrices (by mode) and 15-min volume counts are recoverable. Occupancy per interview not available.', '', 'Partial matrices + volumes digitised')
issue('LOW', G5, 'Exp9', F9, 'Expansion factor for Walk = 12/2 = 6 based on only 2 interviews; the 33% sampling target was not met for walk (17%) and barely for cars (64% sampled).', '', 'Kept')
ZONES = pd.DataFrame([(1, 'Core 1 & 2'), (2, 'Core 3 & 4'), (3, 'Core 5, Nano & RnD Bldg'), (4, 'Lecture Halls & Admin Block, CC, Library, Conf Centre, Audi'),
                      (5, 'New D, E Type, Market Complex'), (6, 'IITG Main Gate and outside'), (7, 'D, E, F Type'), (8, 'A, B, C Type, Guest House 1 & 2, Hospital'),
                      (9, 'F Type bungalow, Both Girls Hostels, MSH'), (10, 'Boys Hostels Umiam, Barak, Kameng, Gaurang, etc, Transit'),
                      (11, 'Boys Hostels Siang, Kapili, Dihing, KV, etc'), (12, 'KV Gate and outside')], columns=['zone', 'description_as_in_sheet'])
ZONES['source'] = 'Expt 9.xlsx Sheet1!J39:J50 (hostel names as typed there: Kemeng, Shiyang, Disang)'

# ============================================================== Exp 2 V-Box (reported only) + Exp 3 G12 / AK
for nm, val, u in M.G5_E2:
    rep(G5, 'Exp2', nm, val, u, 'Instructor-provided trip', 'Transportation Lab 2 (1).docx', 'Tables §5, §7', 'docx table')
issue('LOW', G5, 'Exp2', 'Transportation Lab 2 (1).docx', 'Speed value 46.96 km/h is labelled 5th percentile in §5.1 and 15th percentile in §7 Results. Min speed 0.632 km/h listed although zero-speed rows removed only.', '', 'Recorded with both labels')
for q, val in M.G12_E2_SPEED_PCTL.items():
    rep(G12, 'Exp2', f'Speed p{q}', val, 'km/h', 'Instructor-provided trip', 'VBOX_Group12.pdf', 'p.2 A1', 'pdf text')
for q, (a, d, l) in M.G12_E2_ACC_PCTL.items():
    rep(G12, 'Exp2', f'Long. accel p{q}', a, 'm/s2', '', 'VBOX_Group12.pdf', 'p.4 B1', 'pdf text')
    rep(G12, 'Exp2', f'Long. decel p{q}', d, 'm/s2', '', 'VBOX_Group12.pdf', 'p.4 B1', 'pdf text')
    rep(G12, 'Exp2', f'Lateral accel p{q}', l, 'm/s2', '', 'VBOX_Group12.pdf', 'p.4 B1', 'pdf text')
for k, val in M.G12_E2_KS.items():
    rep(G12, 'Exp2', f'KS-type fit statistic, speed ({k})', val, '-', '', 'VBOX_Group12.pdf', 'p.3 A3', 'pdf text', note='Weibull lowest → best fit')
rep(G12, 'Exp2', 'Speed at max accel / decel', '≈25-30 / ≈25-30', 'km/h', '', 'VBOX_Group12.pdf', 'p.5 B4', 'pdf text')
issue('INFO', 'Group 5 & Group 12', 'Exp2', 'both V-Box reports', 'V-Box trip data were provided by the instructor (not collected by students; route/location not given). Groups report different percentiles for what is likely the same trip (p85 63.44 vs 59.22 km/h) because of different smoothing/zero-speed filtering. Raw V-Box file not in the folder (Group 5 links a OneDrive workbook).', '', 'Only reported statistics included')
issue('LOW', G12, 'Exp2', 'VBOX_Group12.pdf p.4', 'Long. acceleration, deceleration and lateral acceleration percentiles are identical at p15/p25 (0.11, 0.18) — suggests the same column was used for all three.', '', 'Kept')
for lane, rws in M.G12_E3_LANE_2MIN.items():
    for iv, n, pcu in rws:
        rep(G12, 'Exp3', f'Vehicles / PCU in {iv} min', f'{n} / {pcu}', 'veh / PCU', f'{lane} lane', 'exp 3.pdf', 'p.2 Part A tables', 'pdf text')
for lane in ('Inner', 'Outer'):
    a = M.G12_E3_LANE_2MIN[lane]
    rep(G12, 'Exp3', 'Total vehicles 10 min', sum(x[1] for x in a), 'veh', f'{lane} lane', 'exp 3.pdf', 'p.2 (sum of table)', 'derived from pdf text')
comb = M.G12_E3_LANE_2MIN['Combined']
for i, (iv, n, pcu) in enumerate(comb):
    a, b = M.G12_E3_LANE_2MIN['Inner'][i], M.G12_E3_LANE_2MIN['Outer'][i]
    if n != a[1] + b[1] or abs(pcu - (a[2] + b[2])) > 0.05:
        issue('MEDIUM', G12, 'Exp3', 'exp 3.pdf Part A', f'Combined lane row {iv}: {n} veh / {pcu} PCU but Inner+Outer = {a[1]+b[1]} veh / {a[2]+b[2]:.1f} PCU', '', 'Kept')
for cls, (av, p50, p85) in M.G12_E3_FFS.items():
    rep(G12, 'Exp3', 'Free-flow speed avg / p50 / p85', f'{av} / {p50} / {p85}', 'km/h', cls, 'exp 3.pdf', 'p.4 table', 'pdf text')
for cls, (av, p85, p50) in M.G12_E3_STREAM.items():
    rep(G12, 'Exp3', 'Stream speed avg / p85 / median', f'{av} / {p85} / {p50}', 'km/h', cls, 'exp 3.pdf', 'p.5 table', 'pdf text')
rep(G12, 'Exp3', 'Normality test (all FFS)', 'mean 40.71, SD 18.71, dof 10, crit 18.31, chi2 not given → reject', '', '', 'exp 3.pdf', 'p.4', 'pdf text')
rep(G12, 'Exp3', 'Peak observed flow', '~2600', 'PCU/h', '', 'exp 3.pdf', 'p.5 comments', 'pdf text')
issue('HIGH', G12, 'Exp3', 'exp 3.pdf p.4-5', 'Stream (combined) speed statistics are identical to the free-flow statistics for all four classes (e.g. Car 41.88/48.62/41.53) — the combined analysis was not actually separate. For Two Wheeler, p85 47.81 is barely above mean 45.04 while p50 is 40.51 (inconsistent shape). No PHF number and no chi-square value reported. Procedure says 1.5 h recording, 100 m trap, 5-min counts, but tables are 10 min in 2-min bins. No raw per-vehicle data.', '', 'Reported values kept, flagged')
tot_ak = {'Lane 2': 163, 'Lane 1': 122}
for lane, d in M.AK_E3_COMPOSITION_PCT.items():
    for cls, pc in d.items():
        n = round(pc / 100 * tot_ak[lane])
        rep(AK, 'Exp3', 'Traffic composition share', pc, '%', f'{lane} {cls}', 'Mid-block Traffic Volume and Speed Study by Video.pdf', 'p.6 pie-chart labels', 'image (read at 200 dpi)',
            note=f'= {n}/{tot_ak[lane]} exactly → implied count {n}')
issue('MEDIUM', AK, 'Exp3', 'Mid-block Traffic Volume and Speed Study by Video.pdf', 'Only charts are included (no tables, no raw data). Pie shares back-solve exactly to integer counts (Lane 2: 163 vehicles; Lane 1: 122). Speed histograms show bins up to 220-240 km/h for cars and ~160 km/h for LCV — same entry-time errors as seen in Group 5 workbook. Composition is close to Group 5 (same mid-block video likely shared by the class).', '', 'Composition recorded; speeds not digitised')

# ============================================================== Catalog
CAT = pd.DataFrame([
 # group, exp, title, date, site, method/equipment, duration/sample, raw data available, source files, key results (as reported), status
 (G5, 1, 'Driver Vision Test', '2026-01-13', 'Lab', 'Vision tester', '-', 'No (no report in folder)', '(TOC only)', '-', 'Not available'),
 (G5, 2, 'Vehicle speed/accel/decel profile (V-Box)', '2026-01-20', 'Instructor-provided GPS trip', 'V-Box GPS data (given)', '4961 rows (3994 moving)', 'No (OneDrive link only)', 'Transportation Lab 2 (1).docx', 'Mean 55.51, p85 63.44 km/h; max decel 2.07 m/s²; Weibull best fit', 'Reported stats only'),
 (G5, 3, 'Mid-block volume & speed by video', '2026-01-27', 'Mid-block, 2 lanes one direction (site not named); 50 m trap', 'Video, frame timing', '10 min; 321 vehicles', 'YES per-vehicle in/out times', F3 + '; Transport Lab 3.docx', 'Shoulder 178 veh/206.2 PCU, median 150 veh/250.3 PCU; PHF 0.81/0.93', 'Raw digitised + recomputed'),
 (G5, 4, 'Time headway distribution', '2026-02-03', 'Same video as Exp3', 'Video frame-by-frame', '321 vehicles (189 shoulder, 131 median... see check)', 'YES (same list as Exp3)', F4 + '; transportation exp 4.docx', 'Mean headway 3.04 s shoulder, 4.42 s median; neg-exp fits', 'Raw digitised + recomputed'),
 (G5, 5, 'Spot speed (radar gun)', '2026-02-21', 'Amingaon, Guwahati, 4-lane divided, 10:00', 'Radar gun', 'N5', 'YES', F5 + '; Experiment5_SpotSpeed_Report.docx', 'Free-flow mean: Car 58.5, 2W 51.0, Truck 40.6 km/h', 'Raw digitised + recomputed'),
 (G5, 6, 'Moving observer (volume & travel time)', '2026-02-21', 'Amingaon, Guwahati; L = 3 km', 'Test car, 2 runs each direction', '4 runs', 'YES (class counts per run)', F6 + '; Experiment 6_Moving observer method.docx', 'Lane 1: q 2000 PCU/h, v 42.0 km/h; Lane 2: q 1966, v 34.8', 'Raw digitised + recomputed'),
 (G5, 7, 'Turning movement count & Webster signal design', '2026-03-21', 'T-intersection (not named)', 'Video', '22 min, 3 approaches', 'YES', 'Expt 7(Sheet1).csv; Experiment 7.pdf', 'C0 = 65 s; phase greens 54.15/4.2/6.6 s', 'Raw digitised; design flagged'),
 (G5, 8, 'Parking study (licence plate in/out)', '2026-03-10 (TOC) / 2026-02-21 (body)', 'Road-side parking near KV School, IITG', 'Plate survey, 10-min checks', '10:20-11:45; 115 vehicles', 'YES (plates pseudonymised)', F8 + '; Expt 8_Parking Study.docx', 'Load 60.67 veh-h, volume 114, mean duration 29.3 min', 'Raw digitised + recomputed'),
 (G5, 9, 'Roadside interview O-D', '2026-03-17', 'Market Complex, IIT Guwahati; 10:30-12:00', 'Interview + volume count', '59 interviews; 106 counted', 'YES', F9 + '; Bhavesh Bajaj_Expt9.pdf', 'Shopping 61% of sampled trips; EF car 1.57, 2W 1.72', 'Raw digitised'),
 (G12, 2, 'V-Box profile', '-', 'Instructor-provided trip', 'V-Box (given)', '-', 'No', 'VBOX_Group12.pdf', 'Speed p85 59.22 km/h; lognormal for accel', 'Reported stats only'),
 (G12, 3, 'Mid-block volume & speed by video', '-', 'Mid-block (not named)', 'Video, 2 cameras 100 m apart', '10 min tables', 'Partial (2-min lane totals)', 'exp 3.pdf', 'FFS car 41.9, 2W 45.0 km/h', 'Reported values'),
 (G12, 4, 'Time headway distribution', '-', 'Video', 'Frame-by-frame', '252 vehicles (113 inner, 139 outer)', 'Binned frequencies only', F12_4, 'Mean 5.33 s / 4.43 s; combined 2.44 s', 'Bins digitised'),
 (G12, 5, 'Spot speed (radar gun)', '-', 'Mid-block (not named)', 'Radar gun', '296 readings', 'YES', 'CE323_EXP5.pdf', 'See recomputed stats', 'Raw digitised + recomputed'),
 (G12, 6, 'Moving observer', '-', 'Stretch L = 1.9 km (not named)', 'Test vehicle, 4 laps', '4 laps, 26.6 min', 'YES (class counts per lap)', 'CE323_EXP6.pdf', 'Avg flow 1903 PCU/h; speed 19.8 km/h (test vehicle)', 'Raw digitised; method flagged'),
 (G12, 7, 'TMC & Webster signal design', '-', 'T-intersection (not named)', 'Manual tally', 'duration not stated (×3 to hour)', 'YES (movement × class totals)', 'CE323_EXP7 (1).pdf', 'C0 = 140 s (ΣV/S forced to 0.9)', 'Raw digitised; design flagged'),
 (G12, 8, 'Parking study (licence plate)', '-', 'Parking area (not named); 10:40-12:00', 'Bay snapshots every 10 min', '52 bay rows × 9 snapshots', 'YES (tokens)', 'CE323_EXP8.pdf', 'Load 3690 veh-min; index 78.85%', 'Raw digitised; results flagged'),
 (G12, 9, 'Roadside interview O-D (campus)', '2026-03-23 (photo GPS)', 'IITG campus 26.18748 N, 91.68987 E', 'Interview + 15-min counts', '53 interviews; 435 counted', 'Partial matrices + counts', 'Experiment9_grp 12.pdf', 'EF walk 3.67, cycle 7.61, 2W 10.25, car 10.0; Zone 4 top destination', 'Digitised'),
 (AK, 3, 'Mid-block volume & speed by video', '2026-01-27', 'Mid-block (not named)', 'Video', 'Lane 1: 122 veh, Lane 2: 163 veh (implied)', 'Charts only', 'Mid-block Traffic Volume and Speed Study by Video.pdf', 'Lane 2 2W 45.4%, Lane 1 Car 56.6%', 'Composition only'),
 (AK, 5, 'Spot speed (radar gun)', '2026-02-21 (datasheet)', 'Urban two-lane road (not named)', 'Radar gun', '345 readings (report)', 'Handwritten images — NOT digitised', 'Spot Speed Study using Radar speed gun.pdf', 'Car mean ≈54.6, 2W ≈52, Auto ≈33, car p85 ≈67 km/h', 'Needs owner transcription'),
], columns=['group', 'experiment_no', 'title', 'date', 'site', 'method_equipment', 'duration_sample', 'raw_data_available', 'source_files', 'key_results_as_reported', 'dataset_status'])
CAT.loc[(CAT.group == G5) & (CAT.experiment_no == 5), 'duration_sample'] = f"{(E5.group==G5).sum()} readings"
CAT.loc[(CAT.group == G12) & (CAT.experiment_no == 5), 'duration_sample'] = f"{(E5.group==G12).sum()} readings"
CAT.loc[(CAT.group == G5) & (CAT.experiment_no == 4), 'duration_sample'] = f"321 vehicles ({(E3V.lane=='Shoulder').sum()} shoulder, {(E3V.lane=='Median').sum()} median)"
CAT.loc[(CAT.group == G12) & (CAT.experiment_no == 9), 'duration_sample'] = f"{E9P.sampled_trips.sum()} interviews; {sum(sum(x) for x in M.G12_E9_VOLUME.values())} counted"
issue('INFO', G12, 'All', 'Group 12 PDF cover pages', 'Cover pages of CE323_EXP5-8 say "6th Semester (Jan – May 2025)"; other Group 12 reports say 2026. Treated as a typo (experiments reference Jan-Mar 2026 lab schedule).', '', '')
issue('INFO', 'All', 'All', 'Lab Report folder', 'Experiment 1 (Driver Vision Test) has no report in the folder; Group 12 Exp2-8 give no survey dates or site names.', '', '')

# ============================================================== Sources
srcs = []
for f in sorted(os.listdir(SRC)):
    b = open(SRC + f, 'rb').read()
    srcs.append(dict(file=f, bytes=len(b), sha256_12=hashlib.sha256(b).hexdigest()[:12]))
SRCS = pd.DataFrame(srcs)
owner = {'Group12': G12, 'VBOX': G12, 'CE323': G12, 'exp 3': G12, 'Experiment9_grp': G12, 'Spot Speed Study': AK, 'Mid-block': AK}
def own(f):
    for k, g in owner.items():
        if f.startswith(k): return g
    return G5
SRCS['group'] = SRCS.file.map(own)
def expn(f):
    m = re.search(r'(?:exp|Expt|EXP|Lab|Experiment|experiment)\s*_?(\d)', f, re.I)
    if 'VBOX' in f: return 2
    if 'Mid-block' in f or 'Traffic volume video' in f: return 3
    if 'Spot Speed' in f: return 5
    if 'Moving' in f: return 6
    return int(m.group(1)) if m else None
SRCS['experiment_no'] = SRCS.file.map(expn)
SRCS['type'] = SRCS.file.str.extract(r'\.(\w+)$')[0]
SRCS.loc[SRCS.file.str.startswith('Bhavesh_Traffic'), 'experiment_no'] = 3

# ============================================================== Dictionary + README
DICT = pd.DataFrame([
 ('*', 'group', 'Group 5 / Group 12 / Individual (Ayush Kumar). Student names and roll numbers intentionally omitted.'),
 ('*', 'source_file / source_location', 'Exact file and sheet!cell / page / table the value was read from.'),
 ('*', 'extraction_method', 'xlsx / csv / docx table / pdf text / pdf table / image (read manually). Image-read values are flagged as such.'),
 ('E3E4_G5_Vehicles', 't_in_s, t_out_s', 'Seconds from video start when the front bumper crosses the entry / exit line (final values from the lane analysis sheets).'),
 ('E3E4_G5_Vehicles', 't_in_rawsheet, t_out_rawsheet', 'Same times as typed in the raw-entry sheet "Speed (free & non free)". Differences are flagged.'),
 ('E3E4_G5_Vehicles', 'speed_kmh_recomputed', '50 m / (t_out_s − t_in_s) × 3.6 (trap length 50 m confirmed from sheet speeds).'),
 ('E3E4_G5_Vehicles', 'headway_to_leader_s', 'In-time gap to the previous vehicle in the same lane (recomputed). headway_reported_s is the sheet value (gap to next vehicle).'),
 ('E3E4_G5_Vehicles', 'free_flow_reported / free_flow_recomputed_gt5s', 'Sheet flag vs recomputed flag (gap to next vehicle > 5 s).'),
 ('E3_Volume', 'count, pcu', 'Classified 2-min counts; PCU factors from the group sheet (MA 4.5, Truck 3.5, LCV 2.2, Bus 3.5, Car 1, Auto 2, 2W 0.5).'),
 ('E5_SpotSpeed', 'flow_state', 'Group 5: "Free flow" / "Non-free (car-following)". Group 12: "Free flow" / "Forced (car-following)". Group 5 "Comb." columns are duplicates and not repeated.'),
 ('E6_MovingObserver_Counts', 'count_type', 'Overtaking test vehicle / Overtaken by test vehicle / Opposite-direction count.'),
 ('E6_G5_Runs', 'q/t/v/k _reported vs _recomputed', 'Group values vs textbook moving-observer formulae recomputed from the same counts (PCU).'),
 ('E7_TMC', 'pcu_reported / pcu_recomputed', 'Group 5 PCU factors: Car 1, Bus 3.5, 2W 0.5, LCV 2.2, Auto 2, Truck 3.5. Group 12 (IRC): Truck 3, LCV 1.5, Bus 3, Car 1, Auto 1, 2W 0.5.'),
 ('E8_G5_Parking_Vehicles', 'vehicle_token', 'Pseudonymous ID replacing the recorded plate digits (same plate → same token within a group).'),
 ('E8_G12_Parking_Bays', 'bay_row', 'Row of the group bay table; one row can hold successive vehicles.'),
 ('E9_G5_Interviews', 'origin_zone / destination_zone / purpose_code', 'Campus zone codes (see E9_Zones) and purpose codes 1 Class, 2 Office/work, 3 Hostel return, 4 Medical, 5 Shopping, 6 Visitor, 7 Other.'),
 ('Reported_Results', 'check', 'MATCH / MISMATCH when a reported number could be recomputed from raw data in this dataset (tolerance max(0.02, 0.5%)).'),
 ('Validation_Log', 'severity', 'HIGH = do not use reported value / method error; MEDIUM = inconsistency to disclose; LOW = minor; INFO = context.'),
], columns=['sheet', 'column', 'meaning'])
README = pd.DataFrame({'CE323 Transport Engineering Lab-II — consolidated dataset': [
 'Built ' + dt.date.today().isoformat() + ' from 25 files in "Lab Report" (Group 5, Group 12, and Ayush Kumar individual reports), IIT Guwahati, Jan-Mar 2026.',
 'Instructor: Prof. A.K. Maurya. Personal details (names, roll numbers, licence plates) removed or pseudonymised.',
 'Rule 1 — raw values verbatim. Nothing in raw sheets is corrected; recomputed values sit in separate columns.',
 'Rule 2 — every row has source_file, source_location and extraction_method.',
 'Rule 3 — every inconsistency is written to Validation_Log with severity; nothing silently fixed.',
 'Raw observation sheets: E3E4_G5_Vehicles, E3_Volume, E4_G12_HeadwayBins, E5_SpotSpeed, E6_MovingObserver_Counts, E6_G5_Runs, E7_TMC, E8_G5_Parking_Vehicles, E8_G5_Accumulation, E8_G12_Parking_Bays, E9_G5_Interviews, E9_G12_PartialOD, E9_Volume, E9_Zones.',
 'Derived: E4_G5_PairHeadways, Recomputed_Stats. Reported: Reported_Results (with MATCH/MISMATCH vs recomputation).',
 'Not digitised: Ayush Kumar Exp5 handwritten radar sheets (see AK_E5_Template); V-Box raw trip files (not in folder); Exp1 (no reports).',
]})
# ============================================================== Recomputed stats (clean)
rs = []
def add(grp, exp, sub, s, what, unit='km/h'):
    s = pd.Series(s, dtype=float).dropna()
    if len(s) == 0: return
    rs.append(dict(group=grp, experiment=exp, subset=sub, quantity=what, n=len(s), mean=round(s.mean(), 3), sd=round(s.std(), 3) if len(s) > 1 else None,
                   p15=round(pctl(s, 15), 3), p50=round(pctl(s, 50), 3), p85=round(pctl(s, 85), 3), p98=round(pctl(s, 98), 3), min=s.min(), max=s.max(), unit=unit))
ok = E3V[E3V.speed_plausibility == 'OK']
for (lane, cls), g in E3V.groupby(['lane', 'class_std']):
    add(G5, 'Exp3', f'{lane} {cls} all', g.speed_kmh_recomputed, 'Speed (50 m trap), all vehicles')
    add(G5, 'Exp3', f'{lane} {cls} plausible only', g[g.speed_plausibility == 'OK'].speed_kmh_recomputed, 'Speed, 5-90 km/h only')
for lane, g in E3V.groupby('lane'):
    add(G5, 'Exp4', lane, g.headway_to_leader_s, 'Time headway to leader', 's')
for (grp, cls, st), g in E5.groupby(['group', 'vehicle_class', 'flow_state']):
    add(grp, 'Exp5', f'{cls} {st}', g.speed_kmh, 'Spot speed')
for (grp, cls), g in E5.groupby(['group', 'vehicle_class']):
    add(grp, 'Exp5', f'{cls} combined', g.speed_kmh, 'Spot speed')
add(G5, 'Exp8', 'All vehicles', E8V.duration_recomputed_min, 'Parking duration', 'min')
for t_, g in E8V.groupby('vehicle_type'):
    add(G5, 'Exp8', t_, g.duration_recomputed_min, 'Parking duration', 'min')
RECOMP = pd.DataFrame(rs)

ISS = pd.DataFrame(ISSUES); REP = pd.DataFrame(REPORTED)
AKT = pd.DataFrame(columns=['sheet_page', 'column_as_printed', 'column_meaning_confirmed_by_owner', 'row', 'speed_kmh', 'circled_or_underlined(free_flow?)', 'notes'])

SHEETS = [('README', README), ('Catalog', CAT), ('Sources', SRCS), ('Validation_Log', ISS), ('Reported_Results', REP), ('Recomputed_Stats', RECOMP),
          ('E3E4_G5_Vehicles', E3V), ('E3_Volume', E3VOL), ('E4_G5_PairHeadways', E4PAIRS), ('E4_G12_HeadwayBins', E4BINS),
          ('E5_SpotSpeed', E5), ('E6_MovingObserver_Counts', E6C), ('E6_G5_Runs', R6), ('E7_TMC', E7),
          ('E8_G5_Parking_Vehicles', E8V), ('E8_G5_Accumulation', E8ACC), ('E8_G12_Parking_Bays', E8B),
          ('E9_G5_Interviews', E9I), ('E9_G12_PartialOD', E9P), ('E9_Volume', E9V), ('E9_Zones', ZONES), ('Data_Dictionary', DICT), ('AK_E5_Template', AKT)]
if __name__ == '__main__':
    xl = OUT + '/CE323_Lab_Dataset.xlsx'
    with pd.ExcelWriter(xl, engine='openpyxl') as w:
        for name, df in SHEETS:
            df.to_excel(w, sheet_name=name, index=False)
    for name, df in SHEETS:
        d = df.copy()
        for c in d.columns:
            if d[c].dtype == object: d[c] = d[c].map(lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else str(x))
        d.to_parquet(f'{OUT}/parquet/{name}.parquet', index=False)
        d.to_csv(f'{OUT}/csv_{name}.csv', index=False) if False else None
    print({n: len(d) for n, d in SHEETS})
    print(ISS.severity.value_counts().to_dict(), REP.check.value_counts().to_dict())
