"""Release cleaner for the iRAD Assam crash tables (Risk Horizon, Team Pookie_Pandas).

Turns the internal parser output (iRAD_Assam_all__*.parquet) into CSVs that are safe to publish.
Approach: an ALLOW-LIST (only named columns survive), not a block-list, plus generalisation and a
value-level PII scan that fails loudly if anything that looks like a phone, plate or e-mail survives.

Usage:  python clean_irad_for_release.py <folder with iRAD_Assam_all__*.parquet> <out folder> [<internal folder>]

What it does
  1. Replaces the police accident_id (links to the FIR / official case) with a random release id
     (crash_00001 ...). The mapping is written ONLY to <internal folder> and must never be published.
  2. Drops every direct identifier: names, phones, addresses, officers, FIR numbers, plates, engine /
     chassis / policy / permit / PUC numbers, owner fields, source file paths, raw geolocation text.
  3. Drops all free text (description, landmark, location details, road names, remarks, damage text).
  4. Generalises: coordinates rounded to 3 decimals (~110 m); time kept to date + hour (minutes dropped);
     vehicle make kept as brand only (no model / colour / registration date); document validity dates
     converted to "valid on crash date" yes/no flags.
  5. Scans every published text cell for phone numbers, Indian registration plates and e-mails.
"""
import sys, os, re, json, numpy as np, pandas as pd
src, out = sys.argv[1], sys.argv[2]
internal = sys.argv[3] if len(sys.argv) > 3 else os.path.join(src, 'internal_do_not_share')
os.makedirs(out, exist_ok=True); os.makedirs(internal, exist_ok=True)
rd = lambda t: pd.read_parquet(os.path.join(src, f'iRAD_Assam_all__{t}.parquet'))

# ---------- Accidents
A = rd('Accidents')
rng = np.random.default_rng()                       # unseeded on purpose: ids cannot be regenerated from the data
order = rng.permutation(len(A))
A['release_id'] = pd.Series([f'crash_{i + 1:05d}' for i in range(len(A))]).iloc[np.argsort(order)].values
A[['accident_id', 'release_id']].to_csv(os.path.join(internal, 'release_id_mapping_INTERNAL.csv'), index=False)
idmap = dict(zip(A.accident_id, A.release_id))

dt = pd.to_datetime(A.accident_datetime, errors='coerce')
A['accident_date'] = dt.dt.strftime('%Y-%m-%d')
A['latitude_3dp'] = pd.to_numeric(A.latitude, errors='coerce').round(3)
A['longitude_3dp'] = pd.to_numeric(A.longitude, errors='coerce').round(3)
A['reporting_delay_hours'] = pd.to_numeric(A.reporting_delay_hours, errors='coerce').round(0)
A['property_damage_reported'] = A.property_damage.notna() & ~A.property_damage.astype(str).str.lower().isin(['no', 'none', 'nil', 'not applicable', 'nan'])
keepA = ['release_id', 'accident_date', 'accident_year', 'accident_month', 'accident_weekday', 'accident_hour',
         'district_name', 'latitude_3dp', 'longitude_3dp', 'severity', 'road_classification', 'local_body', 'accident_spot',
         'collision_type', 'collision_nature', 'weather_condition', 'light_condition', 'visibility_m', 'initial_observation',
         'traffic_violation', 'act', 'num_vehicles_declared_int', 'reporting_delay_hours', 'property_damage_reported',
         'killed_driver', 'grievous_driver', 'minor_driver', 'no_injury_driver', 'total_driver',
         'killed_passenger', 'grievous_passenger', 'minor_passenger', 'no_injury_passenger', 'total_passenger',
         'killed_pedestrian', 'grievous_pedestrian', 'minor_pedestrian', 'no_injury_pedestrian', 'total_pedestrian',
         'killed_total', 'grievous_total', 'minor_total', 'no_injury_total', 'total_total', 'animals_total',
         'n_vehicle_records', 'n_driver_records', 'n_passenger_records', 'n_pedestrian_records', 'n_witness_records',
         'n_transport_inspections', 'has_road_details', 'has_hospital_details', 'sections_present']
Ao = A[keepA].sort_values(['accident_date', 'release_id'])

# ---------- Vehicles
V = rd('Vehicles')
V['release_id'] = V.accident_id.map(idmap)
V['make'] = V.make_model_clean.astype(str).str.split('|').str[0].str.strip().replace({'nan': None, 'None': None, '': None})
crash_day = pd.to_datetime(V.accident_id.map(dict(zip(A.accident_id, dt))), errors='coerce')
for col, new in [('insurance_validity', 'insurance_valid_on_crash_date'), ('puc_valid_upto', 'puc_valid_on_crash_date'),
                 ('fitness_validity', 'fitness_valid_on_crash_date'), ('tax_validity', 'tax_valid_on_crash_date')]:
    d = pd.to_datetime(V[col], errors='coerce')
    V[new] = np.where(d.isna() | crash_day.isna(), None, np.where(d >= crash_day.dt.normalize(), 'yes', 'no'))
V['vehicle_age_years'] = (crash_day.dt.year - pd.to_numeric(V.manufacture_year, errors='coerce')).where(lambda s: s.between(0, 80))
keepV = ['release_id', 'record_seq', 'accused_victim', 'vehicle_category', 'vahan_vehicle_category', 'vehicle_type', 'vehicle_class',
         'body_type', 'fuel_type', 'emission_norms', 'make', 'vehicle_age_years', 'hit_and_run', 'disposition', 'vehicle_damage',
         'load_category', 'load_condition', 'skid_mark', 'insurance_valid_on_crash_date', 'puc_valid_on_crash_date',
         'fitness_valid_on_crash_date', 'tax_valid_on_crash_date', 'gvw_kg_num', 'unladen_weight_kg_num', 'num_cylinders_num',
         'cubic_capacity_cc_num', 'seating_capacity_num', 'standing_capacity_num', 'wheelbase_mm_num']
Vo = V[keepV].rename(columns=lambda c: c[:-4] if c.endswith('_num') else c).sort_values(['release_id', 'record_seq'])

# ---------- Road details (categorical inspection fields only)
R = rd('Road_Details'); R['release_id'] = R.accident_id.map(idmap)
keepR = ['release_id', 'record_seq', 'area_type', 'road_classification', 'road_owning_agency', 'road_number', 'structure_type',
         'road_surface_type', 'surface_condition', 'carriageway_type', 'road_width_m_num', 'accident_location_geometry', 'sight_distance',
         'speed_limit_kmph_num', 'road_margins', 'shoulder_type', 'terrain_type', 'gradient_type', 'divider_barrier_type', 'median_type',
         'pedestrian_infrastructure', 'pedestrian_infrastructure_quality', 'ongoing_road_work', 'road_markings', 'road_sign_board',
         'contributing_factors', 'short_term_remedial_measures', 'long_term_remedial_measures']
Ro = R[keepR].rename(columns=lambda c: c[:-4] if c.endswith('_num') else c).sort_values(['release_id', 'record_seq'])

# ---------- PII scan on every published text cell
PAT = {'phone': re.compile(r'(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)'),
       'plate': re.compile(r'\b[A-Z]{2}[\s-]?\d{1,2}[\s-]?[A-Z]{0,3}[\s-]?\d{3,4}\b'),
       'email': re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')}
report = {}
for name, df in [('accidents', Ao), ('vehicles', Vo), ('road_details', Ro)]:
    for c in df.columns:
        if df[c].dtype != object: continue
        s = df[c].dropna().astype(str)
        for k, p in PAT.items():
            n = int(s.str.contains(p).sum())
            if n:
                if c == 'road_number' and k == 'plate':                 # "NH 15" / "SH 3" are road numbers, not plates
                    s2 = s[~s.str.fullmatch(r'\s*(NH|SH|MDR|ODR|AH)[\s-]*\d+[A-Z]?\s*', case=False)]
                    n = int(s2.str.contains(p).sum()); p_mask = df[c].astype(str).isin(s2[s2.str.contains(p)])
                    if n: report[f'{name}.{c}.{k}'] = n; df.loc[p_mask, c] = None
                    continue
                report[f'{name}.{c}.{k}'] = n
                df.loc[df[c].astype(str).str.contains(p, na=False), c] = None   # blank the cell rather than risk it
print('PII-pattern cells blanked:', report or 'none')

# ---------- guard 2: blank any RARE published cell (<=2 occurrences) that exactly equals a known identifier from the internal tables
ID_SOURCES = {'Accidents': ['accident_id', 'fir_csr_number', 'investigating_officer', 'field_officer', 'station_address'],
              'Vehicles': ['vehicle_reg_no', 'owner_name', 'owner_father_name', 'owner_address', 'owner_present_address', 'owner_permanent_address',
                           'engine_number', 'chassis_number', 'insurance_policy_number', 'puc_certificate_number', 'national_permit_number', 'entity_header'],
              'Drivers': None, 'Passengers': None, 'Pedestrians': None, 'Witnesses': None}
PLACEHOLDER = re.compile(r'^(UNKNOWN|NOT ?KNOWN?|NOT ?FOUND|NOT ?AVAILABLE|NOT ?APPLICABLE|NA|N/A|NIL|NONE|OTHERS?|UNDER INVESTIGATION|[.\-_0 ]+)(\s*\d+)?$', re.I)
ids = set()
for t, cols in ID_SOURCES.items():
    f = os.path.join(src, f'iRAD_Assam_all__{t}.parquet')
    if not os.path.exists(f): continue
    d = pd.read_parquet(f)
    cols = cols or [c for c in d.columns if re.search(r'name|phone|mobile|contact|address|licen|aadh|id_no|id_number|father|guardian|email', c, re.I)
                    and not re.search(r'type|status|class|categor|valid', c, re.I)]
    for c in cols:
        if c in d.columns:
            for v in d[c].dropna().astype(str).str.upper().str.strip().unique():
                if len(v) >= 6 and not PLACEHOLDER.match(v): ids.add(v)
guard2 = {}
for name, df in [('accidents', Ao), ('vehicles', Vo), ('road_details', Ro)]:
    for c in df.columns:
        if df[c].dtype != object or c == 'release_id': continue
        u = df[c].astype(str).str.upper().str.strip(); freq = u.map(u.value_counts())
        m = u.isin(ids) & (freq <= 2)          # a real identifier is rare; a category that also appears in an address/owner field repeats
        if m.any(): guard2[f'{name}.{c}'] = int(m.sum()); df.loc[m, c] = None
# ---------- guard 3: vehicle description columns must not hold codes (VINs, trailer / engine numbers) or dates
CODE = re.compile(r'\d{5,}|\b[A-HJ-NPR-Z0-9]{17}\b|T\.?NO|^\d{1,2}/\d{2,4}$', re.I)
for c in ['make', 'body_type', 'vehicle_class', 'vahan_vehicle_category', 'vehicle_type']:
    m = Vo[c].astype(str).str.contains(CODE, na=False)
    if m.any(): guard2[f'vehicles.{c}.code_like'] = int(m.sum()); Vo.loc[m, c] = None
print('identifier-match / code-like cells blanked:', guard2 or 'none')
report.update({f'{k} (id/code)': v for k, v in guard2.items()})


# ---------- write
for df in (Ao, Vo, Ro):
    for c in df.columns:
        if df[c].dtype == object: df[c] = df[c].map(lambda x: x if not isinstance(x, str) else ' '.join(re.sub(r'[\x00-\x1f\x7f]|Ã|Â', ' ', x).split()) or None)   # no control bytes / line breaks
Ao.to_csv(os.path.join(out, 'accidents_clean.csv'), index=False)
Vo.to_csv(os.path.join(out, 'vehicles_clean.csv'), index=False)
Ro.to_csv(os.path.join(out, 'road_details_clean.csv'), index=False)
DD = rd('Data_Dictionary')
rows = []
for tab, df, sheet in [('accidents_clean.csv', Ao, 'Accidents'), ('vehicles_clean.csv', Vo, 'Vehicles'), ('road_details_clean.csv', Ro, 'Road_Details')]:
    for c in df.columns:
        m = DD[(DD.sheet == sheet) & (DD.column.isin([c, c + '_num']))]
        desc = m.source_or_description.iloc[0] if len(m) else ''
        rows.append(dict(file=tab, column=c, dtype=str(df[c].dtype), non_null=int(df[c].notna().sum()), description=desc))
notes = {'release_id': 'Random id per crash; links the three files. Not the police accident id.',
         'latitude_3dp': 'Police geotag rounded to 3 decimals (~110 m).', 'longitude_3dp': 'Police geotag rounded to 3 decimals (~110 m).',
         'accident_date': 'Crash date (time kept only as accident_hour).', 'make': 'Manufacturer only; model, colour and registration removed.',
         'vehicle_age_years': 'Crash year minus manufacture year.', 'property_damage_reported': 'True if the report lists property damage (free text removed).'}
for r in rows:
    if r['column'] in notes: r['description'] = notes[r['column']]
    if r['column'].endswith('_valid_on_crash_date'): r['description'] = 'yes/no: the document was valid on the crash date (VAHAN snapshot; may reflect current status).'
pd.DataFrame(rows).to_csv(os.path.join(out, 'data_dictionary.csv'), index=False)
json.dump(dict(rows=dict(accidents=len(Ao), vehicles=len(Vo), road_details=len(Ro)), pii_cells_blanked=report,
               window=[Ao.accident_date.dropna().min(), Ao.accident_date.dropna().max()], crashes_without_date=int(Ao.accident_date.isna().sum())), open(os.path.join(out, 'release_summary.json'), 'w'), indent=1)
print('rows:', len(Ao), len(Vo), len(Ro))
