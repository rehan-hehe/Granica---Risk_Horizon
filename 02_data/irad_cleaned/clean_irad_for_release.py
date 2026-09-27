"""Draft cleaner for the iRAD tables: drops columns whose names suggest personal data. REVIEW THE OUTPUT BY HAND.
Usage: python clean_irad_for_release.py <folder with iRAD_Assam_all__*.parquet> <out folder>"""
import sys, os, re, glob, pandas as pd
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
PII = re.compile(r'name|phone|mobile|contact|address|father|guardian|licen[cs]e|dl_|registration|reg_no|engine|chassis|aadhaar|aadhar|email|id_proof|narrat|remark|statement|signature', re.I)
KEEP_TABLES = ['Accidents', 'Vehicles', 'Road_Details', 'Data_Dictionary', 'Summary']
for f in glob.glob(os.path.join(src, 'iRAD_Assam_all__*.parquet')):
    t = f.split('__')[-1].replace('.parquet', '')
    if t not in KEEP_TABLES: continue
    d = pd.read_parquet(f); drop = [c for c in d.columns if PII.search(c)]
    d.drop(columns=drop).to_csv(os.path.join(out, f'{t.lower()}_clean.csv'), index=False)
    print(t, 'dropped:', drop)
