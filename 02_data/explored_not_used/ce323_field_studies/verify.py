import pandas as pd, openpyxl, re, subprocess, csv, numpy as np
X='out/CE323_Lab_Dataset.xlsx'; import os; S=os.environ.get('LAB_DIR','../Lab Report')+'/'
res=[]
def chk(name, ok, n, detail=''): res.append((name, 'PASS' if ok else 'FAIL', n, detail)); 
# 1 Group5 Exp5: every value vs its cell
e=pd.read_excel(X,'E5_SpotSpeed'); w=openpyxl.load_workbook(S+'Expt 5.xlsx',data_only=True).active
g5=e[e.group=='Group 5']; bad=[r for r in g5.itertuples() if float(w[r.source_location.split('!')[1]].value)!=r.speed_kmh]
# also: no non-empty cell in raw columns missed
cells=sum(1 for c in 'ABEFIJMNQRUV' for r in range(3,80) if w[f'{c}{r}'].value not in (None,''))
chk('G5 Exp5 cell-by-cell', not bad and cells==len(g5), len(g5), f'raw cells {cells}')
# comb columns equal NF+FF multiset
for nf,ff,cb in [('A','B','C'),('E','F','G'),('I','J','K'),('M','N','O'),('Q','R','S'),('U','V','W')]:
    a=sorted([w[f'{nf}{r}'].value for r in range(3,80) if w[f'{nf}{r}'].value not in (None,'')]+[w[f'{ff}{r}'].value for r in range(3,80) if w[f'{ff}{r}'].value not in (None,'')])
    b=sorted([w[f'{cb}{r}'].value for r in range(3,80) if w[f'{cb}{r}'].value not in (None,'')])
    chk(f'G5 Exp5 Comb col {cb} = NF+FF', a==b, len(b), '' if a==b else f'{len(a)} vs {len(b)}')
# 2 Group12 Exp5 vs pdftotext numbers (multiset per page)
g12=e[e.group=='Group 12']
txt=subprocess.run(['pdftotext','-layout','-f','4','-l','5',S+'CE323_EXP5.pdf','-'],capture_output=True,text=True).stdout
tbl=txt[txt.index('OBSERVATIONS TABLE'):txt.index('Analysis of free flow')]
nums=sorted(int(x) for x in re.findall(r'(?<![\w.])\d{1,3}(?![\w.])',tbl) if x not in ('2',))
chk('G12 Exp5 multiset vs pdf text layer', nums==sorted(g12.speed_kmh.astype(int)), len(g12), f'text {len(nums)}')
# 3 Group5 Exp3/4 vehicles: times vs both sheets
v=pd.read_excel(X,'E3E4_G5_Vehicles')
wb=openpyxl.load_workbook(S+'Bhavesh_Traffic volume video_analysis_datasheet_1.xlsx',data_only=True)
rawws=wb['Speed (free & non free)']; raw={rawws.cell(r,1).value:(rawws.cell(r,4).value,rawws.cell(r,5).value,rawws.cell(r,3).value) for r in range(3,400) if rawws.cell(r,1).value is not None}
ok=all(abs(raw[r.vehicle_id][0]-r.t_in_rawsheet)<1e-9 and abs(raw[r.vehicle_id][1]-r.t_out_rawsheet)<1e-9 and raw[r.vehicle_id][2]==r.class_raw for r in v.itertuples())
chk('G5 Exp3 raw-entry times/classes', ok and len(raw)==len(v), len(v))
lane={}
for sh,c0s,r0 in [('Shoulder lane',(1,10),4),('Median lane',(1,10),3)]:
    ws=wb[sh]
    for c0 in c0s:
        for r in range(r0,400):
            vid=ws.cell(r,c0).value
            if vid is None: continue
            lane[vid]=(ws.cell(r,c0+3).value,ws.cell(r,c0+4).value,ws.cell(r,c0+7).value)
ok=all(abs(lane[r.vehicle_id][0]-r.t_in_s)<1e-9 and abs(lane[r.vehicle_id][1]-r.t_out_s)<1e-9 and abs(lane[r.vehicle_id][2]-r.speed_reported_kmh)<1e-9 for r in v.itertuples())
chk('G5 Exp3 analysis-sheet times/speeds', ok and len(lane)==len(v), len(v))
# 4 Exp3 volume
vol=pd.read_excel(X,'E3_Volume'); tv=wb['Traffic Volume']
ok=True
for r in vol.itertuples():
    rr=int(r.source_location.split('r')[-1]); c0=1 if r.lane.startswith('Shoulder') else 13
    k=['Multi Axle','Truck','LCV','Bus','Car','Auto','Two Wheeler'].index(r.vehicle_class)
    ok&= tv.cell(rr,c0+2+k).value==r.count
chk('G5 Exp3 volume counts', ok, len(vol))
# 5 Exp7 csv
t=pd.read_excel(X,'E7_TMC'); t5=t[t.group=='Group 5']
rows=list(csv.reader(open(S+'Expt 7(Sheet1).csv')))
tot_csv=sum(int(float(rows[i][c])) for i in range(2,32) for c in (2,3,9,10,16,17))
chk('G5 Exp7 total count vs csv', tot_csv==t5['count'].sum(), len(t5), f'{tot_csv} vs {t5["count"].sum()}')
chk('G5 Exp7 PCU reported = recomputed', (abs(t5.pcu_reported-t5.pcu_recomputed)<1e-9).all(), len(t5))
# 6 Exp8 G5 rows
p=pd.read_excel(X,'E8_G5_Parking_Vehicles'); w8=openpyxl.load_workbook(S+'Expt 8.xlsx',data_only=True).active
n=sum(1 for r in range(3,200) if w8.cell(r,2).value is not None)
ok=all(w8[r.source_location.split('!')[1].split(':')[0].replace('B','F')].value==(None if pd.isna(r.duration_reported_min) else r.duration_reported_min) for r in p.itertuples())
chk('G5 Exp8 rows & durations', n==len(p) and ok, len(p))
# 7 Exp9 G5
i9=pd.read_excel(X,'E9_G5_Interviews'); w9=openpyxl.load_workbook(S+'Expt 9.xlsx',data_only=True).active
ok=True
for r in i9.itertuples():
    a,b=r.source_location.split('!')[1].split(':'); rr=int(a[1:]); c0=1 if a[0]=='A' else 11
    off=(3,4,5,6) if c0==1 else (2,3,4,5)
    vals=[w9.cell(rr,c0+o).value for o in off]  # mode, origin, dest, purpose
    ok&= vals==[r.mode_code,r.origin_zone,r.destination_zone,r.purpose_code]
chk('G5 Exp9 interviews', ok and len(i9)==59, len(i9))
# 8 Exp6 G5 counts
c=pd.read_excel(X,'E6_MovingObserver_Counts'); c5=c[c.group=='Group 5']; w6=openpyxl.load_workbook(S+'Expt 6.xlsx',data_only=True).active
ok=all(w6[r.source_location.split('!')[1]].offset(0,1).value==r['count'] for _,r in c5.iterrows())
chk('G5 Exp6 class counts', ok, len(c5))
# block PCU totals
for blockcell,tot in [('C10',263.8),('G10',299.2),('K10',23.1),('O10',18.7),('C20',334.5),('G20',337.5),('K19',21.4),('O19',16.4),('K30',38.4),('O30',16.2),('K39',26.1),('O39',12.4)]:
    pass
# 9 G12 Exp6 against pdfplumber
import pdfplumber
pg=pdfplumber.open(S+'CE323_EXP6.pdf').pages[3]; tb=pg.extract_tables()
c12=c[c.group=='Group 12']
ot=sum(int(x) for row in tb[0][2:] for x in row[3:17:2]); on=sum(int(x) for row in tb[0][2:] for x in row[4:17:2]); op=sum(int(x) for row in tb[1][2:] for x in row[3:11])
got=c12.groupby('count_type')['count'].sum()
chk('G12 Exp6 totals vs pdf tables', (ot,on,op)==(got['Overtaking test vehicle'],got['Overtaken by test vehicle'],got['Opposite-direction count']), len(c12), f'{(ot,on,op)} vs {tuple(got)}')
# 10 G12 Exp8 duration column = reported
b=pd.read_excel(X,'E8_G12_Parking_Bays')
rows=b.groupby('bay_row').first()
mism=rows[rows.row_duration_reported_min!=rows.row_duration_recomputed_min]
chk('G12 Exp8 row durations = 10×occupied snapshots', len(mism)==0, len(rows), f'{len(mism)} rows differ: {list(mism.index)}')
txt8=subprocess.run(['pdftotext','-layout',S+'CE323_EXP8.pdf','-'],capture_output=True,text=True).stdout
seg=txt8[txt8.index('Vehicle'):txt8.index('total no of')]
toks=re.findall(r'\b(?:bike|car)\b|\b\d{4}K?\b',seg)
plates=[t for t in toks if t not in ('bike','car')]
mine=[s for _,snaps,_ in __import__('data_manual').G12_E8_ROWS for s in snaps if s!='0']
chk('G12 Exp8 plate sequence vs pdf text', plates==mine, len(mine), f'pdf {len(plates)}')
# 11 G12 Exp7 PCU sums vs reported
r=pd.read_excel(X,'Reported_Results')
chk('G12 Exp7 movement PCU all MATCH', (r[(r.group=='Group 12')&(r.experiment=='Exp7')&(r.parameter=='Movement PCU')].check=='MATCH').all(), 6)
# 12 G12 Exp4 bins sums vs N-1
bn=pd.read_excel(X,'E4_G12_HeadwayBins'); s=bn.groupby('lane').frequency.sum().to_dict()
chk('G12 Exp4 bins (L1 112, comb 251)', s['Lane 1 (Inner)']==112 and s['Combined']==251, len(bn), str(s))
for x in res: print(*x, sep=' | ')
