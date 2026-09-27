"""Rebuilds 03_eda/05_live_traffic and 02_data/public/tomtom_live from the collector output.
Run from the folder that contains Risk_Horizon_Data/ (hourly parquet files in data/live, manifest.jsonl) and risk-horizon-github/.
"""
import glob, json, os, pandas as pd, numpy as np
L='Risk_Horizon_Data/data/live'; G='risk-horizon-github'; E=f'{G}/03_eda/05_live_traffic'; P=f'{G}/02_data/public/tomtom_live'
ft=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{L}/flow_tiles_*.parquet'))],ignore_index=True)
sp=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{L}/segment_points_*.parquet'))],ignore_index=True)
ft=ft.drop_duplicates(['time_utc','tile','geom_hash']).sort_values('time_utc'); sp=sp.drop_duplicates(['time_utc','point_id']).sort_values('time_utc')
IST=lambda s: s.dt.tz_convert('Asia/Kolkata')
# ---- raw all-polls CSVs (public folder)
ft.to_csv(f'{P}/flow_tiles_all_polls.csv',index=False); sp.to_csv(f'{P}/segment_points_all_polls.csv',index=False)
# ---- per-poll time series
ft['poll']=ft.time_utc.dt.floor('min')
ts=ft.groupby('poll').agg(tiles=('tile','nunique'),road_pieces=('geom_hash','nunique'),mean_level=('traffic_level','mean'),
    slow_share=('traffic_level',lambda v:(v<0.5).mean()),closures=('road_closure',lambda v:(v==True).sum())).reset_index()
ts['time_ist']=IST(ts.poll).dt.strftime('%Y-%m-%d %H:%M'); ts['cumulative_readings']=ts.road_pieces.cumsum()
ts[['time_ist','tiles','road_pieces','cumulative_readings','mean_level','slow_share','closures']].round(4).to_csv(f'{E}/flow_polls_timeseries.csv',index=False)
# ---- manifest (failures)
man=[json.loads(l) for l in open('Risk_Horizon_Data/manifest.jsonl') if 'tomtom' in l]
M=pd.DataFrame(man); M.to_csv(f'{E}/collection_log_tomtom.csv',index=False)
fl=M[M.source=='tomtom-flow-tiles']; fail=fl[fl.errors>0]
# ---- segment points
sp['ratio']=sp.current_speed_kmh/sp.free_flow_speed_kmh.replace(0,np.nan); sp['hour_ist']=IST(sp.time_utc).dt.hour
sps=sp.groupby(['site','point_id']).agg(polls=('ratio','size'),mean_speed_kmh=('current_speed_kmh','mean'),free_flow_kmh=('free_flow_speed_kmh','median'),
    min_ratio=('ratio','min'),mean_ratio=('ratio','mean'),hours_below_0_7=('ratio',lambda v:int((v<0.7).sum()))).round(3).reset_index()
sps.to_csv(f'{E}/segment_points_summary.csv',index=False)
ft['hour_ist']=IST(ft.time_utc).dt.hour
hp=pd.DataFrame({'tile_mean_level':ft.groupby('hour_ist').traffic_level.mean(),'tile_road_pieces_per_poll':ts.assign(h=IST(ts.poll).dt.hour).groupby('h').road_pieces.mean(),
    'points_mean_ratio':sp.groupby('hour_ist').ratio.mean()}).round(3); hp.index.name='hour_ist'; hp.to_csv(f'{E}/hour_of_day_profile.csv')
first,last=ts.poll.min(),ts.poll.max()
Mf=fl.assign(ts=pd.to_datetime(fl.ts)).sort_values('ts'); gd=Mf.ts.diff(); gaps=Mf[gd>pd.Timedelta('25min')].assign(gap_minutes=(gd[gd>pd.Timedelta('25min')].dt.total_seconds()/60).round(0))
gaps['gap_starts_after']=Mf.ts.shift(1)[gd>pd.Timedelta('25min')].dt.strftime('%Y-%m-%d %H:%M'); gaps['resumed_at']=gaps.ts.dt.strftime('%Y-%m-%d %H:%M')
gaps[['gap_starts_after','resumed_at','gap_minutes']].to_csv(f'{E}/collection_gaps.csv',index=False)
st=[('first flow poll (IST)',IST(pd.Series([first]))[0].strftime('%Y-%m-%d %H:%M')),('last flow poll in data (IST)',IST(pd.Series([last]))[0].strftime('%Y-%m-%d %H:%M')),
 ('hours covered',round((last-first).total_seconds()/3600,1)),('flow polls in data',len(ts)),('flow-tile readings (road piece x poll)',len(ft)),
 ('distinct road pieces seen',ft.geom_hash.nunique()),('tiles per poll',int(ts.tiles.median())),
 ('segment-point polls',sp.time_utc.dt.floor('min').nunique()),('segment-point readings',len(sp)),('hotspot points',sp.point_id.nunique()),
 ('flow polls logged in manifest',len(fl)),('flow polls with errors (recovered next poll)',len(fail)),('last manifest entry (IST)',M.ts.max()),('gaps > 25 min between polls',len(gaps)),('longest gap (minutes)',int(gaps.gap_minutes.max()) if len(gaps) else 0)]
pd.DataFrame(st,columns=['item','value']).to_csv(f'{E}/collection_status.csv',index=False)
print(pd.DataFrame(st)); print(hp)
# ---- figures
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.dates as md
gapmask=ts.poll.diff()>pd.Timedelta('40min')
ts2=pd.concat([ts,pd.DataFrame({'poll':ts.poll[gapmask]-pd.Timedelta('1min')})]).sort_values('poll')
t=IST(ts2.poll)
fig,ax=plt.subplots(figsize=(12,4.4),dpi=140); ax.fill_between(t,ts2.road_pieces,color='#137F74',alpha=.25); ax.plot(t,ts2.road_pieces,color='#137F74',lw=1.6)
ax.set_ylabel('road pieces with live readings / poll',color='#137F74'); a2=ax.twinx(); a2.plot(t,ts2.mean_level,'k--',lw=1.2); a2.set_ylim(0,1); a2.set_ylabel('mean speed ÷ free-flow')
ax.xaxis.set_major_formatter(md.DateFormatter('%d %b %H:%M',tz=t.dt.tz)); ax.set_title(f'TomTom live traffic, Guwahati corridor · {len(ts)} polls, {len(ft):,} readings · {st[0][1]} → {st[1][1]} IST',loc='left',fontsize=10)
ax.spines[['top']].set_visible(False); plt.tight_layout(); plt.savefig(f'{E}/live_traffic_timeseries.png'); plt.close()
fig,ax=plt.subplots(figsize=(12,4.2),dpi=140); ax.plot(IST(ts.poll),ts.cumulative_readings/1000,color='#14181D',lw=2); ax.set_ylabel('cumulative readings (thousands)')
ax.xaxis.set_major_formatter(md.DateFormatter('%d %b %H:%M',tz=t.dt.tz)); ax.set_title('How the live dataset grew',loc='left',fontsize=10); ax.spines[['top','right']].set_visible(False); plt.tight_layout(); plt.savefig(f'{E}/live_dataset_growth.png'); plt.close()
pv=sp.assign(t=IST(sp.time_utc).dt.floor('h')).pivot_table(index='point_id',columns='t',values='ratio')
pv=pv.reindex(columns=pd.date_range(pv.columns.min(),pv.columns.max(),freq='h'))
fig,ax=plt.subplots(figsize=(12,6),dpi=140); im=ax.imshow(pv.values,aspect='auto',cmap=matplotlib.colormaps['RdYlGn'].with_extremes(bad='#dddddd'),vmin=0.3,vmax=1)
ax.set_yticks(range(len(pv))); ax.set_yticklabels(pv.index,fontsize=7); xs=range(0,pv.shape[1],max(1,pv.shape[1]//8)); ax.set_xticks(list(xs)); ax.set_xticklabels([pv.columns[i].strftime('%d %b %H:%M') for i in xs],fontsize=8)
plt.colorbar(im,label='speed ÷ free-flow'); ax.set_title('24 hotspot points, hour by hour (red = slower than free flow, grey = no poll)',loc='left',fontsize=10); plt.tight_layout(); plt.savefig(f'{E}/hotspot_points_heatmap.png'); plt.close()
fig,ax=plt.subplots(figsize=(8,3.8),dpi=140); ax.plot(hp.index,hp.tile_mean_level,'o-',label='tiles: mean speed ÷ free-flow'); ax.plot(hp.index,hp.points_mean_ratio,'s-',label='24 points: mean speed ÷ free-flow')
ax.set_xlabel('hour of day (IST)'); ax.set_ylim(0,1.05); ax.legend(fontsize=8); ax.set_xticks(range(0,24,3)); ax.spines[['top','right']].set_visible(False); ax.set_title('Traffic by hour of day',loc='left',fontsize=10); plt.tight_layout(); plt.savefig(f'{E}/hour_of_day_profile.png'); plt.close()
