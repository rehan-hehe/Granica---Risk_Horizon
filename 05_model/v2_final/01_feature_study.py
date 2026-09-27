"""Feature study for Stage A (static risk): engineering, redundancy, PCA, selection, tuning — all on a TIME-SAFE protocol.

  Selection-train window  : Jan 2023 - Jun 2024  (features + target)
  Selection-validation    : Jul 2024 - Jun 2025  (the ranking is judged here)
  Final test (untouched)  : Jul 2025 - May 2026  (used only in 02_static_risk.py)
Primary metric: capture-curve area up to 20% of road length (share of validation KSI crashes found in the riskiest
0-20% of road, averaged over that range; 1.0 = perfect). Also capture @1/5/10% and a level-free Poisson deviance.
Model scores are out-of-fold over 5 spatial folds (0.2° blocks): a segment is never scored by a model that saw its area.

Input : 2_readable/10_model_inputs/*.csv (+ internal crash_events.csv for crash dates, aggregated here)
Output: 2_readable/11_model_results/0_feature_study/*  and  3_code/07_model/final_features.json
"""
import os, sys, json, time, warnings, numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rh_features as RF

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INP = os.path.join(ROOT, '2_readable', '10_model_inputs')
OUT = os.path.join(ROOT, '2_readable', '11_model_results', '0_feature_study'); os.makedirs(OUT, exist_ok=True)
t0 = time.time(); LOG = []
def log(msg): print(f'[{time.time() - t0:5.0f}s] {msg}', flush=True)

S, F, DEF, xyG = RF.static_features(INP)
E = pd.read_csv(os.path.join(INP, 'internal_do_not_share', 'crash_events.csv'))
WA, WV = RF.month_list('2023-01', '2024-06'), RF.month_list('2024-07', '2025-06')
GEO = dict(seg_lat=S.mid_lat.values, seg_lon=S.mid_lon.values, seg_len=S.length_m.values)
H, ownA, DEFH = RF.history_features(E, S.segment_id.values, xyG, WA, **GEO)
_, ownV, _ = RF.history_features(E, S.segment_id.values, xyG, WV, **GEO)
DEF = pd.concat([DEF, DEFH], ignore_index=True)
y, yv = ownA['ksi'], ownV['ksi']; L = S.length_m.values
X_all = pd.concat([F.drop(columns='segment_id'), H], axis=1)
log(f'candidate features: {X_all.shape[1]} | train-A KSI {y.sum():.0f} ({ownA["data_months"]} months) | validation KSI {yv.sum():.0f} ({ownV["data_months"]} months)')

blk = np.floor(S.mid_lat / 0.2).astype(int).astype(str) + '_' + np.floor(S.mid_lon / 0.2).astype(int).astype(str)
ub = blk.unique(); fold = blk.map(dict(zip(ub, np.random.default_rng(42).integers(0, 5, len(ub))))).values
BASE_P = dict(loss='poisson', max_iter=250, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0, random_state=0)

def metrics(score, target):
    o = np.argsort(-(score + 1e-9 * L)); cum = np.cumsum(L[o]) / L.sum(); cap = np.cumsum(target[o]) / target.sum()
    grid = np.linspace(0.001, 0.20, 200); cc = np.interp(grid, cum, cap)
    return dict(aucc20=round(float(cc.mean()), 4), cap1=round(float(np.interp(0.01, cum, cap)), 4), cap5=round(float(np.interp(0.05, cum, cap)), 4),
                cap10=round(float(np.interp(0.10, cum, cap)), 4))
def pdev_levelfree(mu, target):
    mu = np.maximum(mu, 1e-9) * target.sum() / np.maximum(mu, 1e-9).sum()
    t = np.where(target > 0, target * np.log(np.maximum(target, 1e-9) / mu), 0); return round(float(2 * np.sum(t - (target - mu))), 0)
def eb(mu, yy):
    k = max(float(((yy - mu) ** 2 - mu).sum() / (mu ** 2).sum()), 1e-3); w = 1 / (1 + k * mu); return w * mu + (1 - w) * yy
def oof_gbm(X, params=BASE_P, target=None):
    target = y if target is None else target; oof = np.zeros(len(X))
    for f in range(5):
        tr, te = fold != f, fold == f
        oof[te] = HistGradientBoostingRegressor(**params).fit(X[tr], target[tr]).predict(X[te])
    return oof
def run(step, desc, cols, params=BASE_P, target=None, X=None, keep=True):
    X = X_all[cols] if X is None else X
    mu = oof_gbm(X, params, target)
    if target is not None: mu = mu * y.sum() / mu.sum()              # other targets: rescale to KSI level for EB
    m, me = metrics(mu, yv), metrics(eb(mu, y), yv)
    row = dict(step=step, experiment=desc, n_features=X.shape[1], val_aucc20=m['aucc20'], val_cap1=m['cap1'], val_cap5=m['cap5'], val_cap10=m['cap10'],
               val_deviance=pdev_levelfree(mu, yv), eb_val_aucc20=me['aucc20'], eb_val_cap5=me['cap5'])
    if keep: LOG.append(row)
    log(f"{step:<14} {desc[:60]:<60} n={X.shape[1]:>3}  AUCC20={m['aucc20']:.4f} cap5={m['cap5']:.4f}  EB={me['aucc20']:.4f}")
    return row, mu

# ---------------- baselines
past = metrics(y + 0.01 * ownA['crashes'], yv)
LOG.append(dict(step='0 baseline', experiment='past crashes on the segment (blackspot-list approach)', n_features=0, val_aucc20=past['aucc20'], val_cap1=past['cap1'],
                val_cap5=past['cap5'], val_cap10=past['cap10']))
log(f"baseline past crashes AUCC20={past['aucc20']:.4f} cap5={past['cap5']:.4f}")
V1 = ['length_m', 'sinuosity', 'turn_deg_per_km', 'is_main_road', 'is_national_highway', 'is_bridge', 'junctions', 'junctions_per_km', 'n_rail_crossing',
      'n_speed_breaker', 'n_signal', 'n_ped_crossing', 'n_bus_stop', 'poi200_market_shop', 'poi200_education', 'poi200_health', 'poi200_transit', 'poi200_eatery',
      'poi200_worship', 'poi200_fuel', 'poi200_total', 'night_light', 'nearby_ksi_1km', 'nearby_crashes_1km', 'road_km_1km', 'mid_lat', 'mid_lon', 'hw_rank']
run('0 baseline', 'v1 feature set (previous model, same protocol)', V1)
run('1 engineered', 'all candidate features', list(X_all.columns))

# ---------------- 2 profiling: missing, sparsity, Spearman, mutual information
smp = np.random.default_rng(0).choice(len(S), 60000, replace=False)
mi = mutual_info_regression(X_all.iloc[smp].values, y[smp], random_state=0, n_neighbors=5)
prof = pd.DataFrame({'feature': X_all.columns, 'missing_share': X_all.isna().mean().values.round(4), 'zero_share': (X_all == 0).mean().values.round(4),
                     'spearman_with_target': [round(pd.Series(X_all[c].values).corr(pd.Series(y), method='spearman'), 4) for c in X_all.columns],
                     'mutual_information': mi.round(5)}).merge(DEF, on='feature', how='left')

# ---------------- 3 redundancy: Spearman clusters |rho| >= 0.85, keep the member with the highest MI
rho = np.nan_to_num(X_all.rank().corr().values.copy()); np.fill_diagonal(rho, 1)
Z = linkage(squareform(1 - np.abs(rho), checks=False), 'average'); cl = fcluster(Z, 0.15, 'distance')
prof['redundancy_cluster'] = cl
keep, dropped = [], {}
for c in np.unique(cl):
    mem = prof[prof.redundancy_cluster == c].sort_values('mutual_information', ascending=False).feature.tolist()
    keep.append(mem[0]); dropped.update({m: f'redundant with {mem[0]} (|rho|>=0.85)' for m in mem[1:]})
low_info = [f for f in keep if (prof.set_index('feature').loc[f, 'zero_share'] > 0.998)]
dropped.update({f: 'almost always zero (>99.8%)' for f in low_info}); keep = [f for f in keep if f not in low_info]
pd.DataFrame(np.round(rho, 3), index=X_all.columns, columns=X_all.columns).to_csv(os.path.join(OUT, 'spearman_correlation_matrix.csv'))
log(f'redundancy: {X_all.shape[1]} -> {len(keep)} features; dropped {sorted(dropped)}')
r_red, _ = run('3 redundancy', 'after removing redundant / near-empty features', keep)

# ---------------- 4 PCA: how much of the static information is duplicated?
static_cols = [c for c in keep if c in F.columns and c not in ('mid_lat', 'mid_lon')]
Zs = X_all[static_cols].copy()
for c in Zs: Zs[c] = np.log1p(Zs[c].clip(lower=0)) if Zs[c].min() >= 0 else Zs[c]
Zs = (Zs - Zs.mean()) / Zs.std().replace(0, 1)
pca = PCA().fit(Zs.values); ev = np.cumsum(pca.explained_variance_ratio_)
pca_tab = pd.DataFrame({'component': np.arange(1, len(ev) + 1), 'explained_variance': pca.explained_variance_ratio_.round(4), 'cumulative': ev.round(4)})
pca_tab.to_csv(os.path.join(OUT, 'pca_static_features_variance.csv'), index=False)
load = pd.DataFrame(pca.components_[:5].T, index=static_cols, columns=[f'PC{i}' for i in range(1, 6)]).round(3)
load.to_csv(os.path.join(OUT, 'pca_loadings_top5.csv'))
k90 = int(np.searchsorted(ev, 0.90) + 1)
log(f'PCA: {len(static_cols)} static features; 80% variance in {int(np.searchsorted(ev, .8) + 1)} comps, 90% in {k90}, 95% in {int(np.searchsorted(ev, .95) + 1)}')
hist_cols = [c for c in keep if c in H.columns]
PCs = pd.DataFrame(pca.transform(Zs.values)[:, :k90], columns=[f'pc{i + 1}' for i in range(k90)])
run('4 PCA', f'static features replaced by {k90} principal components (90% variance) + history + location', None,
    X=pd.concat([PCs, X_all[hist_cols + ['mid_lat', 'mid_lon']].reset_index(drop=True)], axis=1))
poi_cols = [c for c in keep if c.startswith('poi') or c in ('school_500m', 'ped_places_x_main')]
if len(poi_cols) >= 3:
    Zp = np.log1p(X_all[poi_cols]); Zp = (Zp - Zp.mean()) / Zp.std().replace(0, 1); pp = PCA(2).fit(Zp.values)
    Xp = pd.concat([X_all[[c for c in keep if c not in poi_cols]].reset_index(drop=True), pd.DataFrame(pp.transform(Zp.values), columns=['poi_pc1', 'poi_pc2'])], axis=1)
    run('4 PCA', f'pedestrian-place block ({len(poi_cols)} cols) replaced by 2 PCs ({pp.explained_variance_ratio_.sum():.0%} var)', None, X=Xp)

# ---------------- 5 group ablation: information carried by each source
groups = prof.set_index('feature').group.to_dict(); groups.update({'mid_lat': 'location', 'mid_lon': 'location'})
gl = sorted(set(groups[c] for c in keep))
abl = []
for g in gl:
    cols = [c for c in keep if groups[c] != g]
    r, _ = run('5 drop group', f'without {g}', cols); abl.append(dict(group=g, n_features=sum(groups[c] == g for c in keep), drop_aucc20=r['val_aucc20'],
                                                                     loss_when_dropped=round(r_red['val_aucc20'] - r['val_aucc20'], 4)))
for g in gl:
    cols = [c for c in keep if groups[c] == g] + (['length_m'] if g != 'geometry' else [])
    r, _ = run('5 only group', f'only {g} (+length)', cols); [a.update(only_aucc20=r['val_aucc20']) for a in abl if a['group'] == g]
abl = pd.DataFrame(abl).sort_values('loss_when_dropped', ascending=False); abl.to_csv(os.path.join(OUT, 'group_ablation.csv'), index=False)

# ---------------- 5b greedy group elimination: drop a whole source group while that improves validation
cur, cur_s = keep.copy(), r_red['val_aucc20']
while True:
    best_g, best_r = None, None
    for g in sorted(set(groups[c] for c in cur)):
        cols = [c for c in cur if groups[c] != g]
        if not cols: continue
        r, _ = run('5b group elim', f'from current set, without {g}', cols)
        if r['val_aucc20'] > cur_s + 0.001 and (best_r is None or r['val_aucc20'] > best_r['val_aucc20']): best_g, best_r = g, r
    if best_g is None: break
    dropped.update({c: f'group {best_g} removed: validation improved without it' for c in cur if groups[c] == best_g})
    cur = [c for c in cur if groups[c] != best_g]; cur_s = best_r['val_aucc20']; log(f'>>> removed group {best_g}; AUCC20 now {cur_s:.4f}')
keep = cur; r_red = dict(r_red, val_aucc20=cur_s)

# ---------------- 6 feature-level selection: permutation importance on the VALIDATION target (model fit on train-A)
m_full = HistGradientBoostingRegressor(**BASE_P).fit(X_all[keep], y)
vs = np.random.default_rng(1).choice(len(S), 60000, replace=False)
pi = permutation_importance(m_full, X_all[keep].iloc[vs], yv[vs] * y.sum() / yv.sum(), n_repeats=5, random_state=0, scoring='neg_mean_poisson_deviance')
imp = pd.DataFrame({'feature': keep, 'val_permutation_importance': pi.importances_mean, 'std': pi.importances_std}).sort_values('val_permutation_importance', ascending=False)
imp.to_csv(os.path.join(OUT, 'validation_permutation_importance.csv'), index=False)
cand = imp.feature.tolist()[::-1][:12]                                       # the 12 least useful, least useful first
sel = keep.copy(); best = r_red['val_aucc20']
for f_ in cand:                                                              # backward elimination: drop if validation does not get worse
    trial = [c for c in sel if c != f_]
    r, _ = run('6 backward', f'drop {f_}', trial)
    if r['val_aucc20'] >= best - 0.0005: sel, best = trial, max(best, r['val_aucc20']); dropped[f_] = 'no validation value (backward elimination)'
log(f'selected {len(sel)} features: {sel}')
r_sel, _ = run('6 selected', 'selected feature set', sel)

# ---------------- 7 target choice: KSI vs all crashes vs blend
run('7 target', 'train on ALL crashes, rank for KSI', sel, target=ownA['crashes'])
run('7 target', 'train on KSI + 0.5 x non-KSI crashes', sel, target=y + 0.5 * (ownA['crashes'] - y))

# ---------------- 8 hyper-parameters (small random search on the selected set)
rng = np.random.default_rng(7); best_p, best_s = BASE_P, r_sel['val_aucc20']; hp = []
for i in range(8):
    p = dict(BASE_P, learning_rate=float(rng.choice([0.03, 0.05, 0.08, 0.1])), max_leaf_nodes=int(rng.choice([15, 31, 63])),
             min_samples_leaf=int(rng.choice([50, 100, 200, 400])), l2_regularization=float(rng.choice([0.0, 1.0, 5.0])), max_iter=int(rng.choice([300, 500, 800])))
    r, _ = run('8 tuning', f"lr={p['learning_rate']} leaves={p['max_leaf_nodes']} minleaf={p['min_samples_leaf']} l2={p['l2_regularization']} iters={p['max_iter']}", sel, params=p)
    hp.append(dict(p, val_aucc20=r['val_aucc20'], val_deviance=r['val_deviance']))
    if r['val_aucc20'] > best_s + 0.0005: best_p, best_s = p, r['val_aucc20']
pd.DataFrame(hp).to_csv(os.path.join(OUT, 'hyperparameter_search.csv'), index=False)
r_final, mu_gbm = run('9 final A2', 'selected features + tuned parameters', sel, params=best_p)

# ---------------- 9 A1 SPF: low-collinearity interpretable subset of the selected features (VIF < 5)
def spf_design(cols):
    Zd = pd.DataFrame(index=range(len(S)))
    for c in cols:
        v = X_all[c].values
        Zd[c] = np.log1p(np.clip(v, 0, None)) if (v.min() >= 0 and v.max() > 5) else v
    return Zd
spf_pool = [c for c in sel if c not in ('mid_lat', 'mid_lon', 'length_m')]
Zd = spf_design(spf_pool)
from statsmodels.stats.outliers_influence import variance_inflation_factor
while True:
    A_ = sm.add_constant(Zd.iloc[smp]).values
    vif = pd.Series([variance_inflation_factor(A_, i + 1) for i in range(Zd.shape[1])], index=Zd.columns)
    if vif.max() < 5: break
    Zd = Zd.drop(columns=vif.idxmax())
spf_terms = list(Zd.columns); log(f'A1 SPF terms (VIF<5): {spf_terms}')
Xs = sm.add_constant(Zd); off = np.log(np.maximum(L / 1000, 0.01))
alpha = float(sm.NegativeBinomial(y, Xs, offset=off).fit(disp=0, maxiter=200).params['alpha'])
oof_spf = np.zeros(len(S))
for f_ in range(5):
    tr, te = fold != f_, fold == f_
    oof_spf[te] = sm.GLM(y[tr], Xs[tr], family=sm.families.NegativeBinomial(alpha=alpha), offset=off[tr]).fit().predict(Xs[te], offset=off[te])
m_spf = metrics(oof_spf, yv)
LOG.append(dict(step='9 final A1', experiment=f'NB SPF on {len(spf_terms)} low-VIF terms', n_features=len(spf_terms), val_aucc20=m_spf['aucc20'], val_cap1=m_spf['cap1'],
                val_cap5=m_spf['cap5'], val_cap10=m_spf['cap10'], val_deviance=pdev_levelfree(oof_spf, yv), eb_val_aucc20=metrics(eb(oof_spf, y), yv)['aucc20']))
blend = []
for w in np.round(np.arange(0, 1.01, 0.1), 1):
    mu = w * oof_spf + (1 - w) * mu_gbm; blend.append(dict(w_spf=w, **metrics(mu, yv), eb_aucc20=metrics(eb(mu, y), yv)['aucc20']))
blend = pd.DataFrame(blend); blend.to_csv(os.path.join(OUT, 'blend_weight_validation.csv'), index=False)
w_best = float(blend.loc[blend.eb_aucc20.idxmax(), 'w_spf'])
mu_b = w_best * oof_spf + (1 - w_best) * mu_gbm; mb, meb = metrics(mu_b, yv), metrics(eb(mu_b, y), yv)
LOG.append(dict(step='10 final', experiment=f'blend {w_best}*A1 + {1 - w_best:.1f}*A2, then Empirical Bayes', n_features=len(sel), val_aucc20=mb['aucc20'], val_cap1=mb['cap1'],
                val_cap5=mb['cap5'], val_cap10=mb['cap10'], val_deviance=pdev_levelfree(mu_b, yv), eb_val_aucc20=meb['aucc20'], eb_val_cap5=meb['cap5']))
log(f'final blend w_spf={w_best}: model AUCC20={mb["aucc20"]} | EB AUCC20={meb["aucc20"]} cap5={meb["cap5"]}')

# ---------------- save records
pd.DataFrame(LOG).to_csv(os.path.join(OUT, 'experiment_log.csv'), index=False)
prof['kept_in_final_A2'] = prof.feature.isin(sel); prof['used_in_A1_spf'] = prof.feature.isin(spf_terms)
prof['decision'] = np.where(prof.kept_in_final_A2, 'KEEP', prof.feature.map(dropped).fillna('dropped'))
prof = prof.merge(imp[['feature', 'val_permutation_importance']], on='feature', how='left')
prof.sort_values(['kept_in_final_A2', 'val_permutation_importance'], ascending=[False, False]).to_csv(os.path.join(OUT, 'feature_catalog.csv'), index=False)
json.dump(dict(a2_features=sel, a1_spf_terms=spf_terms, a2_params=best_p, blend_weight_spf=w_best, target='ksi', history_radii_km=[0.5, 1, 3], recent_months=12,
               protocol=dict(select_train=[WA[0], WA[-1]], select_validation=[WV[0], WV[-1]], metric='capture-curve area 0-20% road length'),
               runtime_s=round(time.time() - t0)), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_features.json'), 'w'), indent=1)
json.dump(dict(a2_features=sel, a1_spf_terms=spf_terms, a2_params=best_p, blend_weight_spf=w_best), open(os.path.join(OUT, 'final_features.json'), 'w'), indent=1)
log('feature study done')
