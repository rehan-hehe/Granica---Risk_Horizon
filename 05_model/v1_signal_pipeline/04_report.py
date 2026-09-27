"""Step 4 — figures, one results workbook, and the model card, from the Stage A/B/C CSV outputs.
Output: 2_readable/11_model_results/figures/*.png, Risk_Horizon_Model_Results.xlsx
"""
import os, sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
R = os.path.join(ROOT, '2_readable', '11_model_results'); FIG = os.path.join(R, 'figures'); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'figure.dpi': 150, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
rd = lambda *p: pd.read_csv(os.path.join(R, *p), low_memory=False)
ev, cal, rr = rd('A_static_risk', 'A_prospective_test.csv'), rd('A_static_risk', 'A_calibration_by_decile.csv'), rd('A_static_risk', 'A1_spf_rate_ratios.csv')
imp, b1, b2 = rd('A_static_risk', 'A2_tree_feature_importance.csv'), rd('B_conditions', 'B1_crash_likelihood_odds_ratios.csv'), rd('B_conditions', 'B2_severity_odds_ratios.csv')
mult, val, load = rd('B_conditions', 'B2_darkness_multipliers.csv'), rd('C_signal', 'C_band_validation_2025_26.csv'), rd('C_signal', 'C_alert_load_by_condition.csv')
bdef, sig = rd('C_signal', 'C_band_definitions.csv'), rd('C_signal', 'C_segment_signal_table.csv')
BLUE, GREY, RED = '#1f5fa8', '#9a9a9a', '#c8312b'
BANDC = {0: '#dcdcdc', 1: '#f4c542', 2: '#e8702a', 3: '#b0171f'}

# 1 architecture
fig, ax = plt.subplots(figsize=(11, 4.2)); ax.axis('off'); ax.set_xlim(0, 11); ax.set_ylim(0, 4.2)
def box(x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05', fc=c, ec='#333', lw=0.8)); ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=8)
def arrow(x1, y1, x2, y2): ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='->', lw=1))
box(0.1, 3.0, 2.2, 0.9, 'Road segments (OSM)\n+ night light (VIIRS)\n+ iRAD crash counts', '#eef3fb')
box(0.1, 0.4, 2.2, 1.0, 'Crash hours vs control hours\n(case-crossover)\n+ ERA5 weather + sun', '#eef3fb')
box(2.8, 3.35, 2.1, 0.6, 'A1 NB safety performance\nfunction (30%)', '#dde8f7'); box(2.8, 2.55, 2.1, 0.6, 'A2 Poisson boosted\ntrees (70%)', '#dde8f7')
box(5.3, 2.9, 1.7, 0.8, 'Empirical Bayes\n+ own crash history', '#c9dbf2')
box(2.8, 1.05, 2.1, 0.6, 'B1 conditional logit\n(rain, fog: likelihood)', '#fde9d9'); box(2.8, 0.2, 2.1, 0.6, 'B2 logistic severity\n(dark, lighting, road)', '#fde9d9')
box(5.3, 0.55, 1.7, 0.8, 'Condition\nmultipliers\n(never < 1)', '#f9d3b8')
box(7.5, 1.7, 1.5, 1.0, 'risk now =\nstatic x\nmultipliers', '#e2f0d9'); box(9.3, 1.7, 1.6, 1.0, 'Bands 0-3 per\n500 m segment\n+ reasons, ped flag', '#c5e0b4')
box(7.5, 3.25, 3.4, 0.6, 'NOW: live TomTom speed decides when to warn', '#f2f2f2')
for a in [(2.3, 3.45, 2.8, 3.6), (2.3, 3.45, 2.8, 2.85), (4.9, 3.6, 5.3, 3.3), (4.9, 2.85, 5.3, 3.1), (2.3, 0.9, 2.8, 1.35), (2.3, 0.9, 2.8, 0.5),
          (4.9, 1.35, 5.3, 1.0), (4.9, 0.5, 5.3, 0.9), (7.0, 3.3, 7.5, 2.4), (7.0, 0.95, 7.5, 2.0), (9.0, 2.2, 9.3, 2.2), (10.1, 3.25, 10.1, 2.7)]:
    arrow(*a)
ax.text(1.2, 4.05, 'DATA', ha='center', weight='bold'); ax.text(4.9, 4.05, 'MODELS (fit on 2023-24)', ha='center', weight='bold'); ax.text(9.2, 4.05, 'SIGNAL TO THE CAR', ha='center', weight='bold')
fig.tight_layout(); fig.savefig(os.path.join(FIG, '01_model_architecture.png')); plt.close(fig)

# 2 prospective test
lab = ['Past crashes\n(blackspot list)', 'A1 SPF', 'A2 trees', 'Ensemble', 'Final\n(EB)']
col = [GREY, '#8fb3de', '#8fb3de', '#4f86c6', BLUE]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for a, c, t, fmt in [(ax[0], 'test_KSI_in_top5pct_km', 'Future serious crashes in\ntop 5% of road length', '%.1f%%'),
                     (ax[1], 'new_blackspots_in_top5pct_km', f'New blackspots flagged (of {ev.new_blackspots_total[0]})\nin top 5% of road length', '%d'),
                     (ax[2], 'test_KSI_in_top1pct_km', 'Future serious crashes in\ntop 1% of road length', '%.1f%%')]:
    v = ev[c] * (100 if 'KSI' in c else 1); b = a.bar(lab, v, color=col); a.bar_label(b, fmt=fmt, fontsize=8); a.set_title(t, fontsize=9); a.tick_params(axis='x', labelsize=7)
fig.suptitle('Prospective test: trained on 2023-24 only, scored on Jan 2025 - May 2026', fontsize=10); fig.tight_layout()
fig.savefig(os.path.join(FIG, '02_prospective_test.png')); plt.close(fig)

# 3 band validation
fig, ax = plt.subplots(figsize=(7, 3.4)); x = np.arange(len(val))
ax.bar(x - 0.2, val.share_road_km * 100, 0.4, color=GREY, label='% of road length'); ax.bar(x + 0.2, val.share_test_KSI * 100, 0.4, color=RED, label='% of 2025-26 serious crashes')
for i, r in val.iterrows(): ax.text(i + 0.2, r.share_test_KSI * 100 + 1, f'{r.KSI_per_100km:.0f}/100 km', ha='center', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(val.band); ax.legend(frameon=False); ax.set_title('Risk bands (test model): 5% of road holds half of future serious crashes')
fig.tight_layout(); fig.savefig(os.path.join(FIG, '03_band_validation.png')); plt.close(fig)

# 4 SPF rate ratios
s = rr[~rr.term.isin(['const', 'alpha'])].iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 4.4)); yy = np.arange(len(s))
ax.errorbar(s.rate_ratio, yy, xerr=[s.rate_ratio - s.rr_ci95_low, s.rr_ci95_high - s.rate_ratio], fmt='o', color=BLUE, capsize=2, ms=4)
ax.axvline(1, color='k', ls='--', lw=0.7); ax.set_xscale('log'); ax.set_yticks(yy); ax.set_yticklabels(s.term, fontsize=8)
ax.set_xlabel('Rate ratio for serious crashes (log scale; per unit / vs minor roads)'); ax.set_title('A1 safety performance function: what raises crash rates')
fig.tight_layout(); fig.savefig(os.path.join(FIG, '04_spf_rate_ratios.png')); plt.close(fig)

# 5 severity odds ratios + darkness multipliers
s = b2[b2.term != 'Intercept'].copy()
s['label'] = s.term.str.replace(r"C\((\w+), Treatment\('[\w_]+'\)\)\[T\.", r'\1: ', regex=True).str.replace(r'C\(season\)\[T\.', 'season: ', regex=True).str.rstrip(']')
s = s.iloc[::-1]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={'width_ratios': [1.4, 1]}); yy = np.arange(len(s))
ax[0].errorbar(s.odds_ratio_2023_24, yy, xerr=[s.odds_ratio_2023_24 - s.ci95_low, s.ci95_high - s.odds_ratio_2023_24], fmt='o', color=RED, capsize=2, ms=4)
ax[0].axvline(1, color='k', ls='--', lw=0.7); ax[0].set_yticks(yy); ax[0].set_yticklabels(s.label, fontsize=8)
ax[0].set_title('B2: odds of a crash being fatal (2023-24 fit)'); ax[0].set_xlabel('odds ratio (95% CI)')
x = np.arange(len(mult))
ax[1].bar(x - 0.2, mult.p_fatal_daylight * 100, 0.4, color='#f2c14e', label='daylight'); ax[1].bar(x + 0.2, mult.p_fatal_dark * 100, 0.4, color='#2b2d42', label='dark')
for i, r in mult.iterrows(): ax[1].text(i, r.p_fatal_dark * 100 + 1, f'x{r.multiplier_dark:.2f}', ha='center', fontsize=8)
ax[1].set_xticks(x); ax[1].set_xticklabels(mult.lighting_class.str.replace('_', ' '), fontsize=8); ax[1].set_ylabel('% of crashes fatal (model)')
ax[1].legend(frameon=False); ax[1].set_title('Darkness multiplier by street-lighting class')
fig.tight_layout(); fig.savefig(os.path.join(FIG, '05_severity_and_darkness.png')); plt.close(fig)

# 6 maps: Assam static bands; Guwahati corridor day vs dark
def draw(ax, d, colname, title, size):
    for b in (0, 1, 2, 3):
        q = d[d[colname] == b]; ax.scatter(q.mid_lon, q.mid_lat, s=size[b], c=BANDC[b], lw=0, rasterized=True, label=f'band {b}')
    ax.set_title(title, fontsize=9); ax.set_aspect(1.1); ax.set_xticks([]); ax.set_yticks([])
fig, ax = plt.subplots(figsize=(9, 5.4)); draw(ax, sig, 'static_band', 'Static risk band, every 500 m of road in Assam', {0: 0.05, 1: 0.2, 2: 0.6, 3: 1.2})
ax.legend(markerscale=6, frameon=False, loc='lower right'); fig.tight_layout(); fig.savefig(os.path.join(FIG, '06_assam_static_bands.png')); plt.close(fig)
g = sig[sig.mid_lat.between(26.05, 26.36) & sig.mid_lon.between(91.55, 91.95)]
fig, ax = plt.subplots(1, 2, figsize=(12, 5)); sz = {0: 0.4, 1: 1.5, 2: 4, 3: 7}
draw(ax[0], g, 'static_band', 'Guwahati corridor: daytime bands', sz); draw(ax[1], g, 'dark_band', 'Same roads after dark (darkness multiplier)', sz)
ax[1].legend(markerscale=3, frameon=False, loc='lower right'); fig.tight_layout(); fig.savefig(os.path.join(FIG, '07_guwahati_day_vs_dark.png')); plt.close(fig)

# 7 calibration + tree importance
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(cal.decile, cal.predicted_test_KSI, 'o-', color=BLUE, label='predicted (scaled to 17 months)'); ax[0].plot(cal.decile, cal.observed_test_KSI, 's-', color=RED, label='observed 2025-26')
ax[0].set_yscale('log'); ax[0].set_xlabel('risk decile (10 = riskiest)'); ax[0].set_title('Ranking holds; levels run ~40% low\n(reported KSI per month rose 46% in 2025-26)'); ax[0].legend(frameon=False, fontsize=8)
t = imp.head(12).iloc[::-1]; ax[1].barh(t.feature, t.importance, color=BLUE); ax[1].set_title('A2 trees: top features (permutation)'); ax[1].tick_params(labelsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, '08_calibration_and_importance.png')); plt.close(fig)

# workbook
A = json.load(open(os.path.join(R, 'A_static_risk', 'A_model_settings.json'))); B = json.load(open(os.path.join(R, 'B_conditions', 'B_model_settings.json')))
fin = ev.iloc[-1]; base = ev.iloc[0]
readme = pd.DataFrame({'Risk Horizon model results': [
    'Pipeline: A static risk (NB safety performance function + Poisson boosted trees, blended, then Empirical Bayes) x B condition multipliers (conditional logit for likelihood, logistic for severity) -> C bands 0-3 per 500 m segment.',
    f"Headline (fit on 2023-24, tested on Jan 2025-May 2026): top 5% of road length holds {fin.test_KSI_in_top5pct_km:.1%} of future serious crashes vs {base.test_KSI_in_top5pct_km:.1%} for the past-crash list; new blackspots flagged {fin.new_blackspots_in_top5pct_km} vs {base.new_blackspots_in_top5pct_km} of {fin.new_blackspots_total}.",
    'Darkness raises the odds that a crash is fatal (see B2); rain and fog do not raise crash likelihood (see B1), so their signal multiplier is 1.0.',
    f"Severity model separates fatal from non-fatal crashes only weakly (AUC {B['severity_auc_test']} on 2025-26): it is used for average multipliers, not to predict single crashes.",
    f"Blend weight on the SPF chosen on 2023-24 out-of-fold deviance: {A['ensemble_weight_on_spf']}. Runtime: Stage A {A['runtime_s']} s, Stage B {B['runtime_s']} s.",
    'Shareable: everything in this workbook is aggregated. The segment signal table has no crash-level data.']})
with pd.ExcelWriter(os.path.join(R, 'Risk_Horizon_Model_Results.xlsx')) as w:
    for n, df in [('README', readme), ('A_prospective_test', ev), ('A_calibration', cal), ('A1_SPF_rate_ratios', rr), ('A2_tree_importance', imp),
                  ('B1_likelihood_ORs', b1), ('B2_severity_ORs', b2), ('B2_dark_multipliers', mult), ('C_band_definitions', bdef),
                  ('C_band_validation', val), ('C_alert_load', load), ('C_top500_segments', sig.sort_values('static_risk_ksi_per_km_yr', ascending=False).head(500))]:
        df.to_excel(w, sheet_name=n, index=False)
from openpyxl import load_workbook
wb = load_workbook(os.path.join(R, 'Risk_Horizon_Model_Results.xlsx'))
for ws in wb.worksheets:
    ws.freeze_panes = 'A2'
    for c in ws.columns: ws.column_dimensions[c[0].column_letter].width = min(max(10, max(len(str(x.value or '')) for x in c[:200]) + 2), 90)
wb.save(os.path.join(R, 'Risk_Horizon_Model_Results.xlsx'))
print('report done')
