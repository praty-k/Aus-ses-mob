# -*- coding: utf-8 -*-
"""
Created on Fri Dec 8 2023

@author: pkollepara
"""

#%% Importing Packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns
#import pymannkendall as mk
import os

#%% Importing functions
from dataframe_prep_w_cov_all_SA1s import dataframe_prep
from draw_lplot import draw_lplot
from draw_jplot import draw_scatter, draw_hex
from ER_EO_heatmap import compute_heatmap_pop, compute_heatmap, draw_heatmap_pop, draw_heatmap, draw_heatmap_zero_centre
from mono_test import mono_test_mk, mono_test_spearman
from corr import spearman_corr, linear_regression

#%% Creating results directory if it does not exist
dname = 'Refactored_code_plots_cov_all_SA1s'
if os.path.exists(dname) == False:
     os.mkdir(dname)

#%% Plotting styles
plt.style.use('seaborn-v0_8-colorblind')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 12
plt.rc('axes', titlesize=plt.rcParams['font.size'])
plt.rc('axes', axisbelow=True)
#%% Plotting symbols
lm = '$\lambda_{\mathrm{mob}}$'
lc = '$\lambda_{\mathrm{cov}}$'
#rm = '$\Delta v/v$'
#%% Read in tables containing counts from each time period, SEIFA scores, SA1 code and URP
df_mel = pd.read_csv('MEL__seifa_all_filter.csv', na_values='-',
                     usecols=['SA1', 'ER_score', 'EO_score', 'URP', 'apr20base_count', 'apr20wave_count', 'jan22base_count', 'jan22wave_count'])
print(df_mel.shape, df_mel.sample(8))
print(df_mel.describe())

df_syd = pd.read_csv('SYD__seifa_all_filter.csv', na_values='-',
                     usecols=['SA1', 'ER_score', 'EO_score', 'URP', 'apr20base_count', 'apr20wave_count', 'jan22base_count', 'jan22wave_count'])

#%% HEx plot bw ER and EO score
fig, ax = plt.subplots(1, 2, figsize = (12*0.4*2.1, 12*0.4), sharex = True, sharey = True)
cmap = mcolors.ListedColormap(sns.color_palette("Blues",256))
cmap.set_under('lightgrey') 
df_mel.plot(kind='hexbin', x = 'EO_score', y = 'ER_score', gridsize = 35, ax = ax[0], cmap = cmap, vmin = 1)
df_syd.plot(kind='hexbin', x = 'EO_score', y = 'ER_score', gridsize = 35, ax = ax[1], cmap = cmap, vmin = 1)
ax[0].set_facecolor('lightgrey') 
ax[1].set_facecolor('lightgrey') 
ax[0].set_xlabel('EO Score')
ax[1].set_xlabel('EO Score')
ax[0].set_ylabel('ER Score')
ax[1].set_ylabel('ER Score')
ax[0].set_title('Melbourne')
ax[1].set_title('Sydney')
plt.tight_layout()
fig.savefig('ER-EO-Histogram-a.pdf')

#%% Coverage: Read in daily coverage values for SA1s, join with df_mel and df_syd, 
### compute v = no of daily trips / daily coverage  for all time periods
### log_ratio_mob = v_test/v_base
#MELBOURNE
#REading in coverage values
cov_mel_jan22base = pd.read_csv('SA1_coverage/Daily_coverage_GMEL_jan22base.csv', index_col=0)
cov_mel_jan22test = pd.read_csv('SA1_coverage/Daily_coverage_GMEL_jan22test.csv', index_col=0)
cov_mel_apr20base = pd.read_csv('SA1_coverage/Daily_coverage_GMEL_apr20base.csv', index_col=0)
cov_mel_apr20test = pd.read_csv('SA1_coverage/Daily_coverage_GMEL_apr20test.csv', index_col=0)

#joining mobility table of melbourne to each coverage table
df_mel = pd.merge(df_mel, cov_mel_jan22base, how = 'inner', left_on = 'SA1', right_on = 'SA1_code').drop('SA1_code', axis = 1)
df_mel = pd.merge(df_mel, cov_mel_jan22test, how = 'inner', left_on = 'SA1', right_on = 'SA1_code').drop('SA1_code', axis = 1)
df_mel = pd.merge(df_mel, cov_mel_apr20base, how = 'inner', left_on = 'SA1', right_on = 'SA1_code').drop('SA1_code', axis = 1)
df_mel = pd.merge(df_mel, cov_mel_apr20test, how = 'inner', left_on = 'SA1', right_on = 'SA1_code').drop('SA1_code', axis = 1)

#Calculating adjusted mobility count (v) i.e. dividing the measured count by the daily coverage
df_mel['jan22base_v'] = df_mel.jan22base_count/df_mel.jan22_base_daily_coverage
df_mel['jan22test_v'] = df_mel.jan22wave_count/df_mel.jan22_test_daily_coverage
df_mel['apr20base_v'] = df_mel.apr20base_count/df_mel.apr20_base_daily_coverage
df_mel['apr20test_v'] = df_mel.apr20wave_count/df_mel.apr20_test_daily_coverage

#Log transform of (v)
df_mel['apr20_log_ratio_mob'] = np.log(df_mel.apr20test_v/df_mel.apr20base_v)
df_mel['jan22_log_ratio_mob'] = np.log(df_mel.jan22test_v/df_mel.jan22base_v)

#Relative change in v
df_mel['apr20_rel_mob'] = (df_mel.apr20test_v - df_mel.apr20base_v)/df_mel.apr20base_v
df_mel['jan22_rel_mob'] = (df_mel.jan22test_v - df_mel.jan22base_v)/df_mel.jan22base_v

# SYDNEY 
cov_syd_jan22base = pd.read_csv('SA1_coverage/Daily_coverage_GSYD_jan22base.csv', index_col=0)
cov_syd_jan22test = pd.read_csv('SA1_coverage/Daily_coverage_GSYD_jan22test.csv', index_col=0)
cov_syd_apr20base = pd.read_csv('SA1_coverage/Daily_coverage_GSYD_apr20base.csv', index_col=0)
cov_syd_apr20test = pd.read_csv('SA1_coverage/Daily_coverage_GSYD_apr20test.csv', index_col=0)

df_syd = pd.merge(df_syd, cov_syd_jan22base, how = 'inner', left_on = 'SA1', right_on = 'SA1_code').drop('SA1_code', axis = 1)
df_syd = pd.merge(df_syd, cov_syd_jan22test, how = 'inner', left_on = 'SA1', right_on = 'SA1_code').drop('SA1_code', axis = 1)
df_syd = pd.merge(df_syd, cov_syd_apr20base, how = 'inner', left_on = 'SA1', right_on = 'SA1_code').drop('SA1_code', axis = 1)
df_syd = pd.merge(df_syd, cov_syd_apr20test, how = 'inner', left_on = 'SA1', right_on = 'SA1_code').drop('SA1_code', axis = 1)

df_syd['jan22base_v'] = df_syd.jan22base_count/df_syd.jan22_base_daily_coverage
df_syd['jan22test_v'] = df_syd.jan22wave_count/df_syd.jan22_test_daily_coverage
df_syd['apr20base_v'] = df_syd.apr20base_count/df_syd.apr20_base_daily_coverage
df_syd['apr20test_v'] = df_syd.apr20wave_count/df_syd.apr20_test_daily_coverage

df_syd['apr20_log_ratio_mob'] = np.log(df_syd.apr20test_v/df_syd.apr20base_v)
df_syd['jan22_log_ratio_mob'] = np.log(df_syd.jan22test_v/df_syd.jan22base_v)

df_syd['apr20_rel_mob'] = (df_syd.apr20test_v - df_syd.apr20base_v)/df_syd.apr20base_v
df_syd['jan22_rel_mob'] = (df_syd.jan22test_v - df_syd.jan22base_v)/df_syd.jan22base_v


#%% Prepare dataframes: remove NaNs and compute local deciles of EO and ER
df_mel = dataframe_prep(df_mel)
df_syd = dataframe_prep(df_syd)

#%% Exporting SA1s and their EO and ER decile for plotting maps
df_mel[['SA1', 'ER_local_decile', 'EO_local_decile']].to_csv('SA1_SEIFA_deciles_MEL.csv')
df_syd[['SA1', 'ER_local_decile', 'EO_local_decile']].to_csv('SA1_SEIFA_deciles_SYD.csv')


#%% Heatmaps Melbourne
fname = 'Mel-'
hdata = compute_heatmap(df_mel.ER_local_decile.values, df_mel.EO_local_decile.values, df_mel.apr20_log_ratio_mob.values, 10)
f = draw_heatmap_zero_centre(hdata, fname, 'RdBu_r', title = 'Melbourne - April 2020', cbar_title=f'Median {lm}')
f.savefig('Refactored_code_plots_cov_all_SA1s/Mel-Apr20-hmap.pdf')
hdata = compute_heatmap(df_mel.ER_local_decile.values, df_mel.EO_local_decile.values, df_mel.jan22_log_ratio_mob.values, 10)
f = draw_heatmap_zero_centre(hdata, fname, 'RdBu_r', title = 'Melbourne - Jan 2022', cbar_title=f'Median {lm}')
f.savefig('Refactored_code_plots_cov_all_SA1s/Mel-Jan2022-hmap.pdf')
SA1_hist_mel = compute_heatmap(df_mel.ER_local_decile.values, df_mel.EO_local_decile.values, qty = 'histogram')
f = draw_heatmap(SA1_hist_mel, fname = fname, title = 'Counts of SA1s in Melbourne', fmt = '0.0f')
f.savefig('Refactored_code_plots_cov_all_SA1s/Mel-SA1s-hmap.pdf')
SA1_hist_mel_pop = compute_heatmap_pop(df_mel.ER_local_decile.values, df_mel.EO_local_decile.values, df_mel.URP)
f = draw_heatmap_pop(SA1_hist_mel_pop, fname = fname, title = 'Population counts in Melbourne', fmt = '1.1e')
f.savefig('Refactored_code_plots_cov_all_SA1s/Mel-pop-hmap.pdf')

#%% Heatmaps Sydney
fname = 'Syd-'
hdata = compute_heatmap(df_syd.ER_local_decile.values, df_syd.EO_local_decile.values, df_syd.apr20_log_ratio_mob.values, 10)
f = draw_heatmap_zero_centre(hdata, fname, 'RdBu_r', title = 'Sydney - April 2020', cbar_title=f'Median {lm}')
f.savefig('Refactored_code_plots_cov_all_SA1s/Syd-Apr20-hmap.pdf')
hdata = compute_heatmap(df_syd.ER_local_decile.values, df_syd.EO_local_decile.values, df_syd.jan22_log_ratio_mob.values, 10)
f = draw_heatmap_zero_centre(hdata, fname, 'RdBu_r', title = 'Sydney - Jan 2022', cbar_title=f'Median {lm}')
f.savefig('Refactored_code_plots_cov_all_SA1s/Syd-Jan22-hmap.pdf')
SA1_hist_syd = compute_heatmap(df_syd.ER_local_decile.values, df_syd.EO_local_decile.values, qty = 'histogram')
f = draw_heatmap(SA1_hist_syd, fname = fname, title = 'Counts of SA1s in Sydney', fmt = '0.0f')
f.savefig('Refactored_code_plots_cov_all_SA1s/Syd-SA1s-hmap.pdf')
SA1_hist_syd_pop = compute_heatmap_pop(df_syd.ER_local_decile.values, df_syd.EO_local_decile.values, df_syd.URP)
f = draw_heatmap_pop(SA1_hist_syd_pop, fname = fname, title = 'Population counts in Sydney', fmt = '1.1e')
f.savefig('Refactored_code_plots_cov_all_SA1s/Syd-pop-hmap.pdf')

#%% Line plots
fig, axs = plt.subplots(2, 2, figsize = (4*2, 4*2), sharey=True, sharex=True, constrained_layout = True)
Y = 'apr20_log_ratio_mob'
draw_lplot(axs[0, 0], df_mel.EO_local_decile.values, df_mel[Y], xlabel = 'Local decile of EO', ylabel = f'{lm}', title = 'Melbourne - April 2020', subplotlabel = 'A')
draw_lplot(axs[0, 1], df_syd.EO_local_decile.values, df_syd[Y], xlabel = 'Local decile of EO', ylabel = f'{lm}', title = 'Sydney - April 2020', subplotlabel = 'B')
draw_lplot(axs[1, 0], df_mel.ER_local_decile.values, df_mel[Y], xlabel = 'Local decile of ER', ylabel = f'{lm}', title = 'Melbourne - April 2020', subplotlabel = 'C')
draw_lplot(axs[1, 1], df_syd.ER_local_decile.values, df_syd[Y], xlabel = 'Local decile of ER', ylabel = f'{lm}', title = 'Sydney - April 2020', subplotlabel = 'D')
handles, legend_labels = axs[1, 1].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc = 'outside upper right', ncols = 3, fancybox = False)
fig.savefig('Refactored_code_plots_cov_all_SA1s/Apr20-lplot.pdf')

fig, axs = plt.subplots(2, 2, figsize = (4*2, 4*2), sharey=True, sharex=True, constrained_layout = True)
Y = 'jan22_log_ratio_mob'
draw_lplot(axs[0, 0], df_mel.EO_local_decile.values, df_mel[Y], xlabel = 'Local decile of EO', ylabel = f'{lm}', title = 'Melbourne - Jan 2022', subplotlabel = 'A')
draw_lplot(axs[0, 1], df_syd.EO_local_decile.values, df_syd[Y], xlabel = 'Local decile of EO', ylabel = f'{lm}', title = 'Sydney - Jan 2022', subplotlabel = 'B')
draw_lplot(axs[1, 0], df_mel.ER_local_decile.values, df_mel[Y], xlabel = 'Local decile of ER', ylabel = f'{lm}', title = 'Melbourne - Jan 2022', subplotlabel = 'C')
draw_lplot(axs[1, 1], df_syd.ER_local_decile.values, df_syd[Y], xlabel = 'Local decile of ER', ylabel = f'{lm}', title = 'Sydney - Jan 2022', subplotlabel = 'D')
handles, legend_labels = axs[1, 1].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc = 'outside upper right', ncols = 3, fancybox = False)
fig.savefig('Refactored_code_plots_cov_all_SA1s/Jan22-lplot.pdf')


#%% Boxen plots
from draw_univariate_plots import draw_vplot, draw_boxenplot


# fig, axs = plt.subplots(2, 2, figsize = (4*2, 4*2), sharey=True, sharex=True, constrained_layout = True)
# Y = 'apr20_log_ratio_mob'
# draw_vplot(axs[0, 0], x = 'EO_local_decile', y=Y, df=df_mel, xlabel = 'Local decile of EO', ylabel = f'{lm}', title = 'Melbourne - April 2020', subplotlabel = 'A')
# draw_vplot(axs[0, 1], x = 'EO_local_decile', y=Y, df = df_syd, xlabel = 'Local decile of EO', ylabel = f'{lm}', title = 'Sydney - April 2020', subplotlabel = 'B')
# draw_vplot(axs[1, 0], x = 'ER_local_decile', y=Y, df = df_mel, xlabel = 'Local decile of ER', ylabel = f'{lm}', title = 'Melbourne - April 2020', subplotlabel = 'C')
# draw_vplot(axs[1, 1], x = 'ER_local_decile', y=Y, df = df_syd, xlabel = 'Local decile of ER', ylabel = f'{lm}', title = 'Sydney - April 2020', subplotlabel = 'D')
# axs[0, 0].set_ylim(-4.5, 3)


fig, axs = plt.subplots(2, 2, figsize = (4*2, 4*2), sharey=True, constrained_layout = True)
Y = 'apr20_log_ratio_mob'
draw_boxenplot(axs[0, 0], x = 'EO_local_decile', y=Y, df=df_mel, xlabel = 'Local decile of EO', ylabel = f'{lm}', title = 'Melbourne - April 2020', subplotlabel = 'A')
draw_boxenplot(axs[0, 1], x = 'EO_local_decile', y=Y, df = df_syd, xlabel = 'Local decile of EO', ylabel = f'{lm}', title = 'Sydney - April 2020', subplotlabel = 'B')
draw_boxenplot(axs[1, 0], x = 'ER_local_decile', y=Y, df = df_mel, xlabel = 'Local decile of ER', ylabel = f'{lm}', title = 'Melbourne - April 2020', subplotlabel = 'C')
draw_boxenplot(axs[1, 1], x = 'ER_local_decile', y=Y, df = df_syd, xlabel = 'Local decile of ER', ylabel = f'{lm}', title = 'Sydney - April 2020', subplotlabel = 'D')
axs[0, 0].set_ylim(-4.5, 3)
#fig.savefig('Refactored_code_plots_cov_all_SA1s/Apr20-boxenplot_w_l.pdf')
#plt.close()

fig, axs = plt.subplots(2, 2, figsize = (4*2, 4*2), sharey=True, constrained_layout = True)
Y = 'jan22_log_ratio_mob'
draw_boxenplot(axs[0, 0], x = 'EO_local_decile', y=Y, df=df_mel, xlabel = 'Local decile of EO', ylabel = f'{lm}', title = 'Melbourne - Jan 2022', subplotlabel = 'A')
draw_boxenplot(axs[0, 1], x = 'EO_local_decile', y=Y, df = df_syd, xlabel = 'Local decile of EO', ylabel = f'{lm}', title = 'Sydney - Jan 2022', subplotlabel = 'B')
draw_boxenplot(axs[1, 0], x = 'ER_local_decile', y=Y, df = df_mel, xlabel = 'Local decile of ER', ylabel = f'{lm}', title = 'Melbourne - Jan 2022', subplotlabel = 'C')
draw_boxenplot(axs[1, 1], x = 'ER_local_decile', y=Y, df = df_syd, xlabel = 'Local decile of ER', ylabel = f'{lm}', title = 'Sydney - Jan 2022', subplotlabel = 'D')
handles, legend_labels = axs[1, 1].get_legend_handles_labels()

axs[0, 0].set_ylim(-1.75, 1.75)
fig.savefig('Refactored_code_plots_cov_all_SA1s/Jan22-boxenplot_w_l.pdf')
plt.close()


fig, axs = plt.subplots(2, 2, figsize = (4*2, 4*2), sharey=True, constrained_layout = True)
Y = 'apr20_log_ratio_mob'
draw_vplot(axs[0, 0], x = 'EO_local_decile', y=Y, df=df_mel, xlabel = 'Local decile of EO', ylabel = f'{lm}', title = 'Melbourne - April 2020', subplotlabel = 'A')
draw_vplot(axs[0, 1], x = 'EO_local_decile', y=Y, df = df_syd, xlabel = 'Local decile of EO', ylabel = f'{lm}', title = 'Sydney - April 2020', subplotlabel = 'B')
draw_vplot(axs[1, 0], x = 'ER_local_decile', y=Y, df = df_mel, xlabel = 'Local decile of ER', ylabel = f'{lm}', title = 'Melbourne - April 2020', subplotlabel = 'C')
draw_vplot(axs[1, 1], x = 'ER_local_decile', y=Y, df = df_syd, xlabel = 'Local decile of ER', ylabel = f'{lm}', title = 'Sydney - April 2020', subplotlabel = 'D')
axs[0, 0].set_ylim(-4.5, 3)
fig.savefig('Refactored_code_plots_cov_all_SA1s/Apr20-violinplot.pdf')
plt.close()

fig, axs = plt.subplots(2, 2, figsize = (4*2, 4*2), sharey=True, constrained_layout = True)
Y = 'jan22_log_ratio_mob'
draw_vplot(axs[0, 0], x = 'EO_local_decile', y=Y, df=df_mel, xlabel = 'Local decile of EO', ylabel = f'{lm}', title = 'Melbourne - Jan 2022', subplotlabel = 'A')
draw_vplot(axs[0, 1], x = 'EO_local_decile', y=Y, df = df_syd, xlabel = 'Local decile of EO', ylabel = f'{lm}', title = 'Sydney - Jan 2022', subplotlabel = 'B')
draw_vplot(axs[1, 0], x = 'ER_local_decile', y=Y, df = df_mel, xlabel = 'Local decile of ER', ylabel = f'{lm}', title = 'Melbourne - Jan 2022', subplotlabel = 'C')
draw_vplot(axs[1, 1], x = 'ER_local_decile', y=Y, df = df_syd, xlabel = 'Local decile of ER', ylabel = f'{lm}', title = 'Sydney - Jan 2022', subplotlabel = 'D')
handles, legend_labels = axs[1, 1].get_legend_handles_labels()

axs[0, 0].set_ylim(-1.75, 1.75)
fig.savefig('Refactored_code_plots_cov_all_SA1s/Jan22-violinplot.pdf')
plt.close()
# fig.legend(handles, legend_labels, loc = 'outside upper right', ncols = 3, fancybox = False)



#%%

#%% Monotonicity using MK test
res_mk_dict = {**mono_test_mk(df_mel, x = 'EO_local_decile', y = 'apr20_log_ratio_mob', loc = 'Melbourne', prd = 'Apr2020', qty = 'log(mobility_ratio)'), 
**mono_test_mk(df_mel, x = 'ER_local_decile', y = 'apr20_log_ratio_mob', loc = 'Melbourne', prd = 'Apr2020', qty = 'log(mobility_ratio)'), 
**mono_test_mk(df_syd, x = 'EO_local_decile', y = 'apr20_log_ratio_mob', loc = 'Sydney', prd = 'Apr2020', qty = 'log(mobility_ratio)'), 
**mono_test_mk(df_syd, x = 'ER_local_decile', y = 'apr20_log_ratio_mob', loc = 'Sydney', prd = 'Apr2020', qty = 'log(mobility_ratio)'), 

**mono_test_mk(df_mel, x = 'EO_local_decile', y = 'jan22_log_ratio_mob', loc = 'Melbourne', prd = 'Jan2022', qty = 'log(mobility_ratio)'), 
**mono_test_mk(df_mel, x = 'ER_local_decile', y = 'jan22_log_ratio_mob', loc = 'Melbourne', prd = 'Jan2022', qty = 'log(mobility_ratio)'), 
**mono_test_mk(df_syd, x = 'EO_local_decile', y = 'jan22_log_ratio_mob', loc = 'Sydney', prd = 'Jan2022', qty = 'log(mobility_ratio)'), 
**mono_test_mk(df_syd, x = 'ER_local_decile', y = 'jan22_log_ratio_mob', loc = 'Sydney', prd = 'Jan2022', qty = 'log(mobility_ratio)')}

#d = {(prd, x, loc, qty, 'score'): MK_res.s, (prd, x, loc, qty, 'p'): MK_res.p}

#%%
locs = ['Melbourne', 'Sydney']
prds = ['Apr2020', 'Jan2022']
qtys = ['log(mobility_ratio)']
seifas = ['EO_local_decile', 'ER_local_decile']

index = pd.MultiIndex.from_product([prds, seifas], names=['Period', 'SEIFA'])

columns = pd.MultiIndex.from_product([locs, qtys, ['score', 'p', 'LCI', 'UCI']], names=['City', 'qty', 'Mann-Kendall'])

data = np.ones((4, 8))*-3.14
#data = np.random.rand(4, 8)
table = pd.DataFrame(data, index=index, columns=columns)

for prd in prds:
    for seifa in seifas:
        for loc in locs:
            for qty in qtys:
                table.loc[prd, seifa][loc, qty, 'score'] = res_mk_dict[(prd, seifa, loc, qty, 'score')]
                table.loc[prd, seifa][loc, qty, 'p'] = res_mk_dict[(prd, seifa, loc, qty, 'p')]
                table.loc[prd, seifa][loc, qty, 'LCI'] = res_mk_dict[(prd, seifa, loc, qty, 'LCI')]
                table.loc[prd, seifa][loc, qty, 'UCI'] = res_mk_dict[(prd, seifa, loc, qty, 'UCI')]



table.to_html('coverage_adjusted_MK.html')
table.to_latex('coverage_adjusted_MK.tex')



#%%
res_spearman_dict = {**mono_test_spearman(df_mel, x = 'EO_local_decile', y = 'apr20_log_ratio_mob', loc = 'Melbourne', prd = 'Apr2020', qty = 'log(mobility_ratio)'), 
**mono_test_spearman(df_mel, x = 'ER_local_decile', y = 'apr20_log_ratio_mob', loc = 'Melbourne', prd = 'Apr2020', qty = 'log(mobility_ratio)'), 
**mono_test_spearman(df_syd, x = 'EO_local_decile', y = 'apr20_log_ratio_mob', loc = 'Sydney', prd = 'Apr2020', qty = 'log(mobility_ratio)'), 
**mono_test_spearman(df_syd, x = 'ER_local_decile', y = 'apr20_log_ratio_mob', loc = 'Sydney', prd = 'Apr2020', qty = 'log(mobility_ratio)'), 

**mono_test_spearman(df_mel, x = 'EO_local_decile', y = 'jan22_log_ratio_mob', loc = 'Melbourne', prd = 'Jan2022', qty = 'log(mobility_ratio)'), 
**mono_test_spearman(df_mel, x = 'ER_local_decile', y = 'jan22_log_ratio_mob', loc = 'Melbourne', prd = 'Jan2022', qty = 'log(mobility_ratio)'), 
**mono_test_spearman(df_syd, x = 'EO_local_decile', y = 'jan22_log_ratio_mob', loc = 'Sydney', prd = 'Jan2022', qty = 'log(mobility_ratio)'), 
**mono_test_spearman(df_syd, x = 'ER_local_decile', y = 'jan22_log_ratio_mob', loc = 'Sydney', prd = 'Jan2022', qty = 'log(mobility_ratio)')}

#%%
locs = ['Melbourne', 'Sydney']
prds = ['Apr2020', 'Jan2022']
qtys = ['log(mobility_ratio)']
seifas = ['EO_local_decile', 'ER_local_decile']
pd.set_option('display.float_format', lambda x: '%.4g' % x)
index = pd.MultiIndex.from_product([prds, seifas], names=['Period', 'SEIFA'])

columns = pd.MultiIndex.from_product([locs, qtys, ['score', 'p', 'Lower CI', 'Upper CI']], names=['City', 'qty', 'Spearman'])

data = np.ones((4, 8))*-3.14
#data = np.random.rand(4, 8)
table = pd.DataFrame(data, index=index, columns=columns)

for prd in prds:
    for seifa in seifas:
        for loc in locs:
            for qty in qtys:
                table.loc[prd, seifa][loc, qty, 'score'] = res_spearman_dict[(prd, seifa, loc, qty, 'score')]
                table.loc[prd, seifa][loc, qty, 'p'] = res_spearman_dict[(prd, seifa, loc, qty, 'p')]
                table.loc[prd, seifa][loc, qty, 'Lower CI'] = res_spearman_dict[(prd, seifa, loc, qty, 'CI_low')]
                table.loc[prd, seifa][loc, qty, 'Upper CI'] = res_spearman_dict[(prd, seifa, loc, qty, 'CI_high')]
                
# table.to_html('coverage_adjusted_Spearman.html')
# table.to_latex('coverage_adjusted_Spearman.tex')
# table.to_csv('coverage_adjusted_Spearman.csv')


#%% Histograms

fig, axs = plt.subplots(1, 2, figsize = (4*2, 4), sharey=True, sharex=True, 
                        constrained_layout = True)
A = axs[0].hist(df_mel.apr20_log_ratio_mob, bins = 100, histtype = 'step', density = True, label = 'April 2020')
axs[0].vlines(df_mel.apr20_log_ratio_mob.median(), 0, np.max(A[0]), lw = 1, ls = '--', colors = '#0072B2')
A = axs[0].hist(df_mel.jan22_log_ratio_mob, bins = 100, histtype = 'step', density = True, label = 'January 2022')
axs[0].vlines(df_mel.jan22_log_ratio_mob.median(), 0, np.max(A[0]), lw = 1, ls = '--', colors = '#009E73')

A = axs[1].hist(df_syd.apr20_log_ratio_mob, bins = 100, histtype = 'step', density = True, label = 'April 2020')
axs[1].vlines(df_syd.apr20_log_ratio_mob.median(), 0, np.max(A[0]), lw = 1, ls = '--', colors = '#0072B2')
A = axs[1].hist(df_syd.jan22_log_ratio_mob, bins = 100, histtype = 'step', density = True, label = 'January 2022')
axs[1].vlines(df_syd.jan22_log_ratio_mob.median(), 0, np.max(A[0]), lw = 1, ls = '--', colors = '#009E73')

axs[0].set_title(f'Melbourne [N = {df_mel.shape[0]}]')
axs[1].set_title(f'Sydney [N = {df_syd.shape[0]}]')

plt.xlim(-6, 6)

for ax in axs.flatten():
    ax.legend(fancybox = False, loc = 'upper left')

plt.suptitle(f'Distribution of {lm}')
fig.savefig('Refactored_code_plots_cov_all_SA1s/Histograms.pdf')

#%% Histograms -supp - test_v and base_v

fig, axs = plt.subplots(4, 2, figsize = (4*2, 8), sharey=True, sharex=True, 
                        constrained_layout = True)
bin_edges = np.arange(0, 6.1e5, 1000)
A = axs[0, 0].hist(df_mel.apr20base_v, bins = bin_edges, histtype = 'step', density = True, label = r'MEL - Apr 2020')
axs[0, 0].vlines(df_mel.apr20base_v.median(), 0, np.max(A[0]), lw = 1, ls = '--', colors = '#0072B2')
A = axs[0, 1].hist(df_mel.apr20test_v, bins = bin_edges, histtype = 'step', density = True, label = 'MEL - Apr 2020')
axs[0, 1].vlines(df_mel.apr20test_v.median(), 0, np.max(A[0]), lw = 1, ls = '--', colors = '#009E73')

axs[0, 0].set_xlabel(r'$A_{SA1}$')
axs[0, 1].set_xlabel(r'$B_{SA1}$')

A = axs[1, 0].hist(df_syd.apr20base_v, bins = bin_edges, histtype = 'step', density = True, label = 'SYD - Apr 2020')
axs[1, 0].vlines(df_syd.apr20base_v.median(), 0, np.max(A[0]), lw = 1, ls = '--', colors = '#0072B2')
A = axs[1, 1].hist(df_syd.apr20test_v, bins = bin_edges, histtype = 'step', density = True, label = 'SYD - Apr 2020')
axs[1, 1].vlines(df_syd.apr20test_v.median(), 0, np.max(A[0]), lw = 1, ls = '--', colors = '#009E73')

axs[1, 0].set_xlabel(r'$A_{SA1}$')
axs[1, 1].set_xlabel(r'$B_{SA1}$')

A = axs[2, 0].hist(df_mel.jan22base_v, bins = bin_edges, histtype = 'step', density = True, label = 'MEL - Jan 2022')
axs[2, 0].vlines(df_mel.jan22base_v.median(), 0, np.max(A[0]), lw = 1, ls = '--', colors = '#0072B2')
A = axs[2, 1].hist(df_mel.jan22test_v, bins = bin_edges, histtype = 'step', density = True, label = 'MEL - Jan 2022')
axs[2, 1].vlines(df_mel.jan22test_v.median(), 0, np.max(A[0]), lw = 1, ls = '--', colors = '#009E73')

axs[2, 0].set_xlabel(r'$A_{SA1}$')
axs[2, 1].set_xlabel(r'$B_{SA1}$')

A = axs[3, 0].hist(df_syd.jan22base_v, bins = bin_edges, histtype = 'step', density = True, label = 'SYD - Jan 2022')
axs[3, 0].vlines(df_syd.jan22base_v.median(), 0, np.max(A[0]), lw = 1, ls = '--', colors = '#0072B2')
A = axs[3, 1].hist(df_syd.jan22test_v, bins = bin_edges, histtype = 'step', density = True, label = 'SYD - Jan 2022')
axs[3, 1].vlines(df_syd.jan22test_v.median(), 0, np.max(A[0]), lw = 1, ls = '--', colors = '#009E73')

axs[3, 0].set_xlabel(r'$A_{SA1}$')
axs[3, 1].set_xlabel(r'$B_{SA1}$')
#axs[0].set_title(f'Melbourne [N = {df_mel.shape[0]}]')
#axs[1].set_title(f'Sydney [N = {df_syd.shape[0]}]')

#plt.xlim(-6, 6)

axs[0, 0].set_xlim(0, 1e5)
for ax in axs.flatten():
    ax.legend(fancybox = False, loc = 'upper left')

plt.suptitle(r'Distribution of $A_{SA1}$ and $B_{SA1}$')
fig.savefig('Refactored_code_plots_cov_all_SA1s/Histograms_A_B.pdf')

#%% SEIFA scores and lambda mob scatter plots and correlation

f, axs = plt.subplots(1, 2, sharex = True, sharey = True)

df_mel.plot(x = 'EO_score', y = 'apr20_log_ratio_mob', kind = 'scatter', alpha = 0.01, color = 'k', ax = axs[0])


df_mel.plot(x = 'ER_score', y = 'apr20_log_ratio_mob', kind = 'scatter', alpha = 0.01, color = 'k', ax = axs[1])

axs[0].set_ylim(-2, 2)

#%% Linear regression and PEarson's correlation
fig, axs = plt.subplots(2, 2, figsize = (4*2, 4*2), sharey=True, sharex=True, constrained_layout = True)
Y = 'apr20_log_ratio_mob'
slope, intercept, p, r, r_1, r_2 = linear_regression(df_mel.EO_score, df_mel.apr20_log_ratio_mob, CI = .95, namex = 'EO', namey = Y)
statlabel = f"Pearson's R: {r:.3f} \n95% CI: ({r_1:.3f}, {r_2:.3f}) \np-value: {p:.3g}"
draw_scatter(axs[0, 0], df_mel.EO_score.values, df_mel[Y], xlabel = 'EO score', ylabel = f'{lm}', title = 'Melbourne - April 2020', subplotlabel = 'A', statlabel = statlabel, slope = slope, intercept = intercept)

slope, intercept, p, r, r_1, r_2 = linear_regression(df_syd.EO_score, df_syd.apr20_log_ratio_mob, CI = .95, namex = 'EO', namey = Y)
statlabel = f"Pearson's R: {r:.3f} \n95% CI: ({r_1:.3f}, {r_2:.3f}) \np-value: {p:.3g}"
draw_scatter(axs[0, 1], df_syd.EO_score.values, df_syd[Y], xlabel = 'EO score', ylabel = f'{lm}', title = 'Sydney - April 2020', subplotlabel = 'B', statlabel = statlabel, slope = slope, intercept = intercept)

slope, intercept, p, r, r_1, r_2 = linear_regression(df_mel.ER_score, df_mel.apr20_log_ratio_mob, CI = .95, namex = 'EO', namey = Y)
statlabel = f"Pearson's R: {r:.3f} \n95% CI: ({r_1:.3f}, {r_2:.3f}) \np-value: {p:.3g}"
draw_scatter(axs[1, 0], df_mel.ER_score.values, df_mel[Y], xlabel = 'ER score', ylabel = f'{lm}', title = 'Melbourne - April 2020', subplotlabel = 'C', statlabel = statlabel, slope = slope, intercept = intercept)

slope, intercept, p, r, r_1, r_2 = linear_regression(df_syd.ER_score, df_syd.apr20_log_ratio_mob, CI = .95, namex = 'EO', namey = Y)
statlabel = f"Pearson's R: {r:.3f} \n95% CI: ({r_1:.3f}, {r_2:.3f}) \np-value: {p:.3g}"
draw_scatter(axs[1, 1], df_syd.ER_score.values, df_syd[Y], xlabel = 'ER score', ylabel = f'{lm}', title = 'Sydney - April 2020', subplotlabel = 'D', statlabel = statlabel, slope = slope, intercept = intercept)
axs[0, 0].set_ylim(-4, 4)
axs[0, 0].set_xlim(600, )
fig.savefig('Refactored_code_plots_cov_all_SA1s/Apr20-scatterplot.pdf')
fig.savefig('Refactored_code_plots_cov_all_SA1s/Apr20-scatterplot.png')

fig, axs = plt.subplots(2, 2, figsize = (4*2, 4*2), sharey=True, sharex=True, constrained_layout = True)
Y = 'jan22_log_ratio_mob'

slope, intercept, p, r, r_1, r_2 = linear_regression(df_mel.EO_score, df_mel.jan22_log_ratio_mob, CI = .95, namex = 'EO', namey = Y)
statlabel = f"Pearson's R: {r:.3f} \n95% CI: ({r_1:.3f}, {r_2:.3f}) \np-value: {p:.3g}"
draw_scatter(axs[0, 0], df_mel.EO_score.values, df_mel[Y], xlabel = 'EO score', ylabel = f'{lm}', title = 'Melbourne - Jan 2022', subplotlabel = 'A', statlabel = statlabel, slope = slope, intercept = intercept)

slope, intercept, p, r, r_1, r_2 = linear_regression(df_syd.EO_score, df_syd.jan22_log_ratio_mob, CI = .95, namex = 'EO', namey = Y)
statlabel = f"Pearson's R: {r:.3f} \n95% CI: ({r_1:.3f}, {r_2:.3f}) \np-value: {p:.3g}"
draw_scatter(axs[0, 1], df_syd.EO_score.values, df_syd[Y], xlabel = 'EO score', ylabel = f'{lm}', title = 'Sydney - Jan 2022', subplotlabel = 'B', statlabel = statlabel, slope = slope, intercept = intercept)

slope, intercept, p, r, r_1, r_2 = linear_regression(df_mel.ER_score, df_mel.jan22_log_ratio_mob, CI = .95, namex = 'EO', namey = Y)
statlabel = f"Pearson's R: {r:.3f} \n95% CI: ({r_1:.3f}, {r_2:.3f}) \np-value: {p:.3g}"
draw_scatter(axs[1, 0], df_mel.ER_score.values, df_mel[Y], xlabel = 'ER score', ylabel = f'{lm}', title = 'Melbourne - Jan 2022', subplotlabel = 'C', statlabel = statlabel, slope = slope, intercept = intercept)

slope, intercept, p, r, r_1, r_2 = linear_regression(df_syd.ER_score, df_syd.jan22_log_ratio_mob, CI = .95, namex = 'EO', namey = Y)
statlabel = f"Pearson's R: {r:.3f} \n95% CI: ({r_1:.3f}, {r_2:.3f}) \np-value: {p:.3g}"
draw_scatter(axs[1, 1], df_syd.ER_score.values, df_syd[Y], xlabel = 'ER score', ylabel = f'{lm}', title = 'Sydney - Jan 2022', subplotlabel = 'D', statlabel = statlabel, slope = slope, intercept = intercept)

axs[0, 0].set_ylim(-2, 2)
axs[0, 0].set_xlim(600, )
fig.savefig('Refactored_code_plots_cov_all_SA1s/Jan22-scatterplot.pdf')
fig.savefig('Refactored_code_plots_cov_all_SA1s/Jan22-scatterplot.png')

#======================================================================================================
#%% Spearman's correlation
# fig, axs = plt.subplots(2, 2, figsize = (4*2, 4*2), sharey=True, sharex=True, constrained_layout = True)
# Y = 'apr20_log_ratio_mob'
# r, pvalue, r_1, r_2 = spearman_corr(df_mel.EO_score, df_mel.apr20_log_ratio_mob, CI = .95, namex = 'EO', namey = Y)
# statlabel = f"Spearman's R: {r:.3f} \n95% CI: ({r_1:.3f}, {r_2:.3f}) \np-value: {pvalue:.3g}"
# draw_scatter(axs[0, 0], df_mel.EO_score.values, df_mel[Y], xlabel = 'EO score', ylabel = f'{lm}', title = 'Melbourne - April 2020', subplotlabel = 'A', statlabel = statlabel)

# r, pvalue, r_1, r_2 = spearman_corr(df_syd.EO_score, df_syd.apr20_log_ratio_mob, CI = .95, namex = 'EO', namey = Y)
# statlabel = f"Spearman's R: {r:.3f} \n95% CI: ({r_1:.3f}, {r_2:.3f}) \np-value: {pvalue:.3g}"
# draw_scatter(axs[0, 1], df_syd.EO_score.values, df_syd[Y], xlabel = 'EO score', ylabel = f'{lm}', title = 'Sydney - April 2020', subplotlabel = 'B', statlabel = statlabel)

# r, pvalue, r_1, r_2 = spearman_corr(df_mel.ER_score, df_mel.apr20_log_ratio_mob, CI = .95, namex = 'EO', namey = Y)
# statlabel = f"Spearman's R: {r:.3f} \n95% CI: ({r_1:.3f}, {r_2:.3f}) \np-value: {pvalue:.3g}"
# draw_scatter(axs[1, 0], df_mel.ER_score.values, df_mel[Y], xlabel = 'ER score', ylabel = f'{lm}', title = 'Melbourne - April 2020', subplotlabel = 'C', statlabel = statlabel)

# r, pvalue, r_1, r_2 = spearman_corr(df_syd.ER_score, df_syd.apr20_log_ratio_mob, CI = .95, namex = 'EO', namey = Y)
# statlabel = f"Spearman's R: {r:.3f} \n95% CI: ({r_1:.3f}, {r_2:.3f}) \np-value: {pvalue:.3g}"
# draw_scatter(axs[1, 1], df_syd.ER_score.values, df_syd[Y], xlabel = 'ER score', ylabel = f'{lm}', title = 'Sydney - April 2020', subplotlabel = 'D', statlabel = statlabel)
# axs[0, 0].set_ylim(-4, 4)
# axs[0, 0].set_xlim(600, )

# fig.savefig('Refactored_code_plots_cov_all_SA1s/Apr20-spearman-scatterplot.png')

# fig, axs = plt.subplots(2, 2, figsize = (4*2, 4*2), sharey=True, sharex=True, constrained_layout = True)
# Y = 'jan22_log_ratio_mob'

# r, pvalue, r_1, r_2 = spearman_corr(df_mel.EO_score, df_mel.jan22_log_ratio_mob, CI = .95, namex = 'EO', namey = Y)
# statlabel = f"Spearman's R: {r:.3f} \n95% CI: ({r_1:.3f}, {r_2:.3f}) \np-value: {pvalue:.3g}"
# draw_scatter(axs[0, 0], df_mel.EO_score.values, df_mel[Y], xlabel = 'EO score', ylabel = f'{lm}', title = 'Melbourne - Jan 2022', subplotlabel = 'A', statlabel = statlabel)

# r, pvalue, r_1, r_2 = spearman_corr(df_syd.EO_score, df_syd.jan22_log_ratio_mob, CI = .95, namex = 'EO', namey = Y)
# statlabel = f"Spearman's R: {r:.3f} \n95% CI: ({r_1:.3f}, {r_2:.3f}) \np-value: {pvalue:.3g}"
# draw_scatter(axs[0, 1], df_syd.EO_score.values, df_syd[Y], xlabel = 'EO score', ylabel = f'{lm}', title = 'Sydney - Jan 2022', subplotlabel = 'B', statlabel = statlabel)

# r, pvalue, r_1, r_2 = spearman_corr(df_mel.ER_score, df_mel.jan22_log_ratio_mob, CI = .95, namex = 'EO', namey = Y)
# statlabel = f"Spearman's R: {r:.3f} \n95% CI: ({r_1:.3f}, {r_2:.3f}) \np-value: {pvalue:.3g}"
# draw_scatter(axs[1, 0], df_mel.ER_score.values, df_mel[Y], xlabel = 'ER score', ylabel = f'{lm}', title = 'Melbourne - Jan 2022', subplotlabel = 'C', statlabel = statlabel)

# r, pvalue, r_1, r_2 = spearman_corr(df_syd.ER_score, df_syd.jan22_log_ratio_mob, CI = .95, namex = 'EO', namey = Y)
# statlabel = f"Spearman's R: {r:.3f} \n95% CI: ({r_1:.3f}, {r_2:.3f}) \np-value: {pvalue:.3g}"
# draw_scatter(axs[1, 1], df_syd.ER_score.values, df_syd[Y], xlabel = 'ER score', ylabel = f'{lm}', title = 'Sydney - Jan 2022', subplotlabel = 'D', statlabel = statlabel)

# axs[0, 0].set_ylim(-2, 2)
# axs[0, 0].set_xlim(600, )

# fig.savefig('Refactored_code_plots_cov_all_SA1s/Jan22-spearman-scatterplot.png')



#%%
# =============================================================================
# #%% Relative mobility w/o log transform
# 
# #%% Heatmaps Melbourne
# fname = 'Mel-'
# hdata = compute_heatmap(df_mel.ER_local_decile.values, df_mel.EO_local_decile.values, df_mel.apr20_rel_mob.values, 10)
# draw_heatmap_zero_centre(hdata, fname, 'RdBu_r', title = 'Melbourne - April 2020 (Adjusted for coverage)', cbar_title=f'Median {rm}')
# hdata = compute_heatmap(df_mel.ER_local_decile.values, df_mel.EO_local_decile.values, df_mel.jan22_rel_mob.values, 10)
# draw_heatmap_zero_centre(hdata, fname, 'RdBu_r', title = 'Melbourne - Jan 2022 (Adjusted for coverage)', cbar_title=f'Median {rm}')
# SA1_hist_mel = compute_heatmap(df_mel.ER_local_decile.values, df_mel.EO_local_decile.values, qty = 'histogram')
# f = draw_heatmap(SA1_hist_mel, fname = fname, title = 'Counts of SA1s in Melbourne (Adjusted for coverage)', fmt = '0.0f')
# 
# #%% Heatmaps Sydney
# fname = 'Syd-'
# hdata = compute_heatmap(df_syd.ER_local_decile.values, df_syd.EO_local_decile.values, df_syd.apr20_rel_mob.values, 10)
# draw_heatmap_zero_centre(hdata, fname, 'RdBu_r', title = 'Sydney - April 2020 (Adjusted for coverage)', cbar_title=f'Median {rm}')
# hdata = compute_heatmap(df_syd.ER_local_decile.values, df_syd.EO_local_decile.values, df_syd.jan22_rel_mob.values, 10)
# draw_heatmap_zero_centre(hdata, fname, 'RdBu_r', title = 'Sydney - Jan 2022 (Adjusted for coverage)', cbar_title=f'Median {rm}')
# SA1_hist_syd = compute_heatmap(df_syd.ER_local_decile.values, df_syd.EO_local_decile.values, qty = 'histogram')
# f = draw_heatmap(SA1_hist_syd, fname = fname, title = 'Counts of SA1s in Sydney (Adjusted for coverage)', fmt = '0.0f')
# 
# #%% Scatter plots
# fig, axs = plt.subplots(2, 2, figsize = (4*2, 4*2), sharey=True, sharex=True, constrained_layout = True)
# Y = 'apr20_rel_mob'
# draw_scatter(axs[0, 0], df_mel.EO_local_decile.values, df_mel[Y], xlabel = 'Local deciles of EO', ylabel = f'{rm}', title = 'Melbourne - April 2020 (Adjusted for coverage)', subplotlabel = 'A')
# draw_scatter(axs[0, 1], df_syd.EO_local_decile.values, df_syd[Y], xlabel = 'Local deciles of EO', ylabel = f'{rm}', title = 'Sydney - April 2020 (Adjusted for coverage)', subplotlabel = 'B')
# draw_scatter(axs[1, 0], df_mel.ER_local_decile.values, df_mel[Y], xlabel = 'Local deciles of ER', ylabel = f'{rm}', title = 'Melbourne - April 2020 (Adjusted for coverage)', subplotlabel = 'C')
# draw_scatter(axs[1, 1], df_syd.ER_local_decile.values, df_syd[Y], xlabel = 'Local deciles of ER', ylabel = f'{rm}', title = 'Sydney - April 2020 (Adjusted for coverage)', subplotlabel = 'D')
# handles, legend_labels = axs[1, 1].get_legend_handles_labels()
# fig.legend(handles, legend_labels, loc = 'outside upper right', ncols = 3, fancybox = False)
# #fig.savefig(f'Apr20-scatterplot.png', dpi=400)
# 
# fig, axs = plt.subplots(2, 2, figsize = (4*2, 4*2), sharey=True, sharex=True, constrained_layout = True)
# Y = 'jan22_rel_mob'
# draw_scatter(axs[0, 0], df_mel.EO_local_decile.values, df_mel[Y], xlabel = 'Local deciles of EO', ylabel = f'{rm}', title = 'Melbourne - Jan 2022 (Adjusted for coverage)', subplotlabel = 'A')
# draw_scatter(axs[0, 1], df_syd.EO_local_decile.values, df_syd[Y], xlabel = 'Local deciles of EO', ylabel = f'{rm}', title = 'Sydney - Jan 2022 (Adjusted for coverage)', subplotlabel = 'B')
# draw_scatter(axs[1, 0], df_mel.ER_local_decile.values, df_mel[Y], xlabel = 'Local deciles of ER', ylabel = f'{rm}', title = 'Melbourne - Jan 2022 (Adjusted for coverage)', subplotlabel = 'C')
# draw_scatter(axs[1, 1], df_syd.ER_local_decile.values, df_syd[Y], xlabel = 'Local deciles of ER', ylabel = f'{rm}', title = 'Sydney - Jan 2022 (Adjusted for coverage)', subplotlabel = 'D')
# handles, legend_labels = axs[1, 1].get_legend_handles_labels()
# fig.legend(handles, legend_labels, loc = 'outside upper right', ncols = 3, fancybox = False)
# 
# #fig.savefig(f'Jan22-scatterplot.png', dpi=400)
# 
# #%% Histograms
# 
# fig, axs = plt.subplots(1, 2, figsize = (4*2, 4), sharey=True, sharex=True, 
#                         constrained_layout = True)
# A = axs[0].hist(df_mel.apr20_rel_mob, bins = 100, histtype = 'step', density = True, label = f'April 2020 [N = {df_mel.shape[0]}]')
# axs[0].vlines(df_mel.apr20_rel_mob.median(), 0, np.max(A[0]), lw = 1, ls = '--', colors = '#0072B2')
# A = axs[0].hist(df_mel.jan22_rel_mob, bins = 100, histtype = 'step', density = True, label = f'January 2022 [N = {df_mel.shape[0]}]')
# axs[0].vlines(df_mel.jan22_rel_mob.median(), 0, np.max(A[0]), lw = 1, ls = '--', colors = '#009E73')
# 
# A = axs[1].hist(df_syd.apr20_rel_mob, bins = 100, histtype = 'step', density = True, label = f'April 2020 [N = {df_syd.shape[0]}]')
# axs[1].vlines(df_syd.apr20_rel_mob.median(), 0, np.max(A[0]), lw = 1, ls = '--', colors = '#0072B2')
# A = axs[1].hist(df_syd.jan22_rel_mob, bins = 100, histtype = 'step', density = True, label = f'January 2022 [N = {df_syd.shape[0]}]')
# axs[1].vlines(df_syd.jan22_rel_mob.median(), 0, np.max(A[0]), lw = 1, ls = '--', colors = '#009E73')
# 
# axs[0].set_title('Melbourne')
# axs[1].set_title('Sydney')
# 
# plt.xlim(-6, 6)
# 
# for ax in axs.flatten():
#     ax.legend(fancybox = False, loc = 'upper left')
# 
# plt.suptitle(f'Distribution of {rm} (Adjusted for coverage)')
# 
# 
# 
# =============================================================================
