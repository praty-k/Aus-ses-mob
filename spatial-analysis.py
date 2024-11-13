# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:11:25 2024

@author: pkollepara
"""

import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
import pandas as pd
from pandas.plotting import scatter_matrix
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import seaborn as sns
#from patsy import dmatrices

import geopandas as gpd
#from pysal.viz import splot
#from splot.esda import plot_moran
#from pysal.explore import esda
import spreg
#from pysal.lib import weights
#import contextily

#%% Importing functions
from dataframe_prep_w_cov_all_SA1s import dataframe_prep
from draw_lplot import draw_lplot
from draw_jplot import draw_scatter, draw_hex
from ER_EO_heatmap import compute_heatmap, draw_heatmap, draw_heatmap_zero_centre 
from mono_test import mono_test_mk, mono_test_spearman
from corr import spearman_corr, linear_regression
#%%

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
#%% Plotting symbols
lm = '$\lambda_{\mathrm{mob}}$'
lc = '$\lambda_{\mathrm{cov}}$'
lm_av = r'$\langle\lambda_{\mathrm{mob}}\rangle$'
#rm = '$\Delta v/v$'
#%% Read in tables containing counts from each time period, SEIFA scores, SA1 code and URP
df_mel = pd.read_csv('MEL__seifa_all_filter.csv', na_values='-',
                     usecols=['SA1', 'ER_score', 'EO_score', 'URP', 'apr20base_count', 'apr20wave_count', 'jan22base_count', 'jan22wave_count'])
print(df_mel.shape, df_mel.sample(8))
print(df_mel.describe())

df_syd = pd.read_csv('SYD__seifa_all_filter.csv', na_values='-',
                     usecols=['SA1', 'ER_score', 'EO_score', 'URP', 'apr20base_count', 'apr20wave_count', 'jan22base_count', 'jan22wave_count'])


gdf_Aus_SA1 = gpd.read_file("SA1_2016_AUST.shp", include_fields = ['SA1_MAIN16', 'AREASQKM16', 'geometry'])
gdf_Aus_SA1['SA1_MAIN16'] = gdf_Aus_SA1['SA1_MAIN16'].astype('int64')

gdf_Aus_suburbs = gpd.read_file("SAL_2021_AUST_GDA2020.shp")
gdf_mel_cbd = gdf_Aus_suburbs[gdf_Aus_suburbs.SAL_NAME21 == 'Melbourne'] 


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

#%% Remove outliers
retain_proportion = 100/100 #68, 95, 99.7
mel_apr20_lower = df_mel.apr20_log_ratio_mob.quantile((1-retain_proportion)/2)
mel_apr20_upper = df_mel.apr20_log_ratio_mob.quantile(retain_proportion + (1-retain_proportion)/2)
mel_jan22_lower = df_mel.jan22_log_ratio_mob.quantile((1-retain_proportion)/2)
mel_jan22_upper = df_mel.jan22_log_ratio_mob.quantile(retain_proportion + (1-retain_proportion)/2)
df_mel = df_mel[(df_mel.apr20_log_ratio_mob >= mel_apr20_lower) & (df_mel.apr20_log_ratio_mob <= mel_apr20_upper) & (df_mel.jan22_log_ratio_mob > mel_jan22_lower) & (df_mel.jan22_log_ratio_mob < mel_jan22_upper)]

syd_apr20_lower = df_syd.apr20_log_ratio_mob.quantile((1-retain_proportion)/2)
syd_apr20_upper = df_syd.apr20_log_ratio_mob.quantile(retain_proportion + (1-retain_proportion)/2)
syd_jan22_lower = df_syd.jan22_log_ratio_mob.quantile((1-retain_proportion)/2)
syd_jan22_upper = df_syd.jan22_log_ratio_mob.quantile(retain_proportion + (1-retain_proportion)/2)
df_syd = df_syd[(df_syd.apr20_log_ratio_mob >= syd_apr20_lower) & (df_syd.apr20_log_ratio_mob <= syd_apr20_upper) & (df_syd.jan22_log_ratio_mob > syd_jan22_lower) & (df_syd.jan22_log_ratio_mob < syd_jan22_upper)]

#%% Prepare dataframes: remove NaNs and compute local deciles of EO and ER
df_mel = dataframe_prep(df_mel)
df_syd = dataframe_prep(df_syd)

#%% Compute population density and distance from CBD

#gdf_full_area = gpd.read_file("SA1_2021_shapefiles/SA1_2021_AUST_GDA2020.shp", include_fields = ['SA1_CODE21', 'GCC_NAME21', 'STE_NAME_21', 'AREASQKM21'])
# The above line is incorrect since we need to use 2016 boundaries

# SA1_2016 are in 'EPSG:4283' 


gdf_mel_SA1 = gdf_Aus_SA1.merge(right = df_mel, how = 'inner', left_on = 'SA1_MAIN16', right_on = 'SA1')
gdf_mel_SA1.geometry = gdf_mel_SA1.geometry.to_crs('EPSG:28348') # converting to a projected crs
gdf_mel_SA1['centroid'] = gdf_mel_SA1.centroid

gdf_syd_SA1 = gdf_Aus_SA1.merge(right = df_syd, how = 'inner', left_on = 'SA1_MAIN16', right_on = 'SA1')
gdf_syd_SA1.geometry = gdf_syd_SA1.geometry.to_crs('EPSG:28348') # converting to a projected crs
gdf_syd_SA1['centroid'] = gdf_syd_SA1.centroid

gdf_mel_cbd['geometry'] = gdf_mel_cbd['geometry'].to_crs('EPSG:28348') #converting to a projected crs
gdf_mel_SA1['distance_to_CBD'] = gdf_mel_SA1['centroid'].distance(gdf_mel_cbd.centroid.iloc[0], align = False) #Distance in metres

gdf_syd_cbd = gdf_Aus_suburbs[gdf_Aus_suburbs.SAL_NAME21 == 'Sydney'] 
gdf_syd_cbd['geometry'] = gdf_syd_cbd['geometry'].to_crs('EPSG:28348') #converting to a projected crs
gdf_syd_SA1['distance_to_CBD'] = gdf_syd_SA1['centroid'].distance(gdf_syd_cbd.centroid.iloc[0], align = False) #Distance in metres

df_mel = df_mel.merge(right = gdf_mel_SA1[['SA1', 'distance_to_CBD', 'AREASQKM16']], how = 'inner', on = 'SA1')
df_mel['distance_to_CBD'] = df_mel['distance_to_CBD']/1000.0 #Converting to kms
df_mel['pop_density'] = df_mel['URP']/df_mel['AREASQKM16']

df_syd = df_syd.merge(right = gdf_syd_SA1[['SA1', 'distance_to_CBD', 'AREASQKM16']], how = 'inner', on = 'SA1')
df_syd['distance_to_CBD'] = df_syd['distance_to_CBD']/1000.0 #Converting to kms
df_syd['pop_density'] = df_syd['URP']/df_syd['AREASQKM16']


#%% Standardise predictor variables and dependent variables
df_mel['ER_score_standard'] = (df_mel['ER_score'] - df_mel['ER_score'].mean())/df_mel['ER_score'].std()
df_mel['EO_score_standard'] = (df_mel['EO_score'] - df_mel['EO_score'].mean())/df_mel['EO_score'].std()
df_mel['pop_density_standard'] = (df_mel.pop_density - df_mel.pop_density.mean())/df_mel.pop_density.std()
df_mel['distance_to_CBD_standard'] = (df_mel.distance_to_CBD - df_mel.distance_to_CBD.mean())/df_mel.distance_to_CBD.std()

df_syd['ER_score_standard'] = (df_syd['ER_score'] - df_syd['ER_score'].mean())/df_syd['ER_score'].std()
df_syd['EO_score_standard'] = (df_syd['EO_score'] - df_syd['EO_score'].mean())/df_syd['EO_score'].std()
df_syd['pop_density_standard'] = (df_syd.pop_density - df_syd.pop_density.mean())/df_syd.pop_density.std()
df_syd['distance_to_CBD_standard'] = (df_syd.distance_to_CBD - df_syd.distance_to_CBD.mean())/df_syd.distance_to_CBD.std()

df_mel['apr20_log_ratio_mob_standard'] = (df_mel.apr20_log_ratio_mob - df_mel.apr20_log_ratio_mob.mean())/df_mel.apr20_log_ratio_mob.std()
df_mel['jan22_log_ratio_mob_standard'] = (df_mel.jan22_log_ratio_mob - df_mel.jan22_log_ratio_mob.mean())/df_mel.jan22_log_ratio_mob.std()

df_syd['apr20_log_ratio_mob_standard'] = (df_syd.apr20_log_ratio_mob - df_syd.apr20_log_ratio_mob.mean())/df_syd.apr20_log_ratio_mob.std()
df_syd['jan22_log_ratio_mob_standard'] = (df_syd.jan22_log_ratio_mob - df_syd.jan22_log_ratio_mob.mean())/df_syd.jan22_log_ratio_mob.std()


#%% Join geometry to data tables


gdf_mel = gdf_Aus_SA1[['SA1_MAIN16', 'geometry']].merge(right = df_mel, how = 'right', left_on = 'SA1_MAIN16', right_on = 'SA1')
gdf_syd = gdf_Aus_SA1[['SA1_MAIN16', 'geometry']].merge(right = df_syd, how = 'right', left_on = 'SA1_MAIN16', right_on = 'SA1')

gdf_mel.to_file('MEL_processed_geodataframe.shp')  
gdf_syd.to_file('SYD_processed_geodataframe.shp')  

gdf_mel.to_file('MEL_processed_geodataframe_geopackage.gpkg')  
gdf_syd.to_file('SYD_processed_geodataframe_geopackage.gpkg')  

#%% Read processed data tables
gdf_mel = gpd.read_file('MEL_processed_geodataframe_geopackage.gpkg')
gdf_syd = gpd.read_file('SYD_processed_geodataframe_geopackage.gpkg')

# f, ax = plt.subplots(1, 2)
# gdf_mel.plot(ax = ax[0], column = 'apr20_log_ratio_mob', cmap = 'RdBu_r', legend = True, scheme = 'quantiles')
# gdf_mel['apr_20_log_ratio_mob_rnd'] = gdf_mel.apr20_log_ratio_mob.sample(frac = 1).reset_index(drop=True)
# gdf_mel.plot(ax = ax[1], column = 'apr_20_log_ratio_mob_rnd', cmap = 'RdBu_r', legend = True, scheme = 'quantiles')
# gdf_mel_cbd.boundary.plot(ax = ax[0], )

# y, X = dmatrices('apr20_log_ratio_mob ~ ER_score_standard + EO_score_standard', data = gdf_mel, return_type = 'dataframe')
# model = sm.OLS(endog = y, exog = X)
# results_mel_apr20 = model.fit()
# gdf_mel['apr20_fit_resid'] = results_mel_apr20.resid
# gdf_mel.plot(column = 'apr20_fit_resid', cmap = 'RdBu_r', legend = True, scheme = 'quantiles')

#%% Spatial lag and error models
def run_analysis(gdf, Y, df_name, predictor_names, nn = 8, draw_maps = False, plot_moran = False, do_spatial_regression = False):
    pass
    wk = weights.distance.KNN.from_dataframe(gdf, k=nn)
    wk.transform = 'R'
    Y_lag = Y + '_lag'
    Y_lag_std = Y + '_lag_standard'
    Y_std = Y + '_standard'
    gdf[Y_lag] = weights.spatial_lag.lag_spatial(wk, gdf[Y])
    gdf[Y_lag_std] = weights.spatial_lag.lag_spatial(wk, gdf[Y_std])
    
    # Fit a non-spatial OLS to the data
    dependent_name = Y_std
    #predictor_names = ['ER_score_standard', 'EO_score_standard', 'pop_density_standard', 'distance_to_CBD_standard']
    fit_non_spatial = spreg.OLS(
                    y = gdf[dependent_name].values, 
                    x = gdf[predictor_names].values, 
                    name_y = Y_std, 
                    name_x = predictor_names, 
                    name_ds = 'gdf')
    f, ax = plt.subplots(1, 1)
    qqplot(data = fit_non_spatial.u[:, 0], line = '45', markerfacecolor='none', alpha = 0.5, ax = ax)
    
    ax.set(title = df_name + ' ' + Y)
    plt.savefig('qqplot_' + df_name + '_' + Y + '.pdf')
    plt.close()
    if draw_maps:
        # Visualize lag on map
        f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        gdf.plot(ax = ax[0], column = Y, cmap = 'RdBu_r', legend = True, scheme = 'quantiles')
        ax[0].set_title(Y)
        gdf.plot(ax = ax[1], column = Y_lag, cmap = 'RdBu_r', legend = True, scheme = 'quantiles')
        ax[1].set_title(Y_lag)

        # VIsualize residuals on map and on qqplot
        gdf
        f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        gdf.plot(ax = ax[0], column = Y, cmap = 'RdBu_r', legend = True, scheme = 'quantiles')
        ax[0].set_title(Y)
        gdf.plot(ax = ax[1], column = fit_non_spatial.u[:, 0], cmap = 'RdBu_r', legend = True, scheme = 'quantiles')
        ax[1].set_title(Y+'_standard_residuals')
        
    # Compute lag
    gdf[Y_lag_std] = weights.spatial_lag.lag_spatial(wk, gdf[Y_std])
    
    # Moran plot and I
    
    if plot_moran:
        f, ax = plt.subplots(1, 1)
        moran = esda.moran.Moran(gdf[Y_std], wk)
        splot.esda.moran_scatterplot(moran, ax = ax, scatter_kwds={'alpha': 0.5, 'facecolor': 'none', 'edgecolor': 'lightgrey'})
        plt.suptitle(df_name + ' ' + Y_std)
        ax.set(title = f"Moran's I: {moran.I}, p-value: {moran.p_sim}")
        plt.savefig('moran_' + df_name + '_' + Y + '.pdf')
        plt.close()

    with open(f'Compact_summary_{df_name}_{Y}.txt', 'w') as f:
        print('__________________ \n', file = f)
        print(f'dataframe: {df_name}', file = f)
        print(f'Y: {Y}', file = f)
        print('__________________ \n', file = f)
        
        x = fit_non_spatial
        print('Non spatial OLS', file = f)
        print('---------------', file = f)
        print(f'AIC = {x.aic}', file = f)
        print(f'Predictor  |  Coefficient  |  p-value', file = f)
        for name, coeff, p in zip(x.name_x, x.betas[:, 0], x.t_stat):
            print(f'{name}  |  {coeff}  |  {p[1]}', file = f)
        print(f'EXP[Residuals] = {np.mean(fit_non_spatial.u[:, 0])}')
        print('', file = f)
        
        #
        if do_spatial_regression:
            fit_lag = spreg.ML_Lag(
                y = gdf[dependent_name].values, x = gdf[predictor_names].values, w = wk,
                name_y = Y_std, name_x = predictor_names, name_ds = df_name)
            
            fit_error = spreg.ML_Error(
                y = gdf[dependent_name].values, x = gdf[predictor_names].values, w = wk,
                name_y = Y_std, name_x = predictor_names, name_ds = df_name)
            #
            
            
            x = fit_lag
            print('Spatial lag model', file = f)
            print('---------------', file = f)
            print(f'AIC = {x.aic}', file = f)
            print(f'Predictor  |  Coefficient  |  p-value', file = f)
            for name, coeff, p in zip(x.name_x, x.betas[:, 0], x.z_stat):
                print(f'{name}  |  {coeff}  |  {p[1]}', file = f)
            print('', file = f)
            
            x = fit_error
            print('Spatial error model', file = f)
            print('---------------', file = f)
            print(f'AIC = {x.aic}', file = f)
            print(f'Predictor  |  Coefficient  |  p-value', file = f)
            for name, coeff, p in zip(x.name_x, x.betas[:, 0], x.z_stat):
                print(f'{name}  |  {coeff}  |  {p[1]}', file = f)
            print('\n', file = f)
            
    with open(f'Expanded_summary_{df_name}_{Y}.txt', 'w') as f:
        print(fit_non_spatial.summary, file = f)
        if do_spatial_regression:
            print(fit_lag.summary, file = f)
            print(fit_error.summary, file = f)
    
def run_analysis(gdf, Y, df_name, predictor_names, nn = 8, draw_maps = False, plot_moran = False, do_spatial_regression = False, compact_summary_file = None):
    wk = weights.distance.KNN.from_dataframe(gdf, k=nn)
    wk.transform = 'R'
    Y_lag = Y + '_lag'
    Y_lag_std = Y + '_lag_standard'
    Y_std = Y + '_standard'
    gdf[Y_lag] = weights.spatial_lag.lag_spatial(wk, gdf[Y])
    gdf[Y_lag_std] = weights.spatial_lag.lag_spatial(wk, gdf[Y_std])
    
    # Fit a non-spatial OLS to the data
    dependent_name = Y_std
    #predictor_names = ['ER_score_standard', 'EO_score_standard', 'pop_density_standard', 'distance_to_CBD_standard']
    fit_non_spatial = spreg.OLS(
                    y = gdf[dependent_name].values, 
                    x = gdf[predictor_names].values, 
                    name_y = Y_std, 
                    name_x = predictor_names, 
                    name_ds = 'gdf')
    f, ax = plt.subplots(1, 1)
    qqplot(data = fit_non_spatial.u[:, 0], line = '45', markerfacecolor='none', alpha = 0.5, ax = ax)
    
    ax.set(title = df_name + ' ' + Y)
    plt.savefig('qqplot_' + df_name + '_' + Y + '.pdf')
    plt.close()
    if draw_maps:
        # Visualize lag on map
        f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        gdf.plot(ax = ax[0], column = Y, cmap = 'RdBu_r', legend = True, scheme = 'quantiles')
        ax[0].set_title(Y)
        gdf.plot(ax = ax[1], column = Y_lag, cmap = 'RdBu_r', legend = True, scheme = 'quantiles')
        ax[1].set_title(Y_lag)

        # VIsualize residuals on map and on qqplot
        gdf
        f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        gdf.plot(ax = ax[0], column = Y, cmap = 'RdBu_r', legend = True, scheme = 'quantiles')
        ax[0].set_title(Y)
        gdf.plot(ax = ax[1], column = fit_non_spatial.u[:, 0], cmap = 'RdBu_r', legend = True, scheme = 'quantiles')
        ax[1].set_title(Y+'_standard_residuals')
        
    # Compute lag
    gdf[Y_lag_std] = weights.spatial_lag.lag_spatial(wk, gdf[Y_std])
    
    # Moran plot and I
    
    if plot_moran:
        f, ax = plt.subplots(1, 1)
        moran = esda.moran.Moran(gdf[Y_std], wk)
        splot.esda.moran_scatterplot(moran, ax = ax, scatter_kwds={'alpha': 0.5, 'facecolor': 'none', 'edgecolor': 'lightgrey'})
        plt.suptitle(df_name + ' ' + Y_std)
        ax.set(title = f"Moran's I: {moran.I}, p-value: {moran.p_sim}")
        plt.savefig('moran_' + df_name + '_' + Y + '.pdf')
        plt.close()

    #with open(f'Compact_summary_{df_name}_{Y}.csv', 'w') as f:
        #print('__________________ \n', file = f)
    print(f'dataframe,{df_name}', file = compact_summary_file)
    print(f'Y, {Y}', file = compact_summary_file)
    #print('__________________ \n', file = f)
    
    x = fit_non_spatial
    print('Non spatial OLS', file = compact_summary_file)
    #print('---------------', file = f)
    print(f'AIC, {x.aic}', file = compact_summary_file)
    print(f'Predictor,Coefficient,p-value', file = compact_summary_file)
    for name, coeff, p in zip(x.name_x, x.betas[:, 0], x.t_stat):
        print(f'{name},{coeff},{p[1]}', file = compact_summary_file)
    print(f'EXP[Residuals] = {np.mean(fit_non_spatial.u[:, 0])}')
    print('', file = compact_summary_file)
    
    #
    if do_spatial_regression:
        # fit_lag = spreg.ML_Lag(
        #     y = gdf[dependent_name].values, x = gdf[predictor_names].values, w = wk,
        #     name_y = Y_std, name_x = predictor_names, name_ds = df_name)
        
        fit_error = spreg.ML_Error(
            y = gdf[dependent_name].values, x = gdf[predictor_names].values, w = wk,
            name_y = Y_std, name_x = predictor_names, name_ds = df_name)
        # #
        
        
        # x = fit_lag
        # print('Spatial lag model', file = compact_summary_file)
        # #print('---------------', file = f)
        # print(f'AIC,{x.aic}', file = compact_summary_file)
        # print(f'Predictor,Coefficient,p-value', file = compact_summary_file)
        # for name, coeff, p in zip(x.name_x, x.betas[:, 0], x.z_stat):
        #     print(f'{name},{coeff},{p[1]}', file = compact_summary_file)
        # print('', file = compact_summary_file)
        
        x = fit_error
        print('Spatial error model', file = compact_summary_file)
        #print('---------------', file = f)
        print(f'AIC,{x.aic}', file = compact_summary_file)
        print(f'Predictor,Coefficient,p-value', file = compact_summary_file)
        for name, coeff, p in zip(x.name_x, x.betas[:, 0], x.z_stat):
            print(f'{name},{coeff},{p[1]}', file = compact_summary_file)
        #print('\n', file = f)
            
    with open(f'Expanded_summary_{df_name}_{Y}.txt', 'w') as f:
        print(fit_non_spatial.summary, file = f)
        if do_spatial_regression:
            #print(fit_lag.summary, file = f)
            print(fit_error.summary, file = f)

#%%
predictor_names = ['ER_score_standard', 'EO_score_standard']
f = open(f'compact_summary_{predictor_names}.csv', 'w')
for gdf_name, gdf in zip(['gdf_mel', 'gdf_syd'], [gdf_mel, gdf_syd]):
    for Y in ['apr20_log_ratio_mob', 'jan22_log_ratio_mob']:
        run_analysis(gdf, Y, gdf_name, predictor_names, nn=8, draw_maps=False, 
                     plot_moran=False, do_spatial_regression=True, 
                     compact_summary_file=f)
f.close()

#%%        

#%% Create raster plots, 
import geoplot as gplt
from pykrige import OrdinaryKriging

gdf_mel.geometry = gdf_mel.geometry.to_crs('EPSG:28348') # converting to a projected crs

resolution = 10000  # cell size in meters
gridx = np.arange(gdf_mel.bounds.minx.min(), gdf_mel.bounds.maxx.max(), resolution)
gridy = np.arange(gdf_mel.bounds.miny.min(), gdf_mel.bounds.maxy.max(), resolution)

krig = OrdinaryKriging(x=gdf_mel.centroid.x, y=gdf_mel.centroid.y, z=gdf_mel.apr20_log_ratio_mob, variogram_model="spherical")
z, ss = krig.execute("grid", gridx, gridy)
plt.imshow(z);

#%%
f, axs = plt.subplots(1, 2, figsize = (8, 3.5), constrained_layout = True)
gdf_mel.plot(ax = axs[0], column = 'apr20base_count', legend = True, scheme = 'quantiles', missing_kwds={
        "color": "forestgreen",
        "label": "NA"}, legend_kwds={"loc": "upper left", 'ncols': 1, "bbox_to_anchor": (1, 1)})
gdf_mel_cbd.boundary.plot(ax = axs[0], )

gdf_mel.plot(ax = axs[1], column = 'jan22base_count', legend = True, scheme = 'quantiles', missing_kwds={
        "color": "forestgreen",
        "label": "NA"}, legend_kwds={"loc": "upper left", 'ncols': 1, "bbox_to_anchor": (1, 1)})

#%% Plot mobility counts with distance to CBD
f, axs = plt.subplots(4, 2, figsize = (8, 8), constrained_layout = True, sharey = True)
plt.suptitle('MObility counts (adjusted for coverage), 1000 random samples')
gdf_mel.sample(1000).plot(ax = axs[0, 0], kind = 'line', x = 'distance_to_CBD', y = 'apr20base_v', logy=True, lw = 0, marker = 'o', ms = 2, alpha = 0.1, title = 'Melbourne')
gdf_mel.sample(1000).plot(ax = axs[1, 0], kind = 'line', x = 'distance_to_CBD', y = 'jan22base_v', logy=True, lw = 0, marker = 'o', ms = 2, alpha = 0.1, title = 'Melbourne')

#f, axs = plt.subplots(1, 2, figsize = (8, 3.5), constrained_layout = True)
gdf_syd.sample(1000).plot(ax = axs[0, 1], kind = 'line', x = 'distance_to_CBD', y = 'apr20base_v', logy=True, lw = 0, marker = 'o', ms = 2, alpha = 0.1, title = 'Sydney')
gdf_syd.sample(1000).plot(ax = axs[1, 1], kind = 'line', x = 'distance_to_CBD', y = 'jan22base_v', logy=True, lw = 0, marker = 'o', ms = 2, alpha = 0.1, title = 'Sydney')



gdf_mel.sample(1000).plot(ax = axs[2, 0], kind = 'line', x = 'distance_to_CBD', y = 'apr20test_v', logy=True, lw = 0, marker = 'o', ms = 2, alpha = 0.1, title = 'Melbourne')
gdf_mel.sample(1000).plot(ax = axs[3, 0], kind = 'line', x = 'distance_to_CBD', y = 'jan22test_v', logy=True, lw = 0, marker = 'o', ms = 2, alpha = 0.1, title = 'Melbourne')


#f, axs = plt.subplots(1, 2, figsize = (8, 3.5), constrained_layout = True)
gdf_syd.sample(1000).plot(ax = axs[2, 1], kind = 'line', x = 'distance_to_CBD', y = 'apr20test_v', logy=True, lw = 0, marker = 'o', ms = 2, alpha = 0.1, title = 'Sydney')
gdf_syd.sample(1000).plot(ax = axs[3, 1], kind = 'line', x = 'distance_to_CBD', y = 'jan22test_v', logy=True, lw = 0, marker = 'o', ms = 2, alpha = 0.1, title = 'Sydney')

#%%
import shapely

def create_grid(gdf, nx = 10, crs = 'EPSG:28348'):
    xmin, ymin, xmax, ymax = gdf.geometry.to_crs(crs).total_bounds
    
    xs = np.linspace(xmin, xmax, nx)
    cell_dim = xs[1]-xs[0]
    ys = np.arange(ymin, ymax+cell_dim, cell_dim)
    
    cells = []
    for x, xnext in zip(xs[0:-1], xs[1:]):
        for y, ynext in zip(ys[0:-1], ys[1:]):
            polygon = shapely.geometry.box(x, y, xnext, ynext)
    
            cells.append(polygon)
    
    gdf_cells = gpd.GeoDataFrame(cells, columns=['geometry'], crs = crs)
    return gdf_cells

def grid_SA1_map(gdf_grid, gdf, columns):
    gdf_grid_value = gdf_grid.copy()
    for counter, row in gdf_grid.iterrows():
        inter = gdf.geometry.intersection(row.geometry)
        if np.all(inter.geometry.is_empty):
            for column in columns:
                gdf_grid_value.loc[counter, column] = np.nan
        else:
            in_grid = gdf[~inter.geometry.is_empty]
            area_prop = inter[~inter.geometry.is_empty].area/in_grid.area
            for column in columns:
                gdf_grid_value.loc[counter, column] = np.sum(area_prop*in_grid[column])
    return gdf_grid_value


#%%
gdf_grid_mel = create_grid(gdf_mel, 40)
gdf_grid_mel = grid_SA1_map(gdf_grid_mel, gdf_mel, ['apr20base_v', 'apr20test_v', 'jan22base_v', 'jan22test_v'])
gdf_grid_mel['apr20_log_ratio_mob'] = np.log(gdf_grid_mel['apr20test_v']/gdf_grid_mel['apr20base_v'])
gdf_grid_mel['jan22_log_ratio_mob'] = np.log(gdf_grid_mel['jan22test_v']/gdf_grid_mel['jan22base_v'])

gdf_grid_syd = create_grid(gdf_syd, 40)
gdf_grid_syd = grid_SA1_map(gdf_grid_syd, gdf_syd, ['apr20base_v', 'apr20test_v', 'jan22base_v', 'jan22test_v'])
gdf_grid_syd['apr20_log_ratio_mob'] = np.log(gdf_grid_syd['apr20test_v']/gdf_grid_syd['apr20base_v'])
gdf_grid_syd['jan22_log_ratio_mob'] = np.log(gdf_grid_syd['jan22test_v']/gdf_grid_syd['jan22base_v'])
#%%
f, axs = plt.subplots(2, 2, figsize = (8, 8), constrained_layout = True)
gdf_mel.plot(ax = axs[0, 0], column = 'apr20_log_ratio_mob', legend = True, scheme = 'quantiles',
             linewidth = 0.1,
             missing_kwds={
        "color": "forestgreen",
        "label": "NA"}, legend_kwds={"loc": "upper left", 'ncols': 1, "bbox_to_anchor": (1, 1)})
gdf_grid_mel.boundary.plot(ax = axs[0, 0], lw = 0.01, color = 'k') 
gdf_grid_mel.plot(ax = axs[0, 1], column = 'apr20_log_ratio_mob', legend = True, scheme = 'quantiles', 
        legend_kwds={"loc": "upper left", 'ncols': 1, "bbox_to_anchor": (1, 1)}) 
gdf_grid_mel.boundary.plot(ax = axs[0, 1], lw = .01, color = 'k')

gdf_mel.plot(ax = axs[1, 0], column = 'jan22_log_ratio_mob', legend = True, scheme = 'quantiles', 
             linewidth = 0.1, 
             missing_kwds={
        "color": "forestgreen",
        "label": "NA"}, legend_kwds={"loc": "upper left", 'ncols': 1, "bbox_to_anchor": (1, 1)})
gdf_grid_mel.boundary.plot(ax = axs[1, 0], lw = 0.01, color = 'k') 
gdf_grid_mel.plot(ax = axs[1, 1], column = 'jan22_log_ratio_mob', legend = True, scheme = 'quantiles', 
        legend_kwds={"loc": "upper left", 'ncols': 1, "bbox_to_anchor": (1, 1)}) 
gdf_grid_mel.boundary.plot(ax = axs[1, 1], lw = .01, color = 'k')

axs[0, 0].set(title = 'apr20_log_ratio_mob')
axs[0, 1].set(title = 'apr20_log_ratio_mob')
axs[1, 0].set(title = 'jan22_log_ratio_mob')
axs[1, 1].set(title = 'jan22_log_ratio_mob')
plt.suptitle('MEL')

#%%
# normalize color
vcenter = 0
f, axs = plt.subplots(1, 2, figsize = (8, 3.5), constrained_layout = True)

vmax = np.max(np.abs([gdf_grid_mel.apr20_log_ratio_mob.min(), gdf_grid_mel.apr20_log_ratio_mob.max()]))
vmin = -vmax
norm = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin = vmin, vmax = vmax)
cmap = 'RdBu'
cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
f.colorbar(cbar, ax=axs[0])
gdf_grid_mel.plot(ax = axs[0], column = 'apr20_log_ratio_mob', cmap = cmap,  
        legend_kwds={"loc": "upper left", 'ncols': 1, "bbox_to_anchor": (1, 1)}, 
        missing_kwds={"color": "forestgreen"}) 
#gdf_grid_mel.boundary.plot(ax = axs[0, 1], lw = .1, color = 'k')

vmax = np.max(np.abs([gdf_grid_mel.jan22_log_ratio_mob.min(), gdf_grid_mel.jan22_log_ratio_mob.max()]))
vmin = -vmax
norm = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin = vmin, vmax = vmax)
cmap = 'RdBu'
cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
f.colorbar(cbar, ax=axs[1])
gdf_grid_mel.plot(ax = axs[1], column = 'jan22_log_ratio_mob', cmap = cmap,  
        legend_kwds={"loc": "upper left", 'ncols': 1, "bbox_to_anchor": (1, 1)}, 
        missing_kwds={"color": "forestgreen"}) 
#gdf_grid_mel.boundary.plot(ax = axs[1, 1], lw = .1, color = 'k')

axs[0].set(title = 'apr20_log_ratio_mob')
axs[1].set(title = 'jan22_log_ratio_mob')
plt.suptitle('MEL')

#%%
gdf_syd.plot(ax = axs[1, 0], column = 'apr20_log_ratio_mob', legend = True, scheme = 'quantiles', missing_kwds={
        "color": "forestgreen",
        "label": "NA"}, legend_kwds={"loc": "upper left", 'ncols': 1, "bbox_to_anchor": (1, 1)})
gdf_grid_syd.boundary.plot(ax = axs[0], lw = 0.1, color = 'k') 
gdf_grid_syd.plot(ax = axs[1, 1], column = 'apr20_log_ratio_mob', legend = True, scheme = 'quantiles', 
        legend_kwds={"loc": "upper left", 'ncols': 1, "bbox_to_anchor": (1, 1)}) 
gdf_grid_syd.boundary.plot(ax = axs[1], lw = .1, color = 'k')
#%% Aggreagation to SA2 level
gdf_SA2 = gpd.read_file('1270055001_sa2_2016_aust_shape/SA2_2016_AUST.shp', columns = ['SA2_MAIN16', 'AREASQKM16', 'GCC_NAME16', 'geometry'])
gdf_SA2['SA2_MAIN16'] = gdf_SA2['SA2_MAIN16'].astype('int64')

gdf_mel_SA2 = gdf_SA2[gdf_SA2.GCC_NAME16 == 'Greater Melbourne']
gdf_mel['SA2_MAIN16'] = (gdf_mel.SA1_MAIN16/100.0).astype('int64')
df_mel_grouped = gdf_mel[['SA2_MAIN16', 'URP', 'apr20base_v', 'apr20test_v', 'jan22base_v', 'jan22test_v']].groupby('SA2_MAIN16')
list_mel_SA2 = []
for name, group in df_mel_grouped:
    dict_mel_SA2 = {'SA2_MAIN16': name, 
                    'apr20base_v': np.sum(group['apr20base_v'].values * group['URP'].values), 
                    'apr20test_v': np.sum(group['apr20test_v'].values * group['URP'].values), 
                    'jan22base_v': np.sum(group['jan22base_v'].values * group['URP'].values), 
                    'jan22test_v': np.sum(group['jan22test_v'].values * group['URP'].values)
                    }
    list_mel_SA2.append(dict_mel_SA2)
    #df_mel_SA2.loc[name, ['apr20base_v']] = np.average(group['apr20base_v'], weights=group['URP'])

df_mel_SA2 = pd.DataFrame(list_mel_SA2)
gdf_mel_SA2 = gdf_mel_SA2.merge(right = df_mel_SA2, how = 'left', left_on = 'SA2_MAIN16', right_on = 'SA2_MAIN16')

gdf_mel_SA2['apr20_log_ratio_mob'] = np.log(gdf_mel_SA2.apr20test_v/gdf_mel_SA2.apr20base_v)
gdf_mel_SA2['jan22_log_ratio_mob'] = np.log(gdf_mel_SA2.jan22test_v/gdf_mel_SA2.jan22base_v)


gdf_syd_SA2 = gdf_SA2[gdf_SA2.GCC_NAME16 == 'Greater Sydney']
gdf_syd['SA2_MAIN16'] = (gdf_syd.SA1_MAIN16/100.0).astype('int64')
df_syd_grouped = gdf_syd[['SA2_MAIN16', 'URP', 'apr20base_v', 'apr20test_v', 'jan22base_v', 'jan22test_v']].groupby('SA2_MAIN16')
list_syd_SA2 = []
for name, group in df_syd_grouped:
    dict_syd_SA2 = {'SA2_MAIN16': name, 
                    'apr20base_v': np.sum(group['apr20base_v'].values * group['URP'].values), 
                    'apr20test_v': np.sum(group['apr20test_v'].values * group['URP'].values), 
                    'jan22base_v': np.sum(group['jan22base_v'].values * group['URP'].values), 
                    'jan22test_v': np.sum(group['jan22test_v'].values * group['URP'].values)
                    }
    list_syd_SA2.append(dict_syd_SA2)
    #df_syd_SA2.loc[name, ['apr20base_v']] = np.average(group['apr20base_v'], weights=group['URP'])

df_syd_SA2 = pd.DataFrame(list_syd_SA2)
gdf_syd_SA2 = gdf_syd_SA2.merge(right = df_syd_SA2, how = 'left', left_on = 'SA2_MAIN16', right_on = 'SA2_MAIN16')

gdf_syd_SA2['apr20_log_ratio_mob'] = np.log(gdf_syd_SA2.apr20test_v/gdf_syd_SA2.apr20base_v)
gdf_syd_SA2['jan22_log_ratio_mob'] = np.log(gdf_syd_SA2.jan22test_v/gdf_syd_SA2.jan22base_v)

#%% SA2 Mobility plot MEL and SYD

gdf_states = gpd.read_file('STE_2021_AUST_GDA2020.shp') # For coastal outline

f, axs = plt.subplots(2, 2, figsize = (8, 8), constrained_layout = True)
cmap = 'PuOr_r'

vmax = np.max(np.abs([gdf_mel_SA2.apr20_log_ratio_mob.min(), 
                      gdf_mel_SA2.apr20_log_ratio_mob.max(), 
                      gdf_syd_SA2.apr20_log_ratio_mob.min(), 
                      gdf_syd_SA2.apr20_log_ratio_mob.max(), 
                      gdf_mel_SA2.jan22_log_ratio_mob.min(), 
                      gdf_mel_SA2.jan22_log_ratio_mob.max(), 
                      gdf_syd_SA2.jan22_log_ratio_mob.min(), 
                      gdf_syd_SA2.jan22_log_ratio_mob.max(),                       
                      ]))

vmax = 1.5
#vmax = np.max(np.abs([gdf_mel_SA2.apr20_log_ratio_mob.min(), gdf_mel_SA2.apr20_log_ratio_mob.max()]))
vmin = -vmax

#Outliers set to vmax and vmin
gdf_mel_SA2.apr20_log_ratio_mob = gdf_mel_SA2.apr20_log_ratio_mob.mask((gdf_mel_SA2.apr20_log_ratio_mob > vmax), vmax)
gdf_mel_SA2.apr20_log_ratio_mob = gdf_mel_SA2.apr20_log_ratio_mob.mask((gdf_mel_SA2.apr20_log_ratio_mob < vmin), vmin)
gdf_mel_SA2.jan22_log_ratio_mob = gdf_mel_SA2.jan22_log_ratio_mob.mask((gdf_mel_SA2.jan22_log_ratio_mob > vmax), vmax)
gdf_mel_SA2.jan22_log_ratio_mob = gdf_mel_SA2.jan22_log_ratio_mob.mask((gdf_mel_SA2.jan22_log_ratio_mob < vmin), vmin)

gdf_syd_SA2.apr20_log_ratio_mob = gdf_syd_SA2.apr20_log_ratio_mob.mask((gdf_syd_SA2.apr20_log_ratio_mob > vmax), vmax)
gdf_syd_SA2.apr20_log_ratio_mob = gdf_syd_SA2.apr20_log_ratio_mob.mask((gdf_syd_SA2.apr20_log_ratio_mob < vmin), vmin)
gdf_syd_SA2.jan22_log_ratio_mob = gdf_syd_SA2.jan22_log_ratio_mob.mask((gdf_syd_SA2.jan22_log_ratio_mob > vmax), vmax)
gdf_syd_SA2.jan22_log_ratio_mob = gdf_syd_SA2.jan22_log_ratio_mob.mask((gdf_syd_SA2.jan22_log_ratio_mob < vmin), vmin)


gdf_mel_SA2.plot(ax = axs[0, 0], column = 'apr20_log_ratio_mob',
                 cmap = cmap, legend = True, vmax = vmax, vmin = vmin,
                 edgecolor = 'k', linewidth = 0.1,
                 missing_kwds={"color": "white", 'edgecolor': 'lightgreen', 'hatch': '///'}, 
                 legend_kwds={"label": f'{lm_av}'})

#vmax = np.max(np.abs([gdf_mel_SA2.jan22_log_ratio_mob.min(), gdf_mel_SA2.jan22_log_ratio_mob.max()]))
#vmin = -vmax
gdf_mel_SA2.plot(ax = axs[1, 0], column = 'jan22_log_ratio_mob', 
                 cmap = cmap, legend = True, vmax = vmax, vmin = vmin,
                 edgecolor = 'k', linewidth = 0.1,
                 missing_kwds={"color": "white", 'edgecolor': 'palegreen', 'hatch': '///'},
                 legend_kwds={"label": f'{lm_av}'})
#gdf_mel_SA2.boundary.plot(ax = axs[1, 1], lw = .1, color = 'k')

axs[0, 0].set(title = f'Melbourne - April 2020')
axs[1, 0].set(title = f'Melbourne - Jan 2022')

#vmax = np.max(np.abs([gdf_syd_SA2.apr20_log_ratio_mob.min(), gdf_syd_SA2.apr20_log_ratio_mob.max()]))
#vmin = -vmax

gdf_syd_SA2.plot(ax = axs[0, 1], column = 'apr20_log_ratio_mob', 
                 cmap = cmap, legend = True, vmax = vmax, vmin = vmin,
                 edgecolor = 'k', linewidth = 0.1,
                 missing_kwds={"color": "white", 'edgecolor': 'lightgreen', 'hatch': '/////'},
                 legend_kwds={"label": f'{lm_av}'})
#gdf_syd_SA2.boundary.plot(ax = axs[0, 1], lw = .1, color = 'k')

#vmax = np.max(np.abs([gdf_syd_SA2.jan22_log_ratio_mob.min(), gdf_syd_SA2.jan22_log_ratio_mob.max()]))
#vmin = -vmax
gdf_syd_SA2.plot(ax = axs[1, 1], column = 'jan22_log_ratio_mob', 
        cmap = cmap, legend = True, vmax = vmax, vmin = vmin,
        edgecolor = 'k', linewidth = 0.1,
        missing_kwds={"color": "white", 'edgecolor': 'lightgreen', 'hatch': '/////'},
        legend_kwds={"label": f'{lm_av}'}) 
#gdf_syd_SA2.boundary.plot(ax = axs[1, 1], lw = .1, color = 'k')

axs[0, 1].set(title = f'Sydney - April 2020')
axs[1, 1].set(title = f'Sydney - Jan 2022')

for ax in axs.reshape(-1):
    xlims, ylims = ax.get_xlim(), ax.get_ylim()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    gdf_states.plot(ax = ax, color = 'lightgrey', zorder = -2)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_facecolor('lightskyblue')

# vmax = np.max(np.abs([gdf_syd_SA2.jan22_log_ratio_mob.min(), gdf_syd_SA2.jan22_log_ratio_mob.max()]))
# vmin = -vmax

# gdf_syd_SA2.plot(column = 'jan22_log_ratio_mob', cmap = cmap, legend = True, vmax = vmax, vmin = vmin,
#         edgecolor = 'k', linewidth = 0.1,
#         missing_kwds={"color": "white", 'edgecolor': 'green', 'hatch': '///'}) 


#%% Mobility vs distance
gdf_mel['log_apr20test_v'] = np.log(gdf_mel.apr20test_v)
gdf_mel['log_apr20base_v'] = np.log(gdf_mel.apr20base_v)
gdf_mel['log_jan22test_v'] = np.log(gdf_mel.jan22test_v)
gdf_mel['log_jan22base_v'] = np.log(gdf_mel.jan22base_v)

gdf_syd['log_apr20test_v'] = np.log(gdf_syd.apr20test_v)
gdf_syd['log_apr20base_v'] = np.log(gdf_syd.apr20base_v)
gdf_syd['log_jan22test_v'] = np.log(gdf_syd.jan22test_v)
gdf_syd['log_jan22base_v'] = np.log(gdf_syd.jan22base_v)

cols = ['log_apr20test_v', 'log_apr20base_v', 'log_jan22test_v', 'log_jan22base_v', 'distance_to_CBD']

pd.plotting.scatter_matrix(gdf_mel[cols], s = 0.5)
#%%
#wq = weights.contiguity.Queen.from_dataframe(gdf_mel) # island problem
# nn = 8
# wk8 = weights.distance.KNN.from_dataframe(gdf_mel, k=nn)
# wk8.transform = 'R'

# gdf_mel['apr20_log_ratio_mob_lag'] = weights.spatial_lag.lag_spatial(wk8, gdf_mel['apr20_log_ratio_mob'])
# gdf_mel['apr20_log_ratio_mob_lag_standard'] = weights.spatial_lag.lag_spatial(wk8, gdf_mel['apr20_log_ratio_mob_standard'])

# #%% Fit a non-spatial OLS to the data
# dependent_name = 'apr20_log_ratio_mob_standard'
# predictor_names = ['ER_score_standard', 'EO_score_standard', 'pop_density_standard', 'distance_to_CBD_standard']
# fit_non_spatial = spreg.OLS(
#                 y = gdf_mel[dependent_name].values, 
#                 x = gdf_mel[predictor_names].values, 
#                 name_y = 'apr20_log_ratio_mob_standard', 
#                 name_x = predictor_names, 
#                 name_ds = 'gdf_mel')
# #%% Visualize lag on map
# f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
# gdf_mel.plot(ax = ax[0], column = 'apr20_log_ratio_mob', cmap = 'RdBu_r', legend = True, scheme = 'quantiles')
# ax[0].set_title('apr20_log_ratio_mob')
# gdf_mel.plot(ax = ax[1], column = 'apr20_log_ratio_mob_lag', cmap = 'RdBu_r', legend = True, scheme = 'quantiles')
# ax[1].set_title('apr20_log_ratio_mob_lag')

# #%% VIsualize residuals on map and on qqplot
# gdf_mel
# f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
# gdf_mel.plot(ax = ax[0], column = 'apr20_log_ratio_mob', cmap = 'RdBu_r', legend = True, scheme = 'quantiles')
# ax[0].set_title('apr20_log_ratio_mob')
# gdf_mel.plot(ax = ax[1], column = fit_non_spatial.u[:, 0], cmap = 'RdBu_r', legend = True, scheme = 'quantiles')
# ax[1].set_title('apr20_log_ratio_mob_residuals')

# qqplot(data = fit_non_spatial.u[:, 0], line = '45', markerfacecolor='none', alpha = 0.5)
# #%% Moran plot and I
# gdf_mel['apr20_log_ratio_mob_lag_standard'] = weights.spatial_lag.lag_spatial(wk8, gdf_mel['apr20_log_ratio_mob_standard'])
# moran = esda.moran.Moran(gdf_mel["apr20_log_ratio_mob_standard"], wk8)
# plot_moran(moran)

# splot.esda.moran_scatterplot(moran, scatter_kwds={'alpha': 0.5, 'facecolor': 'none', 'edgecolor': 'lightgrey'})

# #%%
# fit_lag = spreg.ML_Lag(y = gdf_mel.apr20_log_ratio_mob_standard.values, 
#           x = gdf_mel[['ER_score_standard', 'EO_score_standard', 
#                        'pop_density_standard', 'distance_to_CBD_standard']].values, w = wk8,
#                           name_y = 'apr20_log_ratio_mob_standard', 
#                           name_x = ['ER_score_standard', 'EO_score_standard', 
#                                        'pop_density_standard', 'distance_to_CBD_standard'], 
#                           name_ds = 'gdf_mel')

# fit_error = spreg.ML_Error(y = gdf_mel.apr20_log_ratio_mob_standard.values, 
#           x = gdf_mel[['ER_score_standard', 'EO_score_standard', 
#                        'pop_density_standard', 'distance_to_CBD_standard']].values, w = wk8,
#                           name_y = 'apr20_log_ratio_mob_standard', 
#                           name_x = ['ER_score_standard', 'EO_score_standard', 
#                                        'pop_density_standard', 'distance_to_CBD_standard'], 
#                           name_ds = 'gdf_mel')

# #%%
# y, X = dmatrices('apr20_log_ratio_mob_standard ~ ER_score_standard + EO_score_standard + pop_density_standard + distance_to_CBD_standard', data = gdf_mel, return_type = 'dataframe')
# model = sm.OLS(endog = y, exog = X)
# results_mel_apr20_standard = model.fit()
# print(results_mel_apr20_standard.summary())


# dependent_name = 'apr20_log_ratio_mob_standard'
# predictor_names = ['ER_score_standard', 'EO_score_standard', 'pop_density_standard', 'distance_to_CBD_standard']
# fit_non_spatial = spreg.OLS(
#                 y = gdf_mel[dependent_name].values, 
#                 x = gdf_mel[predictor_names].values, 
#                 name_y = 'apr20_log_ratio_mob_standard', 
#                 name_x = predictor_names, 
#                 name_ds = 'gdf_mel')
# print(fit_non_spatial.summary)

# #%%
# y, X = dmatrices('jan22_log_ratio_mob_standard ~ ER_score_standard + EO_score_standard + pop_density_standard + distance_to_CBD_standard', data = gdf_mel, return_type = 'dataframe')
# model = sm.OLS(endog = y, exog = X)
# results_mel_apr20_standard = model.fit()
# print(results_mel_apr20_standard.summary())


# dependent_name = 'jan22_log_ratio_mob_standard'
# predictor_names = ['ER_score_standard', 'EO_score_standard', 'pop_density_standard', 'distance_to_CBD_standard']
# fit_non_spatial = spreg.OLS(
#                 y = gdf_mel[dependent_name].values, 
#                 x = gdf_mel[predictor_names].values, 
#                 name_y = 'apr20_log_ratio_mob_standard', 
#                 name_x = predictor_names, 
#                 name_ds = 'gdf_mel')
# print(fit_non_spatial.summary)
