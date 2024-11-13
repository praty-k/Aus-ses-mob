 # -*- coding: utf-8 -*-
"""
Created on Thu May  2 21:53:37 2024

@author: pkollepara
"""

import numpy as np
import statsmodels.api as sm
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from patsy import dmatrices

import geopandas as gpd

#%% Importing functions
from dataframe_prep_w_cov_all_SA1s import dataframe_prep
from draw_lplot import draw_lplot
from draw_jplot import draw_scatter, draw_hex
from ER_EO_heatmap import compute_heatmap, draw_heatmap, draw_heatmap_zero_centre 
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
retain_proportion = 99.7/100 #68, 95, 99.7
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

#%% Standardise predictor variables
df_mel['ER_score_standard'] = (df_mel['ER_score'] - df_mel['ER_score'].mean())/df_mel['ER_score'].std()
df_mel['EO_score_standard'] = (df_mel['EO_score'] - df_mel['EO_score'].mean())/df_mel['EO_score'].std()
df_mel['pop_density_standard'] = (df_mel.pop_density - df_mel.pop_density.mean())/df_mel.pop_density.std()
df_mel['distance_to_CBD_standard'] = (df_mel.distance_to_CBD - df_mel.distance_to_CBD.mean())/df_mel.distance_to_CBD.std()

df_syd['ER_score_standard'] = (df_syd['ER_score'] - df_syd['ER_score'].mean())/df_syd['ER_score'].std()
df_syd['EO_score_standard'] = (df_syd['EO_score'] - df_syd['EO_score'].mean())/df_syd['EO_score'].std()
df_syd['pop_density_standard'] = (df_syd.pop_density - df_syd.pop_density.mean())/df_syd.pop_density.std()
df_syd['distance_to_CBD_standard'] = (df_syd.distance_to_CBD - df_syd.distance_to_CBD.mean())/df_syd.distance_to_CBD.std()
#%% Run multiple regressions
print('Multiple regression all variables \n')
y, X = dmatrices('apr20_log_ratio_mob ~ ER_score + EO_score + pop_density + distance_to_CBD', data = df_mel, return_type = 'dataframe')
model = sm.OLS(endog = y, exog = X)
results_mel_apr20 = model.fit()
print(results_mel_apr20.summary())
y, X = dmatrices('apr20_log_ratio_mob ~ ER_score + EO_score + pop_density + distance_to_CBD', data = df_syd, return_type = 'dataframe')
model = sm.OLS(endog = y, exog = X)
results_syd_apr20 = model.fit()
print(results_syd_apr20.summary())

y, X = dmatrices('jan22_log_ratio_mob ~ ER_score + EO_score + pop_density + distance_to_CBD', data = df_mel, return_type = 'dataframe')
model = sm.OLS(endog = y, exog = X)
results_mel_jan22 = model.fit()
print(results_mel_jan22.summary())
y, X = dmatrices('jan22_log_ratio_mob ~ ER_score + EO_score + pop_density + distance_to_CBD', data = df_syd, return_type = 'dataframe')
model = sm.OLS(endog = y, exog = X)
results_syd_jan22 = model.fit()
print(results_syd_jan22.summary())

#%% Run multiple regressions standardized
print('Multiple regression all standardized variables \n')
y, X = dmatrices('apr20_log_ratio_mob ~ ER_score_standard + EO_score_standard + pop_density_standard + distance_to_CBD_standard', 
                 data = df_mel, return_type = 'dataframe')
model = sm.OLS(endog = y, exog = X)
results_mel_apr20_standard = model.fit()

print(results_mel_apr20_standard.summary())

y, X = dmatrices('apr20_log_ratio_mob ~ ER_score_standard + EO_score_standard + pop_density_standard + distance_to_CBD_standard', data = df_syd, return_type = 'dataframe')
model = sm.OLS(endog = y, exog = X)
results_syd_apr20_standard = model.fit()
print(results_syd_apr20_standard.summary())

y, X = dmatrices('jan22_log_ratio_mob ~ ER_score_standard + EO_score_standard + pop_density_standard + distance_to_CBD_standard', data = df_mel, return_type = 'dataframe')
model = sm.OLS(endog = y, exog = X)
results_mel_jan22_standard = model.fit()
print(results_mel_jan22_standard.summary())


y, X = dmatrices('jan22_log_ratio_mob ~ ER_score_standard + EO_score_standard + pop_density_standard + distance_to_CBD_standard', data = df_syd, return_type = 'dataframe')
model = sm.OLS(endog = y, exog = X)
results_syd_jan22_standard = model.fit()
print(results_syd_jan22_standard.summary())


#%%
y, X = dmatrices('apr20_log_ratio_mob ~ EO_score', data = df_mel, return_type = 'dataframe')
model = sm.OLS(endog = y, exog = X)
results_mel_apr20_EO = model.fit()

#%%
print('Step wise regression for standardized variables\n')
alpha_include = 0.15
predictors = ['ER_score_standard', 'EO_score_standard', 'pop_density_standard', 'distance_to_CBD_standard']
pvalues = []
for counter, predictor in enumerate(predictors):
    y, X = dmatrices(f'apr20_log_ratio_mob ~ {predictor}', 
                     data = df_mel, return_type = 'dataframe')
    model = sm.OLS(endog = y, exog = X)
    results_mel_apr20_ = model.fit()
    print(f'{counter+1}. Regression with {predictor}: ')
    print(f"R^2: {results_mel_apr20_.rsquared}, pvalue: {results_mel_apr20_.pvalues.iloc[1]}, coeff: {results_mel_apr20_.params.iloc[1]}, SSR: {results_mel_apr20_.ssr} \n")
    pvalues.append(results_mel_apr20_.pvalues.iloc[-1])
if np.min(pvalues) <= alpha_include:
    predictor_0 = predictors[np.argmin(pvalues)]
    predictors.remove(predictor_0)
    print(f'{predictor_0} selected'+'\n')
    pvalues = []
    for counter, predictor in enumerate(predictors):
        y, X = dmatrices(f'apr20_log_ratio_mob ~ {predictor_0} + {predictor}', 
                         data = df_mel, return_type = 'dataframe')
        model = sm.OLS(endog = y, exog = X)
        results_mel_apr20_ = model.fit()
        print(f'{counter+1}. Regression with {predictor_0} + {predictor}: ')
        print(f"R^2: {results_mel_apr20_.rsquared}, SSR: {results_mel_apr20_.ssr}")
        print("pvalues: ", results_mel_apr20_.pvalues, "\n") 
        #print("coeff: "results_mel_apr20_.params.iloc[1]})
        pvalues.append(results_mel_apr20_.pvalues.iloc[-1])
    if np.min(pvalues) <= alpha_include:
        predictor_1 = predictors[np.argmin(pvalues)]
        predictors.remove(predictor_1)
        print(f'{predictor_1} selected'+'\n')
        pvalues = []
        for counter, predictor in enumerate(predictors):
            y, X = dmatrices(f'apr20_log_ratio_mob ~ {predictor_0} + {predictor_1} + {predictor}', 
                             data = df_mel, return_type = 'dataframe')
            model = sm.OLS(endog = y, exog = X)
            results_mel_apr20_ = model.fit()
            print(f'{counter+1}. Regression with {predictor_0} + {predictor_1} + {predictor}: ')
            print(f"R^2: {results_mel_apr20_.rsquared}, SSR: {results_mel_apr20_.ssr}")
            print("pvalues: ", results_mel_apr20_.pvalues, "\n") 
            #print("coeff: "results_mel_apr20_.params.iloc[1]})
            pvalues.append(results_mel_apr20_.pvalues.iloc[-1])
        
#%%
predictor = 'ER_score_standard + EO_score_standard'
y, X = dmatrices(f'apr20_log_ratio_mob ~ {predictor}', 
                 data = df_mel, return_type = 'dataframe')
model = sm.OLS(endog = y, exog = X)
results_mel_apr20_ = model.fit()
print(f'Regression with {predictor}: ')
print(f"R^2: {results_mel_apr20_.rsquared}, f pvalue: {results_mel_apr20_.f_pvalue}, coeff: {results_mel_apr20_.params.iloc[1]}, SSR: {results_mel_apr20_.ssr}")