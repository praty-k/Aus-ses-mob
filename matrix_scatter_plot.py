# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:26:33 2024

@author: PKollepara
"""

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd

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
#%%
gdf_mel = gpd.read_file('MEL_processed_geodataframe_geopackage.gpkg')
gdf_syd = gpd.read_file('SYD_processed_geodataframe_geopackage.gpkg')

#%%
predictors_apr = ['apr20_log_ratio_mob', 'ER_score', 'EO_score', 'distance_to_CBD']
predictors_apr_label = [f'{lm}', 'ER', 'EO', 'Distance to CBD']


df = pd.DataFrame(gdf_mel[predictors_apr].values, columns = predictors_apr_label)
scatter_matrix(df, diagonal = 'kde', s = 0.5, figsize = (8, 8))
plt.tight_layout()
plt.savefig('MEL_apr20_plot_matrix_a', dpi = 600)
plt.close()
#%%
predictors_apr = ['apr20_log_ratio_mob', 'ER_score', 'EO_score', 'distance_to_CBD']
predictors_apr_label = [f'{lm}', 'ER', 'EO', 'Distance to CBD']


df = pd.DataFrame(gdf_syd[predictors_apr].values, columns = predictors_apr_label)
scatter_matrix(df, diagonal = 'kde', s = 0.5, figsize = (8, 8))
plt.tight_layout()
plt.savefig('SYD_apr20_plot_matrix_a', dpi = 600)
plt.close()
#%%
predictors_jan = ['jan22_log_ratio_mob', 'ER_score', 'EO_score', 'distance_to_CBD']
predictors_jan_label = [f'{lm}', 'ER', 'EO', 'Distance to CBD']


df = pd.DataFrame(gdf_mel[predictors_apr].values, columns = predictors_apr_label)
scatter_matrix(df, diagonal = 'kde', s = 0.5, figsize = (8, 8))
plt.tight_layout()
plt.savefig('MEL_jan22_plot_matrix_a', dpi = 600)
plt.close()
#%%
predictors_jan = ['jan22_log_ratio_mob', 'ER_score', 'EO_score', 'distance_to_CBD']
predictors_jan_label = [f'{lm}', 'ER', 'EO', 'Distance to CBD']


df = pd.DataFrame(gdf_syd[predictors_apr].values, columns = predictors_apr_label)
scatter_matrix(df, diagonal = 'kde', s = 0.5, figsize = (8, 8))
plt.tight_layout()
plt.savefig('SYD_jan22_plot_matrix_a', dpi = 600)
plt.close()