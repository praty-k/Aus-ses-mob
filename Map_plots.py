# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:19:11 2023

@author: pkollepara
"""

import geopandas as gpd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%
plt.style.use('seaborn-v0_8-colorblind')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 12
plt.rc('axes', titlesize=plt.rcParams['font.size'])
cmap = (sns.color_palette('rocket', as_cmap = True))

#%%
df_mel = pd.read_csv('SA1_SEIFA_deciles_MEL.csv', index_col = 0)
df_syd = pd.read_csv('SA1_SEIFA_deciles_SYD.csv', index_col = 0)
#%%
gdf = gpd.read_file('SA1 shape files/SA1_2016_AUST.shp')
gdf.SA1_MAIN16 = pd.to_numeric(gdf.SA1_MAIN16)

#%%
gdf_vic = gdf[gdf.STE_NAME16 == 'Victoria']
gdf_vic = pd.merge(right = df_mel, left = gdf_vic, how = 'left', right_on = 'SA1', left_on = 'SA1_MAIN16')
gdf_vic.EO_local_decile = gdf_vic.EO_local_decile.astype('Int64')
gdf_vic.ER_local_decile = gdf_vic.ER_local_decile.astype('Int64')
gdf_vic = gdf_vic[['SA1_MAIN16', 'ER_local_decile', 'EO_local_decile', 'geometry']]
gdf_vic = gdf_vic.set_geometry('geometry')
mel_boundary = gdf[gdf.GCC_NAME16=='Greater Melbourne'].geometry.unary_union

#%%
f, axs = plt.subplots(1, 2, figsize = (8, 3.5), constrained_layout = True)
gdf_vic.plot(ax = axs[0], column = 'ER_local_decile', categorical = True, cmap = cmap, legend = True, missing_kwds={
        "color": "forestgreen",
        "label": "NA"}, legend_kwds={"loc": "upper left", 'ncols': 1, "bbox_to_anchor": (1, 1)})
gpd.GeoSeries(mel_boundary).boundary.plot(ax = axs[0], edgecolor = 'black', linewidth = .2)

gdf_vic.plot(ax = axs[1], column = 'EO_local_decile', categorical = True, cmap = cmap, legend = True, missing_kwds={
        "color": "forestgreen",
        "label": "NA"}, legend_kwds={"loc": "upper left", 'ncols': 1, "bbox_to_anchor": (1, 1)})
gpd.GeoSeries(mel_boundary).boundary.plot(ax = axs[1], edgecolor = 'black', linewidth = .2)

axs[0].set_title('Local deciles of ER')
axs[0].text(0.02, 0.98, '[A]', transform=axs[0].transAxes, fontsize=12, weight = 'bold',  va='top') 
axs[1].set_title('Local deciles of EO')
axs[1].text(0.02, 0.98, '[B]', transform=axs[1].transAxes, fontsize=12, weight = 'bold',  va='top') 
#handles, legend_labels = ax.get_legend_handles_labels()
#ax.legend(handles, legend_labels, loc = 'outside upper right', ncols = 6, fancybox = False)

for ax in axs:
    ax.set_xlim(144.25, 145.9)
    ax.set_ylim(-38.6, -37.1)
    #ax.set_facecolor('#0066CC')
    ax.set_facecolor('skyblue')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #ax.set_axis_off()

plt.savefig('MEL_SA1_plot.png', dpi = 600)
#%%
gdf_nsw = gdf[gdf.STE_NAME16 == 'New South Wales']
gdf_nsw = pd.merge(right = df_syd, left = gdf_nsw, how = 'left', right_on = 'SA1', left_on = 'SA1_MAIN16')
gdf_nsw.EO_local_decile = gdf_nsw.EO_local_decile.astype('Int64')
gdf_nsw.ER_local_decile = gdf_nsw.ER_local_decile.astype('Int64')
gdf_nsw = gdf_nsw[['SA1_MAIN16', 'ER_local_decile', 'EO_local_decile', 'geometry']]
gdf_nsw = gdf_nsw.set_geometry('geometry')
syd_boundary = gdf[gdf.GCC_NAME16=='Greater Sydney'].geometry.unary_union


#%%
f, axs = plt.subplots(1, 2, figsize = (8, 3.5), constrained_layout = True)
gdf_nsw.plot(ax = axs[0], column = 'ER_local_decile', categorical = True, cmap = cmap, legend = True, missing_kwds={
        "color": "forestgreen",
        "label": "NA"}, legend_kwds={"loc": "upper left", 'ncols': 1, "bbox_to_anchor": (1, 1)})
gpd.GeoSeries(syd_boundary).boundary.plot(ax = axs[0], edgecolor = 'black', linewidth = .2)
gdf_nsw.plot(ax = axs[1], column = 'EO_local_decile', categorical = True, cmap = cmap, legend = True, missing_kwds={
        "color": "forestgreen",
        "label": "NA"}, legend_kwds={"loc": "upper left", 'ncols': 1, "bbox_to_anchor": (1, 1)})
gpd.GeoSeries(syd_boundary).boundary.plot(ax = axs[1], edgecolor = 'black', linewidth = .2)
axs[0].set_title('Local deciles of ER')
axs[0].text(0.02, 0.98, '[A]', transform=axs[0].transAxes, fontsize=12, weight = 'bold',  va='top') 
axs[1].set_title('Local deciles of EO')
axs[1].text(0.02, 0.98, '[B]', transform=axs[1].transAxes, fontsize=12, weight = 'bold',  va='top') 
#handles, legend_labels = ax.get_legend_handles_labels()
#ax.legend(handles, legend_labels, loc = 'outside upper right', ncols = 6, fancybox = False)

for ax in axs:
    ax.set_xlim(149.9, 151.7)
    ax.set_ylim(-34.35, -32.9)
    ax.set_facecolor('skyblue')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.savefig('SYD_SA1_plot.png', dpi = 600)