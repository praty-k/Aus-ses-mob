# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 12:50:29 2024

@author: PKollepara
"""


import geopandas as gpd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
plt.style.use('seaborn-v0_8-colorblind')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 12
plt.rc('axes', titlesize=plt.rcParams['font.size'])
cmap = (sns.color_palette('rocket', as_cmap=True))

# %%
df_mel = pd.read_csv('SA1_SEIFA_deciles_MEL.csv', index_col=0)
df_syd = pd.read_csv('SA1_SEIFA_deciles_SYD.csv', index_col=0)
# %%
gdf = gpd.read_file('SA1 shape files/SA1_2016_AUST.shp')
gdf.SA1_MAIN16 = pd.to_numeric(gdf.SA1_MAIN16)

# %%
gdf_vic = gdf[gdf.STE_NAME16 == 'Victoria']
gdf_vic = gdf_vic.set_geometry('geometry')

# %%
gdf_mel = gdf[gdf.GCC_NAME16 == 'Greater Melbourne']
gdf_mel = pd.merge(right=df_mel, left=gdf_mel, how='left',
                   right_on='SA1', left_on='SA1_MAIN16')
gdf_mel.EO_local_decile = gdf_mel.EO_local_decile.astype('Int64')
gdf_mel.ER_local_decile = gdf_mel.ER_local_decile.astype('Int64')
gdf_mel = gdf_mel[['SA1_MAIN16', 'ER_local_decile',
                   'EO_local_decile', 'geometry']]
gdf_mel = gdf_mel.set_geometry('geometry')
# mel_boundary = gdf[gdf.GCC_NAME16=='Greater Melbourne'].geometry.unary_union

# %%
f, axs = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
gdf_mel.plot(ax=axs[0], column='ER_local_decile', categorical=True, cmap=cmap, legend=True,
             missing_kwds={"color": "white", 'edgecolor': 'lightgreen',
                           'hatch': '/////', 'linewidth': 0.01},
             legend_kwds={"loc": "upper left", 'ncols': 1, "bbox_to_anchor": (1, 1)})
# gpd.GeoSeries(mel_boundary).boundary.plot(ax = axs[0], edgecolor = 'black', linewidth = .2)
gdf_mel.plot(ax=axs[1], column='EO_local_decile', categorical=True, cmap=cmap, legend=True,
             missing_kwds={"color": "white", 'edgecolor': 'lightgreen',
                           'hatch': '/////', 'linewidth': 0.01},
             legend_kwds={"loc": "upper left", 'ncols': 1, "bbox_to_anchor": (1, 1)})
# gpd.GeoSeries(mel_boundary).boundary.plot(ax = axs[1], edgecolor = 'black', linewidth = .2)

axs[0].set_title('Local deciles of ER')
axs[0].text(0.02, 0.98, '[A]', transform=axs[0].transAxes,
            fontsize=12, weight='bold',  va='top')
axs[1].set_title('Local deciles of EO')
axs[1].text(0.02, 0.98, '[B]', transform=axs[1].transAxes,
            fontsize=12, weight='bold',  va='top')
# handles, legend_labels = ax.get_legend_handles_labels()
# ax.legend(handles, legend_labels, loc = 'outside upper right', ncols = 6, fancybox = False)

for ax in axs:
    ax.set_xlim(144.25, 145.9)
    ax.set_ylim(-38.6, -37.1)
    # ax.set_facecolor('#0066CC')
    ax.set_facecolor('lightskyblue')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    gdf_vic.plot(ax=ax, color='lightgrey', zorder=-2)

    # ax.set_axis_off()

# f.savefig('MEL_SA1_plot_revised.png', dpi = 600)

# %% MEL with diff of ER and EO local decile
f, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=False)
gdf_mel.plot(ax=axs[0], column='ER_local_decile', categorical=True, cmap=cmap, legend=True,
             missing_kwds={"color": "white", 'edgecolor': 'lightgreen',
                           'hatch': '/////', 'linewidth': 0.01},
             legend_kwds={"loc": "upper left", 'ncols': 1, "bbox_to_anchor": (1, 1), 'markerscale': 0.5,
                          'labelspacing': 0.1, 'columnspacing': 0.25})
# gpd.GeoSeries(mel_boundary).boundary.plot(ax = axs[0], edgecolor = 'black', linewidth = .2)
gdf_mel.plot(ax=axs[1], column='EO_local_decile', categorical=True, cmap=cmap, legend=True,
             missing_kwds={"color": "white", 'edgecolor': 'lightgreen',
                           'hatch': '/////', 'linewidth': 0.01},
             legend_kwds={"loc": "upper left", 'ncols': 1, "bbox_to_anchor": (1, 1), 'markerscale': 0.5,
                          'labelspacing': 0.1, 'columnspacing': 0.25})

gdf_mel['D_local_decile'] = gdf_mel['EO_local_decile'] - \
    gdf_mel['ER_local_decile']
gdf_mel.plot(ax=axs[2], column='D_local_decile', categorical=True, cmap='PuOr_r', legend=True,
             missing_kwds={"color": "white", 'edgecolor': 'lightgreen',
                           'hatch': '/////', 'linewidth': 0.01},
             legend_kwds={"loc": "upper left", 'ncols': 2, "bbox_to_anchor": (1, 1), 'markerscale': 0.5,
                          'labelspacing': 0.1, 'columnspacing': 0.25})

# gpd.GeoSeries(mel_boundary).boundary.plot(ax = axs[1], edgecolor = 'black', linewidth = .2)

axs[0].set_title('Local decile of ER')
axs[0].text(0.02, 0.98, '[A]', transform=axs[0].transAxes,
            fontsize=12, weight='bold',  va='top')
axs[1].set_title('Local decile of EO')
axs[1].text(0.02, 0.98, '[B]', transform=axs[1].transAxes,
            fontsize=12, weight='bold',  va='top')
axs[2].set_title('Local decile of EO - Local decile of ER')
axs[2].text(0.02, 0.98, '[C]', transform=axs[2].transAxes,
            fontsize=12, weight='bold',  va='top')
# handles, legend_labels = ax.get_legend_handles_labels()
# ax.legend(handles, legend_labels, loc = 'outside upper right', ncols = 6, fancybox = False)

for ax in axs:
    ax.set_xlim(144.25, 145.9)
    ax.set_ylim(-38.6, -37.1)
    # ax.set_facecolor('#0066CC')
    ax.set_facecolor('lightskyblue')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    gdf_vic.plot(ax=ax, color='lightgrey', zorder=-2)

    # ax.set_axis_off()

# f.savefig('MEL_SA1_plot_revised_b.png', dpi = 600)


# %%
gdf_nsw = gdf[gdf.STE_NAME16 == 'New South Wales']
gdf_nsw = gdf_nsw.set_geometry('geometry')

# %%
gdf_syd = gdf[gdf.GCC_NAME16 == 'Greater Sydney']
gdf_syd = pd.merge(right=df_syd, left=gdf_syd, how='left',
                   right_on='SA1', left_on='SA1_MAIN16')
gdf_syd.EO_local_decile = gdf_syd.EO_local_decile.astype('Int64')
gdf_syd.ER_local_decile = gdf_syd.ER_local_decile.astype('Int64')
gdf_syd = gdf_syd[['SA1_MAIN16', 'ER_local_decile',
                   'EO_local_decile', 'geometry']]
gdf_syd = gdf_syd.set_geometry('geometry')

# %%
f, axs = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
gdf_syd.plot(ax=axs[0], column='ER_local_decile', categorical=True, cmap=cmap, legend=True,
             missing_kwds={"color": "white", 'edgecolor': 'lightgreen',
                           'hatch': '/////', 'linewidth': 0.01},
             legend_kwds={"loc": "upper left", 'ncols': 1, "bbox_to_anchor": (1, 1)})
gdf_syd.plot(ax=axs[1], column='EO_local_decile', categorical=True, cmap=cmap, legend=True,
             missing_kwds={"color": "white", 'edgecolor': 'lightgreen',
                           'hatch': '/////', 'linewidth': 0.01},
             legend_kwds={"loc": "upper left", 'ncols': 1, "bbox_to_anchor": (1, 1)})
axs[0].set_title('Local deciles of ER')
axs[0].text(0.02, 0.98, '[A]', transform=axs[0].transAxes,
            fontsize=12, weight='bold',  va='top')
axs[1].set_title('Local deciles of EO')
axs[1].text(0.02, 0.98, '[B]', transform=axs[1].transAxes,
            fontsize=12, weight='bold',  va='top')
# handles, legend_labels = ax.get_legend_handles_labels()
# ax.legend(handles, legend_labels, loc = 'outside upper right', ncols = 6, fancybox = False)

for ax in axs:
    ax.set_xlim(149.9, 151.7)
    ax.set_ylim(-34.35, -32.9)
    ax.set_facecolor('skyblue')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    gdf_nsw.plot(ax=ax, color='lightgrey', zorder=-2)

f.savefig('SYD_SA1_plot_revised.png', dpi=600)

# %% SYD with diff of ER and EO local decile
f, axs = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=False)
gdf_syd.plot(ax=axs[0], column='ER_local_decile', categorical=True, cmap=cmap, legend=True,
             missing_kwds={"color": "white", 'edgecolor': 'lightgreen',
                           'hatch': '/////', 'linewidth': 0.01},
             legend_kwds={"loc": "upper left", 'ncols': 1, "bbox_to_anchor": (1, 1), 'markerscale': 0.5,
                          'labelspacing': 0.1, 'columnspacing': 0.25})
# gpd.GeoSeries(mel_boundary).boundary.plot(ax = axs[0], edgecolor = 'black', linewidth = .2)
gdf_syd.plot(ax=axs[1], column='EO_local_decile', categorical=True, cmap=cmap, legend=True,
             missing_kwds={"color": "white", 'edgecolor': 'lightgreen',
                           'hatch': '/////', 'linewidth': 0.01},
             legend_kwds={"loc": "upper left", 'ncols': 1, "bbox_to_anchor": (1, 1), 'markerscale': 0.5,
                          'labelspacing': 0.1, 'columnspacing': 0.25})

gdf_syd['D_local_decile'] = gdf_syd['EO_local_decile'] - \
    gdf_syd['ER_local_decile']
gdf_syd.plot(ax=axs[2], column='D_local_decile', categorical=True, cmap='PuOr_r', legend=True,
             missing_kwds={"color": "white", 'edgecolor': 'lightgreen',
                           'hatch': '/////', 'linewidth': 0.01},
             legend_kwds={"loc": "upper left", 'ncols': 2, "bbox_to_anchor": (1, 1), 'markerscale': 0.5,
                          'labelspacing': 0.1, 'columnspacing': 0.25})

# gpd.GeoSeries(mel_boundary).boundary.plot(ax = axs[1], edgecolor = 'black', linewidth = .2)

axs[0].set_title('Local deciles of ER')
axs[0].text(0.02, 0.98, '[A]', transform=axs[0].transAxes,
            fontsize=12, weight='bold',  va='top')
axs[1].set_title('Local deciles of EO')
axs[1].text(0.02, 0.98, '[B]', transform=axs[1].transAxes,
            fontsize=12, weight='bold',  va='top')
axs[2].set_title('Local decile of EO - Local decile of ER')
axs[2].text(0.02, 0.98, '[C]', transform=axs[1].transAxes,
            fontsize=12, weight='bold',  va='top')
# handles, legend_labels = ax.get_legend_handles_labels()
# ax.legend(handles, legend_labels, loc = 'outside upper right', ncols = 6, fancybox = False)

for ax in axs:
    ax.set_xlim(149.9, 151.7)
    ax.set_ylim(-34.35, -32.9)
    ax.set_facecolor('skyblue')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    gdf_nsw.plot(ax=ax, color='lightgrey', zorder=-2)

    # ax.set_axis_off()

f.savefig('SYD_SA1_plot_revised_b.png', dpi = 600)
