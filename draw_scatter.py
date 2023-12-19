# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 12:10:50 2023

@author: pkollepara
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import pymannkendall as mk

def draw_scatter(ax, seifa_deciles, y, xlabel=None, ylabel=None, title=None, subplotlabel = None):
    deciles = [*range(0, 10)]
    percent_vals = [0, 25, 75, 100]#np.linspace(0, 100, 6)
    percentiles = {}
    band_medians = {}
    band_95 = {}
    band_05 = {}
    for decile in deciles:
            #df_dec = df[df[x]==decile]
            y_decile = y[seifa_deciles == decile]
            percentile = np.percentile(y_decile, q=percent_vals)
            percentiles[str(decile)]=percentile.tolist()
            median = []
            per_95 = []
            per_05 = []
            for i, band in enumerate(percentile[0:-1]):
                    y_band = y_decile[(y_decile>percentile[i]) & (y_decile<percentile[i+1])]
                    median.append(y_band.median())
                    per_95.append(y_band.quantile(0.95))
                    per_05.append(y_band.quantile(0.05))
            band_medians[str(decile)] = median
            band_95[str(decile)] = per_95
            band_05[str(decile)] = per_05
    
    df_ptls = pd.DataFrame.from_dict(percentiles, orient='index', columns = [str(x) for x in percent_vals])
    df_band_medians = pd.DataFrame.from_dict(band_medians, orient='index', columns = [str(percent_vals[x]) + ' - ' + str(percent_vals[x+1]) for x in range(0, len(percent_vals)-1)])
    df_band_95 = pd.DataFrame.from_dict(band_95, orient='index', columns = [str(percent_vals[x]) + ' - ' + str(percent_vals[x+1]) for x in range(0, len(percent_vals)-1)])
    df_band_05 = pd.DataFrame.from_dict(band_05, orient='index', columns = [str(percent_vals[x]) + ' - ' + str(percent_vals[x+1]) for x in range(0, len(percent_vals)-1)])
    #df_ptls['local_deciles']=df_ptls.index.values
    df_ptls.reset_index(inplace = True)
    df_band_medians.reset_index(inplace = True)
    df_band_95.reset_index(inplace = True)
    df_band_05.reset_index(inplace = True)
    
    print(xlabel, 'for', title)
    sres = spearmanr(df_band_medians.index.values, df_band_medians['25 - 75'].values)
    print('Spearman: ', sres.statistic, 'p-value:', sres.pvalue)
    res = mk.original_test(df_band_medians['25 - 75'].values)
    print('MK score: ', res.s, ', p-value: ', res.p)
    #print(df_band_medians['25 - 75'].values)

    print('\n')
    markers = ['v', 's', '^']
    for idx, col in enumerate(df_band_medians.columns[1:]):
        ax.errorbar(x=df_band_medians.index.values, y=df_band_medians[col], yerr = [np.abs(df_band_05[col].values-df_band_medians[col]), np.abs(df_band_95[col].values-df_band_medians[col])], 
                    lw = 0.5, ls = '--', marker = markers[idx], elinewidth = 0.5, ecolor = 'k', capsize = 6, mew = 0.5, alpha = 1, label = col)
        #ax.plot(df_band_medians.index.values, df_band_medians[col], 
        #            lw = 0.5, ls = '--', marker = markers[idx], alpha = 1, label = col)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    #ax.legend(ncol=3, loc = 'upper right')
    ax.text(0.02, 0.98, f'[{subplotlabel}]', transform=ax.transAxes, fontsize=12, weight = 'bold',  va='top') 
    ax.grid()
    