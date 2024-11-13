# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:20:31 2024

@author: PKollepara
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def draw_vplot(ax, x, y, df, xlabel, ylabel, title, subplotlabel):
    df_agg = df.groupby(x).median()
    sns.set_style('whitegrid')
    sns.violinplot(ax = ax, data = df, x = x, y = y, inner = 'box', density_norm = 'count', color = 'skyblue', linewidth = 0.1, 
                  inner_kws=dict(box_width=15, whis_width=2))
    ax.plot(df_agg.index.values-1, df_agg[y].values, lw = 0.85, marker = 's', ms = 5, ls = '--', color = 'k')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    #ax.legend(ncol=3, loc = 'upper right')
    ax.text(0.02, 0.98, f'[{subplotlabel}]', transform=ax.transAxes, fontsize=12, weight = 'bold',  va='top') 
    ax.grid()
    
def draw_boxenplot(ax, x, y, df, xlabel, ylabel, title, subplotlabel):
    df_agg = df.groupby(x).median()
    #sns.set_style('darkgrid')
   
    sns.boxenplot(ax = ax, data = df, x = x, y = y, linewidth=.5, k_depth = 5,
                  flier_kws = dict(facecolor = 'k', edgecolor = 'k', marker = 'o', 
                                   alpha = .1, linewidths = 0), legend = 'brief')
    ax.plot(df_agg.index.values-1, df_agg[y].values, lw = 0.5, marker = 's', ms = 0, ls = '--', color = 'maroon')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    #ax.legend(ncol=3, loc = 'upper right')
    ax.text(0.02, 0.98, f'[{subplotlabel}]', transform=ax.transAxes, fontsize=12, weight = 'bold',  va='top') 
    ax.grid()
