# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:26:30 2023

@author: pkollepara
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def compute_heatmap(ER_decile, EO_decile, qty, num_threshold = 0):
    deciles = range(0, 10)
    hdata = np.zeros((10, 10))
    hnum = np.zeros((10, 10))
    for ER_i in deciles:
        for EO_i in deciles:
            hnum[ER_i, EO_i] = np.sum((ER_decile-1==ER_i) & (EO_decile-1==EO_i))
            if np.all(qty=='histogram'):
                hdata[ER_i, EO_i] = hnum[ER_i, EO_i]
            else:
                if hnum[ER_i, EO_i]>num_threshold:
                    hdata[ER_i, EO_i] = np.median(qty[(ER_decile-1==ER_i) & (EO_decile-1==EO_i)])
                else:
                    hdata[ER_i, EO_i] = np.nan
    return hdata

def compute_heatmap_pop(ER_decile, EO_decile, qty, num_threshold = 0):
    deciles = range(0, 10)
    hdata = np.zeros((10, 10))
    hnum = np.zeros((10, 10))
    for ER_i in deciles:
        for EO_i in deciles:
            hnum[ER_i, EO_i] = np.sum((ER_decile-1==ER_i) & (EO_decile-1==EO_i))
            if hnum[ER_i, EO_i]>num_threshold:
                hdata[ER_i, EO_i] = np.sum(qty[(ER_decile-1==ER_i) & (EO_decile-1==EO_i)])
            else:
                hdata[ER_i, EO_i] = np.nan
    return hdata


def draw_heatmap(hdata, fname=None, cmap = None, fmt = '0.2f', title=None, htmlfile=None):
    fig, ax = plt.subplots(1, 1, figsize = (10*0.6, 8*0.6))
    sns.heatmap(hdata, annot=True, cbar=True, square = True, cmap = cmap,
                linewidths = 1, linecolor = 'white', ax = ax, fmt = fmt, annot_kws = {'fontsize': 'x-small'})
    ax.invert_yaxis()
    ax.set_xticklabels(list(range(1, 11)))
    ax.set_yticklabels(list(range(1, 11)))
    ax.set_xlabel('Local decile of EO')
    ax.set_ylabel('Local decile of ER')
    if title:    
        ax.set_title(title)
    plt.tight_layout()
    # if fname:
    #     if colname == None:
    #         colname = 'SA1'
        #fig.savefig(f'threshold_{threshold}_validated_plots/'+fname+colname+f'-threshold-{threshold}-validated-heatplot.png', dpi=400)
        #fig.savefig(f'threshold_{threshold}_validated_plots/'+fname+colname+f'-threshold-{threshold}-validated-heatplot.pdf')
        #fig_html(fig, htmlfile)
    return fig

def draw_heatmap_pop(hdata, fname=None, cmap = None, fmt = '0.2f', title=None, htmlfile=None):
    fig, ax = plt.subplots(1, 1, figsize = (10*0.6, 8*0.6))
    sns.heatmap(hdata, annot=True, cbar=True, square = True, cmap = cmap,
                linewidths = 1, linecolor = 'white', ax = ax, fmt = fmt, annot_kws = {'fontsize': 'xx-small', 'rotation': 45})
    ax.invert_yaxis()
    ax.set_xticklabels(list(range(1, 11)))
    ax.set_yticklabels(list(range(1, 11)))
    ax.set_xlabel('Local decile of EO')
    ax.set_ylabel('Local decile of ER')
    if title:    
        ax.set_title(title)
    plt.tight_layout()
    # if fname:
    #     if colname == None:
    #         colname = 'SA1'
        #fig.savefig(f'threshold_{threshold}_validated_plots/'+fname+colname+f'-threshold-{threshold}-validated-heatplot.png', dpi=400)
        #fig.savefig(f'threshold_{threshold}_validated_plots/'+fname+colname+f'-threshold-{threshold}-validated-heatplot.pdf')
        #fig_html(fig, htmlfile)
    return fig


def draw_heatmap_zero_centre(hdata, fname=None, cmap = None, fmt = '.2f', title=None, cbar_title = None, htmlfile=None):
    fig, ax = plt.subplots(1, 1, figsize = (10*0.6, 8*0.6))
    sns.heatmap(hdata, annot=True, cbar=True, square = True, cmap = cmap, center = 0,
                linewidths = 1, linecolor = 'white', ax = ax, fmt = fmt, annot_kws = {'fontsize': 'x-small'}, cbar_kws={'label': cbar_title})
    ax.invert_yaxis()
    ax.set_xticklabels(list(range(1, 11)))
    ax.set_yticklabels(list(range(1, 11)))
    ax.set_xlabel('Local decile of EO')
    ax.set_ylabel('Local decile of ER')
    if title:    
        ax.set_title(title)
    # if cbar_title:
    #     cbar_ax = fig.axes[-1]
    #     cbar_ax.set_label(cbar_title)
    plt.tight_layout()
    # if fname:
    #     if colname == None:
    #         colname = 'SA1'
        #fig.savefig(f'threshold_{threshold}_validated_plots/'+fname+colname+f'-threshold-{threshold}-validated-heatplot.png', dpi=400)
        #fig.savefig(f'threshold_{threshold}_validated_plots/'+fname+colname+f'-threshold-{threshold}-validated-heatplot.pdf')
        #fig_html(fig, htmlfile)
    return fig