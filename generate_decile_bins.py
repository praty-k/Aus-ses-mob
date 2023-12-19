# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:45:30 2023

@author: pkollepara
"""
from scipy.stats import gaussian_kde
import numpy as np

def generate_decile_bins(data, weights):
    ks = gaussian_kde(data, weights=weights, bw_method = 0.05)
    print('bw = ', ks.factor)
    #sample_pts = np.linspace(np.min(data), np.max(data), 100) # This uses only 100 pts for cdf
    sample_pts = np.arange(np.min(data), np.max(data)+1, 1) # This uses a high resolution cdf
    cdf = np.cumsum(ks.pdf(sample_pts)*(sample_pts[1]-sample_pts[0]))
    deciles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    decile_inds = []
    for decile in deciles:
        decile_inds.append(np.argmin(np.abs(cdf-decile/100)))
    bins = np.append([np.min(data)], sample_pts[decile_inds])
    cdf_values = []
    for x in bins:
        cdf_values.append(cdf[np.where(sample_pts==x)[0][0]])
    return bins, cdf_values