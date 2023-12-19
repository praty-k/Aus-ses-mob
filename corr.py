# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:13:45 2023

@author: pkollepara
"""
from scipy.stats import spearmanr, pearsonr, norm, linregress
import numpy as np

def spearman_corr(x, y, CI=95/100, namex = 'x', namey = 'y'): #Uses Fisher transform
    sres = spearmanr(x, y) #
    r = sres.statistic
    z_score = 0.5*np.log((1+r)/(1-r))
    SE = 1/np.sqrt(len(x) - 3)

    z_1 = z_score - SE*norm.isf((1-CI)/2)
    r_1 = np.tanh(z_1)
    z_2 = z_score + SE*norm.isf((1-CI)/2)
    r_2 = np.tanh(z_2)
    #print(r, z_score, SE, z_1, r_1, z_2, r_2)
    print(f'Spearman correlation b/w {namex} and {namey}, p-value, CI low ({CI*100}%), CI high ({CI*100}%)', r, sres.pvalue, r_1, r_2)
    return (r, sres.pvalue, r_1, r_2)

def linear_regression(x, y, CI=95/100, namex = 'x', namey = 'y'): #Uses Fisher transform
    
    lres = linregress(x, y)
    s = lres.slope
    r = lres.rvalue
    c = lres.intercept
    z_score = 0.5*np.log((1+r)/(1-r))
    SE = 1/np.sqrt(len(x) - 3)
    z_1 = z_score - SE*norm.isf((1-CI)/2)
    r_1 = np.tanh(z_1)
    z_2 = z_score + SE*norm.isf((1-CI)/2)
    r_2 = np.tanh(z_2)
    #print(r, z_score, SE, z_1, r_1, z_2, r_2)
    print(f'Linear regression b/w {namex} and {namey} -- slope, r, p-value, CI low ({CI*100}%), CI high ({CI*100}%)', s, r, lres.pvalue, r_1, r_2)
    return (s, c, lres.pvalue, r, r_1, r_2)
