# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 12:07:02 2023

@author: pkollepara
"""
from scipy.stats import spearmanr, norm
import pymannkendall as mk
import numpy as np

#https://vsp.pnnl.gov/help/vsample/design_trend_mann_kendall.htm
#https://pypi.org/project/pymannkendall/
def mono_test_mk(df, x, y, loc = None, prd = None, qty = None, show = False, CI = 95): #df is the dataframe, loc is either Melbourne or Sydney, prd is Apr2020 or Jan2022, qty is log(ratio of mob) or log(ratio of coverage)
    df_xy = df[[x, y]].copy()
    tmp = df_xy.groupby(x).median().reset_index()
    MK_res = mk.original_test(tmp[y].values)
    S = MK_res.s
    z = MK_res.z
    SE = np.sqrt(MK_res.var_s)
    LCI = z*SE - 1.96*SE**2
    UCI = z*SE + 1.96*SE**2
    d = {(prd, x, loc, qty, 'score'): MK_res.s, 
         (prd, x, loc, qty, 'p'): MK_res.p, 
         (prd, x, loc, qty, 'LCI'): LCI, 
         (prd, x, loc, qty, 'UCI'): UCI}
    if show:
        print({'loc': loc, 'prd': prd, 'qty': qty, 'median values': tmp[y].values})
        print(MK_res)
        print('\n')
    print(MK_res)
    return d

def mono_test_spearman(df, x, y, loc = None, prd = None, qty = None, CI = 95/100):
    df_xy = df[[x, y]].copy()
    tmp = df_xy.groupby(x).median().reset_index()
    sp_res = spearmanr(tmp[x].values, tmp[y].values)
    
    r = sp_res.statistic
    z_score = 0.5*np.log((1+r)/(1-r))
    SE = 1/np.sqrt(len(x) - 3)

    z_1 = z_score - SE*norm.isf((1-CI)/2)
    r_1 = np.tanh(z_1)
    z_2 = z_score + SE*norm.isf((1-CI)/2)
    r_2 = np.tanh(z_2)
    
    d = {(prd, x, loc, qty, 'score'): sp_res.statistic, 
         (prd, x, loc, qty, 'p'): sp_res.pvalue,
         (prd, x, loc, qty, 'CI_low'): r_1, 
         (prd, x, loc, qty, 'CI_high'): r_2}
    return d

def mk_test_manual(values):
    count_pos = 0
    count_neg = 0
    count_equal = 0
    for i in range(0, len(values)):
        for j in range(i+1, len(values)):
            if values[i]<values[j]:
                count_pos+=1
            elif values[i]>values[j]:
                count_neg+=1
            else:
                count_equal+=1
    print('pos, neg, equal: ', count_pos, count_neg, count_equal)
    print('MK Score: ', count_pos - count_neg)
            