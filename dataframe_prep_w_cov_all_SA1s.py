# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 09:31:55 2023

@author: pkollepara
"""
import numpy as np
import pandas as pd
from generate_decile_bins import generate_decile_bins

def dataframe_prep(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True) # Replaces any infinities with NaN
    df.dropna(inplace=True) # Removes rows containing NaNs
    
    ER_bins, ER_cdf_values = generate_decile_bins(df.ER_score.values, df.URP.values/df.URP.sum()) 
    EO_bins, EO_cdf_values = generate_decile_bins(df.EO_score.values, df.URP.values/df.URP.sum())

    df['ER_local_decile'] = pd.cut(df.ER_score.values, bins=ER_bins, labels=False, include_lowest=True) + 1
    df['EO_local_decile'] = pd.cut(df.EO_score.values, bins=EO_bins, labels=False, include_lowest=True) + 1

    return df