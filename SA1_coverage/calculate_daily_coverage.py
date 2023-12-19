# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:25:54 2023

@author: pkollepara
"""


import numpy as np
import pandas as pd

cities = ['GMEL', 'GSYD']

for city in cities:
    df = pd.read_csv(f'SA1_visits_within_DZN_{city}_01012020.csv')
    df['jan22_base_daily_coverage'] = df.ratio/31
    df = df[['SA1_code', 'jan22_base_daily_coverage']]
    df = df.dropna()
    df.to_csv(f'Daily_coverage_{city}_jan22base.csv')
    
    df = pd.read_csv(f'SA1_visits_within_DZN_{city}_01012022.csv')
    df['jan22_test_daily_coverage'] = df.ratio/31
    df = df[['SA1_code', 'jan22_test_daily_coverage']]
    df = df.dropna()
    df.to_csv(f'Daily_coverage_{city}_jan22test.csv')
    
    df = pd.read_csv(f'SA1_visits_within_DZN_{city}_15092019.csv')
    df['apr20_base_daily_coverage'] = df.ratio/31
    df = df[['SA1_code', 'apr20_base_daily_coverage']]
    df = df.dropna()
    df.to_csv(f'Daily_coverage_{city}_apr20base.csv')
    
    df = pd.read_csv(f'SA1_visits_within_DZN_{city}_01042020.csv')
    df['apr20_test_daily_coverage'] = df.ratio/31
    df = df[['SA1_code', 'apr20_test_daily_coverage']]
    df = df.dropna()
    df.to_csv(f'Daily_coverage_{city}_apr20test.csv')
