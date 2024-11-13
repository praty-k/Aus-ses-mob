# -*- coding: utf-8 -*-
"""
Created on Sun May 19 12:10:09 2024

Replicating this notebook in python https://crd230.github.io/lab8.html#Load_necessary_packages

@author: PKollepara
"""

# Graphics
import matplotlib.pyplot as plt
import seaborn
from pysal.viz import splot
from splot.esda import plot_moran
import contextily

# Analysis
import geopandas as gpd
import pandas as pd
from pysal.explore import esda
from pysal.lib import weights
from numpy.random import seed

#%%
df = pd.read_csv('https://raw.githubusercontent.com/crd230/data/master/PLACES_WA_2022_release.csv')