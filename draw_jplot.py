# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:35:50 2023

@author: pkollepara
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def draw_scatter(ax, seifa_score, y, xlabel=None, ylabel=None, title=None, subplotlabel = None, statlabel = None, slope = None, intercept = None):
    ax.scatter(x=seifa_score, y=y, alpha = 0.01, color = 'darkslategrey', linewidths = 0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(0.02, 0.98, f'[{subplotlabel}]', transform=ax.transAxes, fontsize=12, weight = 'bold',  va='top') 
    ax.text(0.2, 0.98, statlabel, transform=ax.transAxes, fontsize=12, va='top') 
    if slope!=None:
        ax.axline((0, intercept), slope = slope, color = 'k', lw = 0.5)

    ax.grid()
    
def draw_hex(ax, seifa_score, y, xlabel=None, ylabel=None, title=None, subplotlabel = None):
    ax.hexbin(x=seifa_score, y=y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(0.02, 0.98, f'[{subplotlabel}]', transform=ax.transAxes, fontsize=12, weight = 'bold',  va='top') 