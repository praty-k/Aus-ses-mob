# -*- coding: utf-8 -*-
"""
This script compares 'sample coverage' over all SA1 regions analysed by the 
script 'SA1_coverage_approx.py', and plots a histogram of the differences
These can be shown relative to the baseline interval. 
"""
import pandas as pd
import os
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np 

# if test flag is false, the program will run through the whole list. 
test_flag = False
# if test_flag is true, the program will run through the first test_size elements
test_size = 100
test_label = "test_n" + str(test_size)

# specify time periods and spatial extent to compare (there are four time periods, and 
# two spatial extents, for 8 data sets total.)
#region label
#region_label_corr = "GSYD"
#region_label_data = "SYD_GCC"
region_label_corr = "GMEL"
region_label_data = "MEL_GCC"
#period label
#period_label_data = "01012020"
#period_label_data = "01012022"
#period_label_data = "01042020"
#period_label_data = "15092019"

test_period_label = "01012022"
base_period_label = "01012020"

test_period_label = "01042020"
base_period_label = "15092019"


data_dirname = Path(os.getcwd())

if test_flag:
    output_fname = data_dirname / \
                   ("SA1_comparison_" + test_label + "_" +  region_label_corr + "_" + \
                    base_period_label + "_vs_" + test_period_label + ".csv")
else:
    output_fname = data_dirname / \
                   ("SA1_comparison_" + region_label_corr + "_" + \
                    base_period_label + "_vs_" + test_period_label + ".csv")
                           

base_period_data_fname = data_dirname / \
               ("SA1_visits_within_DZN_" + region_label_corr + "_" + \
                base_period_label + ".csv")
                   
test_period_data_fname = data_dirname / \
               ("SA1_visits_within_DZN_" + region_label_corr + "_" + \
                test_period_label + ".csv")


base_data = pd.read_csv(base_period_data_fname)
test_data = pd.read_csv(test_period_data_fname)


visits_base = base_data['SA1_visits']
visits_test = test_data['SA1_visits']

rel_diff = (visits_test - visits_base) / visits_base

log_rel_diff =  (np.log(visits_test) - np.log(visits_base)) / np.log(visits_base)

log_rel_diff[np.isinf(log_rel_diff)] = np.nan

threshold = 1

plt.hist(rel_diff[abs(rel_diff) < threshold], bins=100)
plt.show()
print(len(rel_diff[abs(rel_diff) < threshold]))


plt.hist(np.log(visits_test), bins = 100)
plt.hist(np.log(visits_base), bins = 100)
plt.show()

plt.hist(np.log(visits_test) - np.log(visits_base), bins = 100)
plt.show()

plt.hist(log_rel_diff, bins=100)
plt.show()




#output_df.to_csv(output_fname)



 