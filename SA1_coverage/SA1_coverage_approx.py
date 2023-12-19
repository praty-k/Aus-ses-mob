# -*- coding: utf-8 -*-
"""
This script approximates the sample coverage level in each SA1 by extracting 
the 'customer penetration' rate from each SA1 into the DZN that completely contains
its boundary. This procedure excludes any SA1 for which there is not DZN 
that completely contains its boundary.

input files are the lists of SA1 regions, linked to the DZNs containing them 
(these were pre-processed in ArcGIS using a spatial join 'within' operator)
after performing a coordinate projection to [GDA Australia Albers]. 
These correspondences are split into those generated for Sydney and 
Melbourne.

The mobility data is then sampled for each SA1 by reading in the mobility
analytics report for the DZN containing it. From the coverage table, the 
'customer penetration' field corresponds to the estimated fraction of the SA1
resident population that travelled to the DZN (which, containing the SA1 
                                               completely, provides an 
                                               approximation of its total sample
                                               coverage.) 
"""
import pandas as pd
import os
from pathlib import Path

# if test flag is false, the program will run through the whole list. 
test_flag = False
# if test_flag is true, the program will run through the first test_size elements
test_size = 100
test_label = "test_n" + str(test_size)

# specify a time period and spatial extent (there are four time periods, and 
# two spatial extents, for 8 data sets total.)
#region label
region_label_corr = "GSYD"
region_label_data = "SYD_GCC"
#period label
#period_label_data = "01012020"
#period_label_data = "01012022"
#period_label_data = "01042020"
period_label_data = "15092019"

output_dirname = Path(os.getcwd())

if test_flag:
    output_fname = output_dirname / \
                   ("SA1_visits_within_DZN_" + test_label + "_" +  region_label_corr + "_" + \
                    period_label_data + ".csv")
else:
    output_fname = output_dirname / \
                   ("SA1_visits_within_DZN_" + region_label_corr + "_" + \
                    period_label_data + ".csv")
                           

# base directory
base_dirname = Path("C:/Users/czachreson/OneDrive - The University of Melbourne" \
                    "/Pre_2022_10_21/Pathzz_MCDS_SHARED")

# directory in which correspondences are located
corr_dirname = base_dirname / "maps"

corr_fname = corr_dirname / ("SA1_within_DZN_" + region_label_corr +"_table.txt")

DZN_fname = corr_dirname / ("GCC_SA1_DZN_tables") / \
            ("DZN_within_" + region_label_corr + ".txt")
            
SA1_fname = corr_dirname / ("GCC_SA1_DZN_tables") / \
            ("SA1_within_" + region_label_corr + ".txt")

# directory in which mobility data is located
data_dirname = base_dirname \
/ "PathzzDataCollection" / "DownloadedAnalyticsReport" / region_label_data / period_label_data 
  

corr_SA1_DZN_raw = pd.read_csv(corr_fname)

SA1_DZN_only = corr_SA1_DZN_raw[['SA1_MAIN16', 'DZN_CODE16' ]].copy()

SA1_table = pd.read_csv(SA1_fname) 

DZN_table = pd.read_csv(DZN_fname)

#loop through DZN table, construct data analytics report filenames
report_fnames = []
DZN_table.reset_index()
for index, row in DZN_table.iterrows():
    SA2_label = str(row['SA2_MAIN16'])
    DZN_label = str(row['DZN_CODE16'])
    
    DZN_fname = SA2_label + "_" + DZN_label + "_DZN_" + period_label_data + ".xlsx"
    
    DZN_fname = data_dirname / DZN_fname
    
    
    report_fnames.append(DZN_fname)
        
    

DZN_table['data_fname'] = report_fnames

SA1_pop = []
SA1_visits = []
ratio = []
SA1_code = []
DZN_code = []
exclusion_code = []

#iterate through SA1 regions and see what the coverage level is from pathzz
# add NaNs for any missing data (either the data isn't there or 
# no DZN could be linked to the SA1)

iterator = 0


SA1_DZN_only.reset_index()
for index, row in SA1_DZN_only.iterrows():
    
    
    exclude = False
    
    ex_code = "NA"
    
    SA1_label = int(row['SA1_MAIN16'])
    
    DZN_label = row['DZN_CODE16']
    
    if DZN_label.isnumeric():
        DZN_label = int(DZN_label)
    else:
        print("no containing DZN for SA1:  " + str(SA1_label) )
        exclude = True
        ex_code = "no containing DZN"
    
    #check if the file exists
    if not(exclude):
        # look up the filename for the DZN: 
        DZN_full = DZN_table.loc[DZN_table['DZN_CODE16'] == DZN_label]
        mob_fname = DZN_full['data_fname'].values[0]
        
        if not(os.path.isfile(mob_fname)):
            print("no file for SA1->DZN:  " + str(SA1_label) + " -> " + str(DZN_label) )
            exclude = True
            ex_code = "no Pathzz data for DZN"
        
        
    
    # check if the file exists and if so, read in: 
    if not(exclude):
        print("checking data for SA1->DZN:  " + str(SA1_label) + " -> " + str(DZN_label) )
        DZN_mobility = pd.read_excel(mob_fname, \
                                     sheet_name = "Catchment", \
                                     skiprows = 4)
            
        SA1_mob = DZN_mobility.loc[DZN_mobility['SA1'] == SA1_label] 
        
        if (SA1_mob.shape[0] == 0):
            print("SA1:  " + str(SA1_label) + " not found for DZN: " + str(DZN_label) )
            exclude = True
            ex_code = "SA1 not found in Pathzz data"
            
    if not(exclude):    
        iterator += 1
            
        # find the SA1 in the catchment list 
        
        SA1_mob = SA1_mob.reset_index()
        
        SA1_pop.append(SA1_mob.at[0, 'Current Population'])
        SA1_visits.append(SA1_mob.at[0, 'Visits'])
        ratio.append(SA1_mob.at[0, 'Visits'] / SA1_mob.at[0, 'Current Population'])
        SA1_code.append(SA1_label)
        DZN_code.append(DZN_label)
        exclusion_code.append(ex_code)
    else:
        print("no data for SA1->DZN:  " + str(SA1_label) + " -> " + str(DZN_label) )
        SA1_pop.append(float('NaN'))
        SA1_visits.append(float('NaN'))
        ratio.append(float('NaN'))
        SA1_code.append(SA1_label)
        DZN_code.append(DZN_label)
        exclusion_code.append(ex_code)
        
        
    if test_flag and iterator == test_size:
        break
            

# append SA1 -> DZN visits to the SA1_DZN_only dataframe
output_df = pd.DataFrame()
output_df['SA1_code'] = SA1_code
output_df['DZN_code'] = DZN_code
output_df['SA1_pop'] = SA1_pop
output_df['SA1_visits'] = SA1_visits
output_df['ratio'] = ratio
output_df['exclusion_code'] = exclusion_code

output_df.to_csv(output_fname)



 