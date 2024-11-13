Readme File 

COVERAGE
The folder "SA1_coverage" contains files with the names "Daily_coverage_<CITY>_<time><"test" or "base">"

These files contain an identifier for SA1s and the daily sample coverage for that SA1, for each city and time period.


MAIN ANALYSIS

The main code that produces most of the plots is "Refactored_data_analysis_w_cov_all_SA1s.py"

It reads in the files "MEL__seifa_all_filter" and "SYD__seifa_all_filter". These files contain the daily average mobility counts for each time period of interest for SA1s, their SEIFA scores, National SEIFA deciles, URP (Usual resident population) and SA1 identifier. 

In conjunction with the coverage files described earlier, the main code produces most of the plots in the text. It also creates two files which are used for plotting the maps.

The following files contain functions that the main code makes use of:
corr.py
dataframe_prep_w_cov_all_SA1s.py
draw_jplot.py
draw_lplot.py
draw_scatter.py
ER_EO_heatmap.py
generate_decile_bins.py
mono_test.py

They make use of the following packages:
numpy, scipy, pandas, matplotlib, seaborn, pymannkendall

spatial-analysis.py performs regression with various models.

The maps of Melbourne and Sydney showing the ER and EO deciles are produced using the file "Map_plots.py". It requires the geopandas package. -- These files have not been uploaded due to storage limitations


EXTERNAL VALIDATION -- These files have not been uploaded due to storage limitations
The folder "External validation" contains all the files needed for validating the Pathzz data set against the DoT data set

In this folder, the file "geocoding102.py" finds the coordinates of SCATS sites, using files from "GMEL_roads" (Open Street map files) and creates the mapping table "102-SCATS-xy.csv"

Then "geocoding_analysis.py" uses this to create a mapping of DZNs and the intersections they contain, this file is "DZN-SCATS-map.csv".

The folder "Mobility_aggregation_by_destination" contains aggregate DZN pass through counts from Pathzz.

The final plots are produced by the file "Validation_pathzz_vicroads.py"

GEOCODING example
Consider the intersection of `Tesla Road` and `Edison Street` which is close to the `Musk Dungeon` (these roads and locations do not exist in Melbourne), the description in the DoT dataset could be any of the following:

 `TESLA RD NR EDISON ST NR MUSK DGN`
 `TESLA RD / EDISON ST NR MUSK DGN`
 `TESLA RD NR EDISON ST (MUSK DGN)`

We use the strings `NR` and `/` as separators, remove the string in brackets and remove the strings like `RD`, `ST`, `LN` or `PL` etc. Now we are left with `TESLA` and `EDISON`, which can be used to search intersections with the same string in the `name` fields produced by OSM geocoding.




