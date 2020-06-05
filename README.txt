Replication code for "Impacts of Chilean forest subsidies on forest cover, carbon and biodiversity"

Authors: Paper authored by Robert Heilmayr, Cristian Echeverria and Eric F. Lambin
Code written by Robert Heilmayr.

Depdendencies: Code was written in a combination of Python 3 and Stata 14.
Required python packages are pandas (0.24.1), numpy (1.16.2), statsmodels (0.10.1) and sklearn (0.20.3)

Data: Input data is available at https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6RDDQH

Summary of scripts: Code should be run in the following order:
1) environment.yml: Make file for conda computing environment used for python code.
2) estimation.do: Estimates coefficients for land use change model
3) simulation.py: Runs Monte Carlo Simulation. Generates all tables and paper results.

Pre-processing scripts are not included in this repository. Additional scripts available upon request:
1) createInputRasters.py: Generates raster input data.
2) econFullExecRaster.py: Draws sample of points, extracts attributes and 
    calculates rents to different land uses for these points. 
    Generates estimation.csv and simulation.csv input files.
3) carbonModel.py: Summarizes carbon data by region and land use.
    Generates co2_metrics.csv input file.
4) metaAnalysis.R: Conduct meta-analysis to generate 
    Generates biodiversity_metrics.csv input file.