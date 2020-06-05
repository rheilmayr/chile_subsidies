/*
Project: "Impacts of Chilean forest subsidies on forest cover, carbon and biodiversity" 

Authors: Robert Heilmayr, Cristian Echeverria and Eric Lambin

Purpose: Runs estimation of econometric model of land use change. Generates
	model parameters that are used in simulation.py.

Inputs: Input data can be downloaded from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6RDDQH 
*/


/// Set-up
clear
clear matrix
set more off


/// Read data
loc dir D:\ // Set to path where you have saved estimation.csv
insheet using `dir'estimation.csv


/// Create variables and interactions for regressions
tabulate luoption, gen(luo)
tabulate region, gen(region)

tabulate lufrom, gen(olu)
rename olu3 olu5
rename olu4 olu19
rename olu2 olu3

gen t2011 = timeperiod==2011
gen yfid = sid*10 + t2011

gen plantdum = luo2
gen agdum = luo4
gen fordum = luo1
gen shrubdum = luo3
loc uses for plant ag
foreach use in `uses'{
	gen `use'rent = `use'dum*`use'_ev
	forval region=2/8 {
		gen `use'dum_reg`region' = `use'dum*region`region'
		}
	gen `use'dum_central = `use'dum*central
	gen `use'dum_south = `use'dum*south
	forval luq=2/3 { 
		gen `use'rent_luq`luq' = `use'rent*luq`luq'
		gen `use'dum_luq`luq' = `use'dum*luq`luq'
		}
	}

gen rent = 0
replace rent = 0 if luoption == 1
replace rent = agrent if luoption == 19
replace rent = plantrent if luoption == 3
forval luq=2/3 {
	gen rent_luq`luq'=rent*luq`luq'
	}

gen plantrent_luq1 = plantrent * luq1
gen agrent_luq1 = agrent * luq1
gen forrent_luq1 = forrent * luq1

gen plantdum_luq1 = plantdum * luq1
gen agdum_luq1 = agdum * luq1
gen fordum_luq1 = fordum * luq1

loc vars plantrent_luq1 plantrent_luq2 plantrent_luq3 plantdum_luq1 plantdum_luq2 plantdum_luq3 ///
	plantdum_south plantdum_central  ///
	agrent_luq1 agrent_luq2 agrent_luq3 agdum_luq1 agdum_luq2 agdum_luq3 ///
	agdum_south agdum_central  ///
	forrent_luq1 forrent_luq2 forrent_luq3 fordum_luq1 fordum_luq2 fordum_luq3 ///
	fordum_south fordum_central
loc olus 1 3 5 19
loc reg_vars
foreach olu in `olus'{
	foreach variable in `vars'{
   		gen olu`olu'_`variable' = olu`olu' * `variable'
		loc reg_vars `reg_vars' olu`olu'_`variable'
		}
	}

	
/// Run primary model specification
eststo pooled: clogit luchoice `reg_vars' ///
	if pooled_sample==1 & oos==0, group(yfid) cluster(comuna)	
esttab using `dir'results_pool.csv, se star(* 0.10 ** 0.05 *** 0.01) replace
count if e(sample) & luchoice


/// Export coefs and covariance matrix
matrix cov = e(V)
matrix coefs = e(b)
mat2txt, matrix(cov) saving(`dir'cov) replace
mat2txt, matrix(coefs) saving(`dir'coefs) replace
eststo clear


/// Run robustness check with individual observation per property
eststo prop_sample: clogit luchoice `reg_vars' ///
	if pooled_sample==1 & oos==0 & prop_keep==1, group(yfid) cluster(comuna)			
esttab using `dir'results_propsample.csv, se star(* 0.10 ** 0.05 *** 0.01) replace	
count if e(sample) & luchoice

