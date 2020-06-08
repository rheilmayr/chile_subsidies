# -*- coding: utf-8 -*-
"""
Project: "Impacts of Chilean forest subsidies on forest cover, carbon and biodiversity" 

Authors: Robert Heilmayr, Cristian Echeverria and Eric Lambin

Purpose: Runs Monte Carlo simulation of land use and creates summary tables
    of land use, carbon and biodiversity. Reproduces all tables and quantiative
    results for paper.

Inputs: Input data can be downloaded from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6RDDQH 
    Prior to running this script, estimations script (estimation.do) needs to be run in Stata.

Note: Full simulation with 1000 iterations can be slow to run (>24 hours).
    
"""

# =============================================================================
# Imports
# =============================================================================
import pandas as pd
import numpy as np
import statsmodels.stats.api as sms
from sklearn import metrics
np.random.seed(805)
import os

# =============================================================================
# Set parameters
# =============================================================================
n = 1000
data_dir = '' # Set to directory where 
out_dir = data_dir + 'sim/'
results_dir = data_dir + 'results/'


# =============================================================================
# Load files
# =============================================================================
# 1. Point observations
data_csv = data_dir + 'simulation.csv'
data_df = pd.read_csv(data_csv, index_col = 0)

# 2. Summary of carbon density by region (tC/ha)
co2_csv = data_dir + 'co2_metrics.csv'

# 3. Results from biodiversity metanalysis
bio_csv = data_dir + 'biodiversity_metrics.csv'

# 4. Results of econometric model (Created in stata using estimation.do)
coefs_csv = data_dir + 'coefs.txt'
cov_csv = data_dir + 'cov.txt' 


# =============================================================================
# Make output directories
# =============================================================================
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
if not os.path.exists(results_dir):
    os.mkdir(results_dir)


# =============================================================================
# Define functions
# =============================================================================
area_wght = 0.1 # convert km2 to thousand hectares

def calc_kappa_sim(val_df):
    """
    Parameters
    ----------
    val_df: pandas dataframe
        Dataframe containing information from validation sites. Contains:
        olu: series
            Observations of original land use from maps
        
        elu: series
            Observations of final land use from maps
        
        sim: series
            Observations of final land use from simulation
        
    Returns
    -------
    kappa: float
        Traditional kappa
    
    kappa_sim: float
        Adjusted kappa to take account of transitions
    
    kappa_transition: float
        Transition component of decomposition of kappa_sim
    
    kappa_translocation: float
        Translocation component of decomposition of kappa_sim
    
    Source
    -------
    New kappa statistics taken from van Vliet et al, 2011
    """
    classes = np.unique(val_df[['olu', 'elu', 'sim']])
    p_e_trans = 0
    p_max = 0
    for j in classes:
        olu_share = np.float((val_df['olu']==j).sum())/np.size(val_df['olu'])
        sum_mult = 0    
        sum_min = 0
        for i in classes:
            trans_share = np.float((val_df.loc[val_df['olu']==j, 'elu']==i).sum()) / \
                (val_df['olu']==j).sum()
            sim_share = np.float((val_df.loc[val_df['olu']==j, 'sim']==i).sum()) / \
                (val_df['olu']==j).sum()
            sum_mult += trans_share * sim_share
            sum_min += np.min((trans_share, sim_share))
        p_e_trans += olu_share * sum_mult
        p_max += olu_share * sum_min
    
    cm = metrics.confusion_matrix(val_df['elu'], val_df['sim'])
    p_0 = np.float(np.diag(cm).sum())/np.sum(cm)
    
    kappa_sim = (p_0 - p_e_trans) / (1. - p_e_trans)
    kappa_transition = (p_max - p_e_trans) / (1. - p_e_trans)
    kappa_translocation = (p_0 - p_e_trans) / (p_max - p_e_trans)
    kappa = metrics.cohen_kappa_score(val_df['elu'], val_df['sim'])
    return kappa, kappa_sim, kappa_transition, kappa_translocation



def sep_formulas(formula_list, lu_types):
    """
    Parameters
    ----------
    formula_list: list
        List containing text strings of formulas to execute for 
        simulation calculation
    
    lu_types: list
        List of land use stubs to be found in formula_list
    
    Returns
    -------
    formula_dict: dict
        Dict containing full prediction formula for each land use type
        
    """
    formula_dict={}
    for lu_type in lu_types:
        l=[formula for formula in formula_list if lu_type in formula]
        s='+'.join(l)
        formula_dict[lu_type]=s
    return formula_dict



def gen_formulas_df(coefs_df, data_df_name = 'data_df', 
                    coefs_df_name = 'coefs_df',
                    lu_types=['ag', 'plant', 'for']):
    """
    Checks to make sure data_dict aligns with model coefficients from stata.
    Generates formulas for prediction generation.
    
    Parameters
    ----------
    coefs_df: pandas series
        Series containing all coefficients for conditional logit model
    
    data_dict: dict
        Dictionary of location of corresponding data
    
    Returns
    -------
    formula_list: list
        List containing text strings of formulas to execute for 
        simulation calculation
    
    data_dict: dict
        Updated data_dict including constants for dummy variables
    """
    coefs_list=[var.split('_') for var in coefs_df.index]
    formula_list=[]
    for var_list in coefs_list:
        formula=[data_df_name + "['"+var+"']" for var in var_list]
        formula=['1.' if 'dum' in string else string for string in formula]
        formula='*'.join(formula)
        formula+="*"+coefs_df_name+"['"+"_".join(var_list)+"']"
        formula_list.append(formula)
    formula_dict=sep_formulas(formula_list, lu_types)
    return formula_dict


class Simulator:
    def load_coefs(self, coefs_csv, cov_csv):
        coefs_df = pd.read_csv(coefs_csv, sep = '\t')
        self.coefs_df = coefs_df.iloc[:, 1:-1]      
        cov_df = pd.read_csv(cov_csv, sep = '\t')
        self.cov_df = cov_df.set_index('Unnamed: 0').iloc[:, :-1]

    def draw_coefs(self, n):
        self.n = n
        if self.n == 1:
            coefs_draw = self.coefs_df.T
            self.coefs_draw = coefs_draw.rename(index = lambda x: x.replace('luchoice:', '')) 
            
        else:
            coefs_draw = np.random.multivariate_normal(self.coefs_df.values[0], 
                        self.cov_df.values, size = n).T
            coefs_draw = pd.DataFrame(coefs_draw, index = self.coefs_df.columns)
            self.coefs_draw = coefs_draw.rename(index = lambda x: x.replace('luchoice:', ''))       

    def one_sim(self, input_df_name, i):
        coefs = self.coefs_draw.iloc[:, i]
        formula_dict = gen_formulas_df(coefs, 
                                      data_df_name = input_df_name,
                                      coefs_df_name = 'coefs',
                                      lu_types=['ag', 'plant', 'for'])
        num_df = pd.DataFrame()
        for lu, formula in formula_dict.items():
            num_df[lu] = np.exp(eval(formula))
        denom = 1 + num_df.sum(axis = 1)
        self.prob_df = num_df.divide(denom, axis = 'index')
        self.prob_df['nmr'] = 1-self.prob_df.sum(axis = 1)
        try:
            classes = self.prob_df.apply(lambda row: self.prob_df.columns[np.random.choice(4, 1, 
                                            p = row)].values[0], axis = 1)
            rename_dict = {'nmr': 5, 'plant': 3, 'for': 1, 'ag': 19}
            classes = classes.apply(lambda x: rename_dict[x])
        except ValueError:
            classes = pd.Series(np.nan, index = eval(input_df_name).index)
            return classes
        return classes

    def sim(self, data_df):
        ## Period 1
        self.p1_inputs = data_df[['south', 'central', 'luq1', 'luq2', 'luq3',
                                  'ag_ev01', 'for_ev', 'plant_ev01', 'p_ev01_ns']]
        self.p1_inputs = self.p1_inputs.rename(columns = {'ag_ev01': 'agrent',
                                                          'for_ev': 'forrent'})
        dummies=pd.get_dummies(data_df['lu_86'])
        dummies = dummies.rename(columns = lambda x: 'olu' +str(int(x)))
        self.p1_inputs[dummies.keys()]=dummies
        self.p1_inputs['p_ev01_rs'] = self.p1_inputs['olu1'] * self.p1_inputs['p_ev01_ns'] \
            + (1 - self.p1_inputs['olu1']) * self.p1_inputs['plant_ev01']
        
        container = pd.DataFrame(data = None, index = self.p1_inputs.index, columns = range(self.n))
        self.p1_outcomes = {'sub': container,
                            'ns': container.copy(),
                            'rs': container.copy()}
        for i in range(self.n):
            ## Subsidy scenario
            self.p1_inputs['plantrent'] = self.p1_inputs['plant_ev01']
            classes = self.one_sim('self.p1_inputs', i)
            self.p1_outcomes['sub'].iloc[:, i] = classes
            
            ## No subsidy scenario
            self.p1_inputs['plantrent'] = self.p1_inputs['p_ev01_ns']
            classes = self.one_sim('self.p1_inputs', i)
            self.p1_outcomes['ns'].iloc[:, i] = classes

            ## Restricted subsidy scenario 
            self.p1_inputs['plantrent'] = self.p1_inputs['p_ev01_rs']
            classes = self.one_sim('self.p1_inputs', i)
            self.p1_outcomes['rs'].iloc[:, i] = classes
            
        ## Period 2
        self.p2_inputs = data_df[['south', 'central', 'luq1', 'luq2', 'luq3',
                                  'ag_ev11', 'for_ev', 'plant_ev11', 'p_ev11_ns']]
        self.p2_inputs = self.p2_inputs.rename(columns = {'ag_ev11': 'agrent',
                                                          'for_ev': 'forrent'})
            
        self.p2_outcomes = {'sub': container.copy(),
                            'ns': container.copy(),
                            'rs': container.copy()}

        for i in range(self.n):
            ## Subsidy scenario
            dummies=pd.get_dummies(self.p1_outcomes['sub'].iloc[:, i])
            dummies = dummies.rename(columns = lambda x: 'olu' +str(int(x)))
            self.p2_inputs[dummies.keys()] = dummies            
            self.p2_inputs['plantrent'] = self.p2_inputs['plant_ev11']
            classes = self.one_sim('self.p2_inputs', i)
            self.p2_outcomes['sub'].iloc[:, i] = classes
            
            ## No subsidy scenario
            dummies=pd.get_dummies(self.p1_outcomes['ns'].iloc[:, i])
            dummies = dummies.rename(columns = lambda x: 'olu' +str(int(x)))
            self.p2_inputs[dummies.keys()] = dummies
            self.p2_inputs['plantrent'] = self.p2_inputs['p_ev11_ns']
            classes = self.one_sim('self.p2_inputs', i)
            self.p2_outcomes['ns'].iloc[:, i] = classes

            ## Restricted subsidy scenario 
            dummies=pd.get_dummies(self.p1_outcomes['rs'].iloc[:, i])
            dummies = dummies.rename(columns = lambda x: 'olu' +str(int(x)))
            self.p2_inputs[dummies.keys()] = dummies
            self.p2_inputs['p_ev11_rs'] = self.p2_inputs['olu1'] * self.p2_inputs['p_ev11_ns'] \
                + (1 - self.p2_inputs['olu1']) * self.p2_inputs['plant_ev11']
            self.p2_inputs['plantrent'] = self.p2_inputs['p_ev11_rs']
            classes = self.one_sim('self.p2_inputs', i)
            self.p2_outcomes['rs'].iloc[:, i] = classes

        return self.p2_outcomes


# =============================================================================
# Run simulation
# =============================================================================
simulator = Simulator()
simulator.load_coefs(coefs_csv, cov_csv)
simulator.draw_coefs(n)
sim_dict = simulator.sim(data_df)
sim_dict['sub'].to_csv(out_dir + 'sim_sub.csv', header = True)
sim_dict['ns'].to_csv(out_dir + 'sim_ns.csv', header = True)
sim_dict['rs'].to_csv(out_dir + 'sim_rs.csv', header = True)     


sim_sub = pd.read_csv(out_dir + 'sim_sub.csv', index_col = 0)
sim_ns = pd.read_csv(out_dir + 'sim_ns.csv', index_col = 0)
sim_rs = pd.read_csv(out_dir + 'sim_rs.csv', index_col = 0)
sim_dict = {'sub': sim_sub,
            'ns': sim_ns,
            'rs': sim_rs}


# =============================================================================
# Summarize probabilty of plantation conversion for Figure 1
# =============================================================================
plant_prob = (sim_sub==3).mean(axis = 1)
plant_prob.name = 'sub_sim'
fig_df = data_df.merge(plant_prob, left_index = True, right_index = True, how = 'left')
fig_df.to_csv(out_dir + 'fig_data.csv')


# =============================================================================
# Transition table (Table 3)
# =============================================================================
def calc_transitions(sim_df):
    sim_df = sim_df.rename(columns = lambda x: int(x))
    sim_df = sim_df.merge(data_df[['lu_86', 'region']], left_index = True, right_index = True, how = 'left')
    groups = sim_df.drop('region', axis = 1).groupby(['lu_86'])
    transition_df = pd.DataFrame({i: groups[i].value_counts() for i in range(n)})
    transition_df.index.names = ['lu_from', 'lu_to']
    transition_df = transition_df * area_wght
    return transition_df.sort_index()

def sim_stats(series):
    stats = {'mean': series.mean(),
             'ci': series.mean() - sms.DescrStatsW(series).tconfint_mean(0.05)[0]}
    return stats

transition_results = {key: calc_transitions(sim_df) for key, sim_df in sim_dict.items()}
transition_comparisons = [('rs', 'ns'),
                          ('sub', 'ns')]


for a, b in transition_comparisons:
    transition_results[(a + '-' + b)] = transition_results[(a)] - transition_results[(b)]

tab_dict = {key: pd.DataFrame(sim_stats(df.T)) for key, df in transition_results.items()}
transition_df = pd.concat(tab_dict, axis = 1).sort_index()


rows = transition_df.index
columns = transition_df.columns.get_level_values(0).unique()
t3_df = pd.DataFrame(index = rows, columns = columns)

for row in rows:
    for column in columns:
        ci = transition_df.loc[row, (column, 'ci')]
        mean = transition_df.loc[row, (column, 'mean')]
        ci = '{0:,.2f}'.format(ci)
        mean = '{0:,.2f}'.format(mean)
        string = str(mean) + ' +/- ' + str(ci)
        t3_df.loc[row, column] = string

lu_labels = {1: 'Native forest', 3: 'Plantation', 5: 'Shrub', 19: 'Agriculture', 'bio': 'Biodiversity', 
             'co2': '\makecell{Carbon \\\ sequestration}'} 
column_labels = {'sub': '\makecell{D.L. 701 \\\ baseline \\\ (S1)}', 
                 'ns': '\makecell{No subsidy \\\ counterfactual \\\ (S2)}', 
                 'rs': '\makecell{Perfect enforcement \\\ counterfactual \\\ (S3)}', 
                 'sub-ns': '\makecell{Impact of subsidy \\\ (S1 - S2)}', 
                 'rs-ns': '\makecell{Impact of perfectly \\\ enforced subsidy \\\ (S3 - S2)}', 
                 'lu_86': "Starting state (1986)",
                 'sub-lu_86': '\makecell{D.L. 701 \\\ baseline \\\ (S1)}',
                 'ns-lu_86': '\makecell{No subsidy \\\ counterfactual \\\ (S2)}',
                 'rs-lu_86': '\makecell{Perfect enforcement \\\ counterfactual \\\ (S3)}'}


t3_df = t3_df[['sub', 'ns', 'rs', 'sub-ns', 'rs-ns']]
chng_lbl = 'Transitions between 1986 and 2011'
dif_lbl = 'Difference between simulations'
clabels = pd.Series([chng_lbl, chng_lbl, chng_lbl, dif_lbl, dif_lbl],
                    index = ['sub', 'ns', 'rs', 'sub-ns', 'rs-ns'])
clabels.name = ("","")
t3_df = t3_df.append(clabels)
t3_df = t3_df.rename(index = lu_labels)
t3_df = t3_df.rename(columns = column_labels)
t3_df.index.names = ['Starting land use', 'Final land use']
t3_df = t3_df.T.set_index('', append=True).T.swaplevel(1, 0, axis = 1)

t3_df.to_latex(buf = results_dir + 't3_transitions.tex', multirow = True,
               column_format = "llccccc", escape = False,
               multicolumn = True, multicolumn_format = 'c')

t3_df.to_csv(results_dir + 't3_transitions.csv')


# =============================================================================
# LU impacts
# =============================================================================
sim_points = sim_dict['sub'].index
areas_mapped_86 = pd.value_counts(data_df.loc[data_df.index.isin(sim_points), 'lu_86']) * area_wght
areas_mapped_86 = areas_mapped_86.sort_index()

replicated_86 = pd.concat([data_df.loc[data_df.index.isin(sim_points), 'lu_86']]*n, axis = 1)
replicated_86.columns = [str(x) for x in np.arange(n)]
replicated_86 = replicated_86.astype(int)
sim_dict['lu_86'] = replicated_86

replicated_11 = pd.concat([data_df.loc[data_df.index.isin(sim_points), 'lu_11']]*n, axis = 1)
replicated_11.columns = [str(x) for x in np.arange(n)]
replicated_11 = replicated_11.astype(int)
sim_dict['lu_11'] = replicated_11

areas_dict = {key: df.apply(pd.value_counts) * area_wght for key, df in sim_dict.items()}

comparisons = [('rs', 'ns'),
               ('sub', 'ns'),
               ('rs', 'sub'),
               ('rs', 'lu_86'),
               ('sub', 'lu_86'),
               ('ns', 'lu_86'),
               ('lu_11', 'lu_86')]

for a, b in comparisons:
    areas_dict[(a + '-' + b)] = areas_dict[(a)] - areas_dict[(b)]

areas_stats = {key: pd.DataFrame(sim_stats(df.T)) for key, df in areas_dict.items()}
areas_df = pd.concat(areas_stats, axis = 1).sort_index()

# =============================================================================
# Biodiversity impacts
# =============================================================================
class estimate_bio:
    def __init__(self, bio_csv, n):
        esize_df = pd.read_csv(bio_csv, index_col = 0).T
        esize_df = esize_df.rename(index = {'pla': 3, 'mat': 5, 'ag': 19})
        esize_df = esize_df.reset_index().rename(columns = {'index': 'lu_to'})
        self.esize_df = esize_df.loc[esize_df['lu_to'].isin([3, 5, 19])].set_index('lu_to')
        self.effect_sizes = np.random.normal(self.esize_df['mean'], self.esize_df['sd'], (n, 3))
        
    def column_calc(self, col):
        col = col.loc[col.index.isin([1, 3, 5, 19])]
        proportions = col / col.sum()
        bio_estimate = proportions.loc[3] * self.effect_sizes[col.name][0] + \
            proportions.loc[5] * self.effect_sizes[col.name][1] + \
            proportions.loc[19] * self.effect_sizes[col.name][2]
        return bio_estimate

    def __call__(self, sim_df):
        sim_df = sim_df.rename(columns = lambda x: int(x))
        areas = sim_df.apply(pd.value_counts)
        bio_df = areas.apply(self.column_calc)
        return bio_df

bio_estimator = estimate_bio(bio_csv, n)
bio_dict = {key: bio_estimator(sim_df).dropna() for key, sim_df in sim_dict.items()}

for a, b in comparisons:
    bio_dict[(a + '-' + b)] = bio_dict[(a)] - bio_dict[(b)]
    
bio_stats = {key: pd.Series(sim_stats(df)) for key, df in bio_dict.items()}
bio_df = pd.DataFrame(pd.concat(bio_stats, axis = 0)).T.rename(index = {0: 'bio'})

# =============================================================================
# Create carbon impacts table
# =============================================================================
class estimate_co2:
    def __init__(self, co2_csv):
        co2_df = pd.read_csv(co2_csv)
        co2_df = co2_df.set_index('region')
#        co2_df = co2_df[['prad', 'mat', 'plant', 'bn']]
        co2_df = co2_df.rename(columns = {'bn': 1, 'mat': 5,
                                          'plant': 3, 'prad': 19})
        co2_df.index.name = 'region'
        co2_df.columns.name = 'lu_to'
        co2_df = co2_df.sort_index(axis = 1)
        co2_df = co2_df.sort_index(axis = 0)
        lu_dict = {1: 'Native forest', 3: 'Plantation', 5: 'Shrub', 
                   19: 'Agriculture'}
        co2_table = co2_df.rename(columns = lambda x: lu_dict[x])
        co2_table = co2_table.loc[[5,13,6,7,8,9,10,14]]
        self.co2_df = co2_df.transpose().unstack()

    def __call__(self, sim_df):
        sim_df = sim_df.rename(columns = lambda x: int(x))
        sim_df = sim_df.merge(data_df[['lu_86', 'region']], left_index = True, right_index = True, how = 'left')
        groups = sim_df.drop('lu_86', axis = 1).groupby(['region'])
        runs = [col for col in sim_df.columns if type(col) == int]
        region_df = pd.DataFrame({i: groups[i].value_counts() for i in runs}) * area_wght
        region_df.index.names = ['region', 'lu_to']
        co2_sim = region_df.apply(lambda x: x.mul(self.co2_df).sum(), axis = 0)
        return co2_sim # Thousand tonnes C

estimator = estimate_co2(co2_csv)
co2_dict = {key: estimator(sim_df) for key, sim_df in sim_dict.items()}

for a, b in comparisons:
    co2_dict[(a + '-' + b)] = co2_dict[(a)] - co2_dict[(b)]
    
co2_stats = {key: pd.Series(sim_stats(df)) for key, df in co2_dict.items()}
co2_df = pd.DataFrame(pd.concat(co2_stats, axis = 0)).T.rename(index = {0: 'co2'})

# =============================================================================
# Combine area, biodiversity and co2 data into single table (Table 2)
# =============================================================================
results_df = pd.concat([areas_df, co2_df, bio_df])

rows = results_df.index
columns = results_df.columns.get_level_values(0).unique()

t2_df = pd.DataFrame(index = rows, 
                        columns = columns)

for row in rows:
    for column in columns:
        ci = results_df.loc[row, (column, 'ci')]
        mean = results_df.loc[row, (column, 'mean')]
        ci = '{0:,.2f}'.format(ci)
        mean = '{0:,.2f}'.format(mean)
        string = str(mean) + ' +/- ' + str(ci)
        t2_df.loc[row, column] = string

row = 'bio'
for column in columns:
    ci = results_df.loc[row, (column, 'ci')]
    mean = results_df.loc[row, (column, 'mean')]
    ci = '{:.3g}'.format(ci)
    mean = '{:.3g}'.format(mean)
    string = str(mean) + ' +/- ' + str(ci)
    t2_df.loc[row, column] = string


t2_df = t2_df[['sub-lu_86', 'ns-lu_86', 'rs-lu_86', 'sub-ns', 'rs-ns']]
t2_df = t2_df.rename(index = lu_labels)
t2_df = t2_df.rename(columns = column_labels)
chng_lbl = 'Change between 1986 and 2011'
dif_lbl = 'Difference between simulations'
t2_df.loc[''] = [chng_lbl, chng_lbl, chng_lbl, dif_lbl, dif_lbl]
t2_df = t2_df.T.set_index('', append=True).T.swaplevel(1, 0, axis = 1)
units = ['\makecell{Thousand \\\ hectares}', 
         '\makecell{Thousand \\\ hectares}',
         '\makecell{Thousand \\\ hectares}',
         '\makecell{Thousand \\\ hectares}', 
         'Kilotonnes of carbon',
         '\makecell{Area-weighted, standardized \\\ species richness}']
t2_df['Units'] = units
t2_df = t2_df.set_index('Units', append=True)
t2_df.to_latex(buf = results_dir + 't2_summary.tex', multirow = True, 
               multicolumn = True, multicolumn_format = 'c', escape = False,
               column_format = "lcccccc")

t2_df.to_csv(results_dir + 't2_summary.csv')

# =============================================================================
# Summary table of regressors (Table S1)
# =============================================================================
agsum_df = data_df['ag_ev01'].append( data_df['ag_ev11']).describe()
plantsum_df = data_df['plant_ev01'].append( data_df['plant_ev11']).describe()
sum_df = data_df[['for_ev', 'north', 'central', 'south', 'luq1', 'luq2', 'luq3']].describe()
sum_df['ag_ev'] = agsum_df
sum_df['plant_ev'] = plantsum_df
sum_df = sum_df.T
currency_unit = 'Million 2005 CHP'
shr_unit = 'Indicator'
sum_df = sum_df.loc[['plant_ev', 'ag_ev', 'for_ev', 'north', 'central', 'south', 'luq1', 'luq2', 'luq3']]
sum_df['units'] = [currency_unit, currency_unit, currency_unit,
                   shr_unit, shr_unit, shr_unit, shr_unit, shr_unit, shr_unit]
sum_df = sum_df[['units', 'mean', 'std', 'min', 'max']]
idx_labels = ['Plantation rents', 'Agriculture rents',
              'Fuelwood rents', 'Located in northern regions', 'Located in central regions', 
              'Located in southern regions', 'High land capability', 
              'Moderate land capability', 'Low land capability']
sum_df.index = idx_labels
col_labels = ['Units', 'Mean', 'Standard deviation', 'Minimum', 'Maximum']
sum_df.columns = col_labels

sum_df.to_csv(results_dir + 's1_sumstats.csv', header = True)

# =============================================================================
# Validation (Table S4)
# =============================================================================
val_df = data_df.loc[data_df['oos']==1]
val_simulator = Simulator()
val_simulator.load_coefs(coefs_csv, cov_csv)
val_simulator.draw_coefs(1)
val_sim_dict = val_simulator.sim(val_df)
modeled = val_sim_dict['sub']
modeled = modeled.rename(columns = {0: 'sub'})
val_df = val_df.merge(modeled, left_index = True, right_index = True, how ='left')

groups = val_df.loc[val_df['oos']==1].groupby(['lu_86'])
observed = groups['lu_11'].value_counts()
validation_df = pd.DataFrame(index = observed.index, columns = ['transition_prob', 'pct_correct'])
for i in [1, 3, 5, 19]:
    for j in [1, 3, 5, 19]:
        validation_df.loc[(i, j), 'transition_prob'] = observed.loc[(i, j)] / observed[(i)].sum()

groups = val_df.loc[val_df['oos']==1].groupby(['lu_86', 'lu_11'])
simulated = groups['sub'].value_counts()
simulated.index.names = ['lu_86', 'lu_11', 'sim_11']
for i in [1, 3, 5, 19]:
    for j in [1, 3, 5, 19]:
        validation_df.loc[(i, j), 'pct_correct'] = simulated.loc[(i, j, j)] / simulated[(i, j)].sum()
       
validation_df['dif'] = (validation_df['pct_correct'] - validation_df['transition_prob']) / validation_df['transition_prob']
validation_df = validation_df.sort_index()      
validation_df.to_csv(results_dir + 's4_validation.csv', header = True)

val_df = val_df.rename(columns = {'lu_11': 'elu', 'lu_86': 'olu', 'sub': 'sim'})
kappa, kappa_sim, kappa_transition, kappa_translocation = \
    calc_kappa_sim(val_df)


for i in [1, 3, 5, 19]:
    validation_df.loc[(i, j), 'pct_correct'] = simulated.loc[(i, j, j)] / simulated[(i, j)].sum()

#==============================================================================
# Regression results with linear combinations (T1)
#==============================================================================
def clean_regression(esttab_csv):    
    reg_df = pd.read_csv(esttab_csv, 
                         engine = 'python', skipfooter=4)
    reg_df = reg_df.apply(lambda x: x.apply(lambda y: y.strip('"').strip('=').strip('"')))
    reg_df = reg_df.rename(columns = {'=\"\"': 'var', '=\"(1)\"': 'value'})    
    reg_df = reg_df.drop([0, 1], axis = 0)
    reg_df['var'] = reg_df['var'].replace('', np.nan)
    reg_df['var'] = reg_df['var'].fillna(method='ffill')
    reg_df['olu'] = reg_df['var'].apply(lambda x: x[:x.find('_')])
    reg_df['var'] = reg_df['var'].apply(lambda x: x[x.find('_')+1:])
    reg_df['stat'] = ['coef', 'std'] * int(np.shape(reg_df)[0]/2)
    reg_df['index'] = reg_df['var'] + '_' + reg_df['stat']
    reg_df = reg_df.pivot(index = 'index', columns = 'olu', values = 'value')
    reg_df = reg_df[['olu1', 'olu3', 'olu5', 'olu19']]
    orderlist = []
    for use in ['plant', 'ag', 'for']:
        for var in ['rent', 'dum']:
            for luq in ['luq1', 'luq2', 'luq3']:
                for stat in ['coef', 'std']:
                    orderlist.append(use + var + '_' + luq + '_' + stat)
        for region in ['central', 'south']:
            for stat in ['coef', 'std']:
                orderlist.append(use + 'dum_' + region + '_' + stat)
    
    reg_df = reg_df.loc[orderlist]
    reg_df = reg_df.rename(columns = {'olu1': 'Native forest', 'olu3': 'Plantation',
                                      'olu5': 'Shrub', 'olu19': 'Agriculture'})
    return(reg_df)

esttab_csv = data_dir + 'results_pool.csv'
reg_df = clean_regression(esttab_csv)
reg_df.to_csv(results_dir + 't1_estimation.csv', header = True)

# =============================================================================
# Robustness tables (S3)
# =============================================================================
## drop multiple observations from same property
esttab_csv = data_dir + 'results_propsample.csv'
reg_df = clean_regression(esttab_csv)
reg_df.to_csv(results_dir + 's3_robustness.csv', header = True)

#==============================================================================
# Paper results
#==============================================================================
results = {}

### Results - simulations
# Between 1986 and 2011, the subsidy scenario (S1) experienced A1 thousand hectares of plantation 
# expansion into areas formerly occupied by native forest (A2 thousand has), shrub (A3 thousand has) 
# and agriculture (A4 thousand has). 
sub_transitions = transition_df['sub']
mean = results_df.loc[3, ('sub-lu_86', 'mean')]
ci = results_df.loc[3, ('sub-lu_86', 'ci')]
results['A1_newplant_sub_area'] = '{0:.0f} +/- {1:.0f}'.format(mean, ci)

mean = sub_transitions.loc[(1,3), 'mean']
ci = sub_transitions.loc[(1,3), 'ci']
results['A2_for2plant_sub_area'] = '{0:.0f} +/- {1:.0f}'.format(mean, ci)

mean = sub_transitions.loc[(5,3), 'mean']
ci = sub_transitions.loc[(3,3), 'ci']
results['A3_shrub2plant_sub_area'] = '{0:.0f} +/- {1:.0f}'.format(mean, ci)

mean = sub_transitions.loc[(19,3), 'mean']
ci = sub_transitions.loc[(19,3), 'ci']
results['A4_ag2plant_sub_area'] = '{0:.0f} +/- {1:.0f}'.format(mean, ci)

# In comparison to the no subsidy counterfactual simulation (S2), the subsidy simulation (S1) 
# resulted in B1 thousand additional hectares of plantation forests. As a result, we estimate that 
# the subsidy was responsible for B2 percent of the expansion of plantation forests 
# between 1986 and 2011. 

dif = results_df.loc[3, ('sub-ns', 'mean')]
ci = results_df.loc[3, ('sub-ns', 'ci')]
results['B1_subimpact_plantarea'] = '{0:0.0f} +/- {1:0.0f}'.format(dif, ci)

shr = dif /  results_df.loc[3, ('sub-lu_86', 'mean')] * 100
results['B2_subimpact_plantpercent'] = '{0:0.2f}%'.format(shr)

#In contrast, the afforestation subsidies resulted in a C1 thousand hectare 
#reduction in the area of native forests. Individual forestry companies have 
#acknowledged larger areas of native forest to plantation conversion, but the 
#majority of these conversions would likely have occurred without government 
#subsidies. We find that C2 of the total forest loss that occurred between 
#1986 and 2011 was the result of government afforestation subsidies. Reductions 
#in the area of native forests were primarily due to direct conversion of native
#forests to plantations. However, subsidy-driven plantation establishment on shrub 
#and marginal agricultural lands prevented the re-establishment of C3 thousand 
#hectares of forests. 
dif = results_df.loc[1, ('sub-ns', 'mean')] * -1
ci = results_df.loc[1, ('sub-ns', 'ci')]
results['C1_subimpact_forarea'] = '{0:0.0f} +/- {1:0.0f}'.format(dif, ci)

shr = dif / results_df.loc[1, ('sub-lu_86', 'mean')] * 100 * -1
results['C2_subimpact_forpercent'] = '{0:0.2f}%'.format(shr)

dif = transition_df.loc[(5, 1), 'sub-ns'] + transition_df.loc[(19, 1), 'sub-ns']
mean = dif['mean'] * -1
ci = dif['ci']
results['C3_indirect_loss'] = '{0:0.0f} +/- {1:0.0f}'.format(mean, ci)

#In this scenario, plantations expanded by D1 thousand hectares, D2 thousand more hectares than 
#under the no subsidy scenario (S2)... Nevertheless, the restricted subsidy did still 
#generate an D3 thousand ha reduction in the total area of forests in 2011 
dif = results_df.loc[3, ('rs-lu_86', 'mean')]
ci = results_df.loc[3, ('rs-lu_86', 'ci')]
results['D1_newplant_rs_area'] = '{0:0.0f} +/- {1:0.0f}'.format(dif, ci)

dif = results_df.loc[3, ('rs-ns', 'mean')]
ci = results_df.loc[3, ('rs-ns', 'ci')]
results['D2_newplant_rsubimpact'] = '{0:0.0f} +/- {1:0.0f}'.format(dif, ci)

dif = results_df.loc[1, ('rs-ns', 'mean')] * -1
ci = results_df.loc[1, ('rs-ns', 'ci')]
results['D3_forarea_rsubimpact'] = '{0:0.0f} +/- {1:0.0f}'.format(dif, ci)

### Results - carbon impacts


#For example, the temperate Valdivian rainforests of the Los Rios region sequester 
#E1 tons of carbon per hectare while nearby plantations tend to sequester E2 
#tons of carbon per hectare.
#Two trends with considerable carbon impacts were evident over the study period: rapid 
#expansion of plantation forests (E3%) and net losses of native forests (E4%). 
co2 = estimator.co2_df
results['E1_forest_co2'] = '{0:0.0f}'.format(co2.loc[(14, 1)])
results['E2_plantation_co2'] = '{0:0.0f}'.format(co2.loc[(14, 3)])

change = results_df[('lu_11-lu_86', 'mean')]
pct_change = change / results_df[('lu_86', 'mean')] * 100
results['E3_plantation_chng'] = '{0:0.2f}%'.format(pct_change.loc[3])
results['E4_forest_chng'] = '{0:0.2f}%'.format(pct_change.loc[1])


#Between 1986 and 2011, the carbon sequestered in aboveground vegetation increased by F1 Tg C 
#(F2 percent)
results['F1_newC_t'] = '{0:,.0f}'.format(change.loc['co2'])
results['F2_newC_percent'] = '{0:0.2f}%'.format(pct_change.loc['co2'])

#As a result, afforestation subsidies decreased total carbon sequestration in 
#aboveground biomass by G1 TgC. In contrast, by prohibiting plantation 
#subsidies on previously forested lands, the restricted subsidy scenario achieved 
#a G2 TgC increase in carbon sequestration. 
dif = results_df.loc['co2', ('sub-ns', 'mean')] * -1
ci = results_df.loc['co2', ('sub-ns', 'ci')]
results['G1_sub_dif_C'] = '{0:0.0f} +/- {1:0.0f}'.format(dif, ci)

dif = results_df.loc['co2', ('rs-ns', 'mean')]
ci = results_df.loc['co2', ('rs-ns', 'ci')]
results['G2_rs_dif_C'] = '{0:0.0f} +/- {1:0.0f}'.format(dif, ci)

# Assuming a relatively modest social cost of carbon of $31 per ton of CO2, 
# we estimate that perfect enforcement of restrictions on native forest 
# conversion could have increased the carbon benefits of the subsidy 
# from G3 dollars to G4 dollars.
scc = 31 # dollars per ton CO2 as per nordhaus 2017
scc = scc * 1000 / 3.67 # convert to dollars per kTC
sub_scc = results_df.loc['co2', ('sub-ns', 'mean')] * scc / 1000000 # million usd
results['G3_sub_scc'] = '{0:0.2f}'.format(-sub_scc)
rs_scc = results_df.loc['co2', ('rs-ns', 'mean')] * scc / 1000000
results['G4_rs_scc'] = '{0:0.2f}'.format(rs_scc)

### Results - Biodiversity
#When aggregated through meta-analysis, previous ecological field studies show 
#that Chilean native forests have H1 standard deviations higher species 
#richness than paired plantation forests, and comparable species 
#richness to paired shrublands (H2 std deviations). Given observed diversity distributions from plots 
#of vascular plants, these differences indicate plantations have a mean species 
#richness that is H3 percent lower than comparable native forests.
bio_df = bio_estimator.esize_df
mean = bio_df.loc[(3,5,19), 'mean']
ci = bio_df.loc[(3,5,19), 'ci_lower']
ci = mean-ci

results['H1_biodif_plant'] = '{0:0.3f} +/- {1:0.3f}'.format(-mean[3], ci[3])

max_effect = mean-ci
min_effect = mean+ci

# Input mean and sd species richness for forest plots from phytosociological studies of vascular plants
for_studies_means = pd.Series([24.5, 26.315, 30.1, 14.4156, 7.68, 46.36, 31.3, 16.2, 9.72])
for_studies_sd = pd.Series([2.9, 5.42, 4.5, 4.857, 1.88, 11.084, 1.89, 1.46, 4.254])

max_mean_ratio = ((max_effect[3] * for_studies_sd) / for_studies_means).describe()['mean'] * -100
min_mean_ratio = ((min_effect[3] * for_studies_sd) / for_studies_means).describe()['mean'] * -100

results['H3_bioeffect'] = '{0:0.0f} to {1:0.0f}%'.format(min_mean_ratio, max_mean_ratio)


#Under our subsidy scenario, we find that the area-weighted, standardized 
#species richness in 2011 is I1 standard deviations. This can be interpreted 
#as indicating that the mean species richness of a randomly selected sample 
#of non-agricultural points from the landscape will have I2 standard deviations 
#fewer species than a comparable sample of forested points. In contrast, 
#simulations of our no-subsidy scenario yield an area-weighted, standardized 
#species richness of I3 standard deviations. Comparing these results, we 
#estimate that Chile’s afforestation subsidies decreased the area-weighted, 
#standardized species richness by I4 standard deviations. Based on biodiversity 
#distributions from phytosociological plots, this indicates that a randomly 
#selected sample of monitoring plots spanning Chile’s non-agricultural lands 
#would have a mean species richness with I5 fewer vascular plant species as a 
#result of Chile’s afforestation subsidies. In aggregate, afforestation subsidies 
# were responsible for I6 percent of the decline in species richness observed 
# between 1986 and 2011. In contrast, strongly enforced 
#restrictions on the availability of plantation subsidies for previously 
#forested lands can mitigate much of the subsidy’s negative biodiversity 
#impacts. We find that the restricted subsidy scenario reduces the area-weighted, 
#standardized species richness by I7, mitigating I8 percent of the biodiversity 
#loss resulting from the unrestricted subsidy.
bio_11 = results_df.loc['bio', 'sub']
results['I1_bio_11'] = '{0:0.3f} +/- {1:0.3f}'.format(bio_11['mean'], bio_11['ci'])
results['I2_bio_11_smpl'] = '{0:0.3f}'.format(-bio_11['mean'])
bio_change = results_df.loc['bio', 'sub-lu_86']
results['I3_bio_sub_change'] = '{0:0.3f} +/- {1:0.3f}'.format(bio_change['mean'], bio_change['ci'])
bio_ns_change = results_df.loc['bio', 'ns-lu_86']
results['I4_bio_ns_change'] = '{0:0.3f} +/- {1:0.3f}'.format(bio_ns_change['mean'], bio_ns_change['ci'])

bio_dif = results_df.loc['bio', 'sub-ns']
results['I5_bio_dif'] = '{0:0.2g} +/- {1:0.2g}'.format(-bio_dif['mean'], bio_dif['ci'])


#biostudy_df['mean_ratio'] = (bio_dif['mean'] * biostudy_df['sd_b']) / biostudy_df['mean_b']
#mean_ratio = (biostudy_df.loc[biostudy_df['taxon']=='p', 'mean_ratio']).describe()['mean']
##biostudy_df['min_mean_ratio'] = ((bio_dif['mean']+bio_dif['ci']) * biostudy_df['sd_b']) / biostudy_df['mean_b']
##biostudy_df['max_species_change'] = biostudy_df['max_mean_ratio'] * biostudy_df['mean_b']
##biostudy_df['min_species_change'] = biostudy_df['min_mean_ratio'] * biostudy_df['mean_b']
##species_change = (biostudy_df.loc[biostudy_df['taxon']=='p', 'species_change']*-1).describe()
##min_sp_change = (biostudy_df.loc[biostudy_df['taxon']=='p', 'min_mean_ratio']).describe()['max']
##max_sp_change = (biostudy_df.loc[biostudy_df['taxon']=='p', 'max_mean_ratio']).describe()['min']
##min_sp_change = (biostudy_df.loc[biostudy_df['taxon']=='p', 'min_species_change']).describe()['max']
##max_sp_change = (biostudy_df.loc[biostudy_df['taxon']=='p', 'max_species_change']).describe()['min']
#results['I5_species_effect'] = '{0:0.2f}%'.format(mean_ratio * -100)

bio_shr_impact = (results_df.loc['bio', 'sub-ns'] / results_df.loc['bio', 'sub-lu_86'])['mean'] * 100
results['I6_species_effect_pct'] = '{0:0.2f}%'.format(bio_shr_impact)


bio_rsdif = results_df.loc['bio', 'rs-sub']
results['I7_bio_rsdif'] = '{0:0.2g} +/- {1:0.2g}'.format(bio_rsdif['mean'], bio_rsdif['ci'])
dif_shr = (-bio_rsdif / bio_dif)['mean'] * 100
results['I8_bio_dif_shr'] = '{0:0.0f}%'.format(dif_shr)

## Conclusion
# and increased carbon sequestration by J1 TgC
dif = (results_df['rs-sub']).loc['co2', 'mean']
results['J1_rs-sub-co2'] = '{0:0.2f}'.format(dif)

# Filling in additional metric for land use section
nf_change = results_df.loc[1, ('sub-lu_86', 'mean')] * -1
results['C3_nf_change'] = '{0:0.0f}'.format(nf_change)


### Export results
output_df = pd.Series(results)
output_df.to_csv(results_dir + 'paper_results.csv', header = True)    
