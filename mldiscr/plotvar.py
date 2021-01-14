import pandas as pd
import numpy as np
import uproot
from matplotlib import pyplot as plt
import pickle

## open sig and bkg datasets

input_bkg  = '../background/data_3btag_with_weights_AR.root'
input_sig  = '../analysis/objects_gg_HH_bbbb_SM.root'

print "... opening input files"

arrs_bkg  = uproot.open(input_bkg)['bbbbTree']
arrs_sig  = uproot.open(input_sig)['bbbbTree']

## convert to dataframes
vars_training = [ 'H1_m', 'H2_m', 'H1_pt', 'H2_pt', 'H1_eta', 'H2_eta', 'H1_phi', 'H2_phi', 'HH_eta', 'HH_phi', 'HH_m', 'HH_pt', 'H1_b1_eta', 'H1_b1_phi', 'H1_b2_eta', 'H1_b2_phi', 'H2_b1_eta', 'H2_b1_phi', 'H2_b2_eta', 'H2_b2_phi', 'H1_b1_m', 'H1_b2_m', 'H2_b1_m', 'H2_b2_m', 'H1_b1_pt', 'H1_b2_pt', 'H2_b1_pt', 'H2_b2_pt']

# extra variables needed for preselections
all_vars = vars_training + ['H1_m', 'H2_m', 'n_btag']
all_vars = list(set(all_vars))

print "... converting to pandas"

data_bkg = arrs_bkg.pandas.df(all_vars + ['bkg_model_w'], entrystop = 100000)
data_sig = arrs_sig.pandas.df(all_vars, entrystop = 100000)

#data_bkg = arrs_bkg.pandas.df(all_vars + ['bkg_model_w'])
#data_sig = arrs_sig.pandas.df(all_vars)

print "... preselecting data"

## apply a selection on the datasets
data_bkg = data_bkg[data_bkg['n_btag'] == 3]
data_sig = data_sig[data_sig['n_btag'] >= 4]

# restrict training to the signal region
data_bkg['chi'] = np.sqrt( (data_bkg['H1_m']-120)*(data_bkg['H1_m']-120)+(data_bkg['H2_m']-110)*(data_bkg['H2_m']-110))
data_sig['chi'] = np.sqrt( (data_sig['H1_m']-120)*(data_sig['H1_m']-120)+(data_sig['H2_m']-110)*(data_sig['H2_m']-110))

data_bkg = data_bkg[data_bkg['chi'] < 30]
data_sig = data_sig[data_sig['chi'] < 30]

print "... making plot"
for var in vars_training:
    print "\t Plotting var ", var
    plt.figure()
    plt.hist(data_sig[var], bins=50, label='signal', color='red', normed=True, histtype='step')
    plt.hist(data_bkg[var], bins=50, label= 'background', color='black', normed=True, histtype='step')
    plt.legend()
    plt.xlabel(var)
    plt.savefig(var+".png")
