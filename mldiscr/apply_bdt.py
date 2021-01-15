import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import uproot
from sklearn.metrics import roc_curve, auc
import matplotlib
from matplotlib import pyplot as plt
import pickle
import root_pandas
import os

infile = open('bdt_training.pkl', 'rb')
in_data = pickle.load(infile)
infile.close()
    
print '... opened model, variables are', in_data['vars']
model         = in_data['model']
vars_training = in_data['vars']

def save_bdt(input_path, output_path = ""):
    input_arrs = uproot.open(input_path)['bbbbTree']
    input_data = input_arrs.pandas.df(vars_training)
    
    print '... applying bdt score'
    pred = model.predict_proba(input_data)[:,-1]
    
    filename = input_path.split('/')[-1]
    filename = filename.replace('.root','_mva.root')
    
    print '... saving bdt score'
    output_arrs = input_arrs.pandas.df()
    output_arrs['chi'] = np.sqrt(np.square(output_arrs['H1_m'] - 125) + np.square(output_arrs['H2_m'] - 120))
    output_arrs['bdt_score'] = pred
    output_arrs.to_root(output_path + filename, key = 'bbbbTree')
    return output_path + filename

# output_tree is one of 'signal', 'background', 'data'
def plot_bdt(input_path, output_path, output_tree):
    vars_selection = ['n_btag', 'chi', 'bdt_score'] # trigger_sf
    if output_tree is 'background':
        vars_selection.append('bkg_model_w')
    input_data = uproot.open(input_path)['bbbbTree'].pandas.df(vars_selection)
    # n btag selection
    if 'background' not in output_tree:
        print '... making n btag selection for ' + output_tree
        input_data = input_data[input_data['n_btag'] >= 4]
    # signal region
    print '... making signal region selection for ' + output_tree
    input_data = input_data[input_data['chi'] < 30]
    print '... saving bdt hists'
    vars_saving = ['bdt_score']
    if 'background' in output_tree:
        vars_saving.append("bkg_model_w")
    output_data = input_data[vars_saving]
    output_data.to_root(output_path, key = output_tree, mode = 'a')

signal_path = '../analysis/objects_gg_HH_bbbb_SM.root'
signal_path = save_bdt(signal_path)

background_path = '../background/data_3btag_with_weights_AR.root'
background_path = save_bdt(background_path)

#signal_path = 'objects_gg_HH_bbbb_SM_mva.root'
#background_path = 'data_3btag_with_weights_AR_mva.root'

output_path = 'bdt_plots.root'
if os.path.isfile(output_path):
    os.remove(output_path)

plot_bdt(signal_path, output_path, 'signal')
plot_bdt(background_path, output_path, 'background')
