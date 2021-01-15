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
import ROOT

lumi = 35922

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
def selection(input_path, output_path, output_tree):
    vars_selection = ['n_btag', 'chi', 'bdt_score', 'trigger_SF', 'btag_SF', 'xs', 'norm_weight'] 
    if output_tree is 'background':
        vars_selection = ['n_btag', 'chi', 'bdt_score', 'bkg_model_w']
    if output_tree is 'data':
        vars_selection = ['n_btag', 'chi', 'bdt_score']
    input_data = uproot.open(input_path)['bbbbTree'].pandas.df(vars_selection)
    # n btag selection
    if output_tree is 'signal':
        print '... making n btag selection for ' + output_tree
        input_data = input_data[input_data['n_btag'] >= 4]
    # signal region
    print '... making signal region selection for ' + output_tree
    input_data = input_data[input_data['chi'] < 30]
    print '... saving selected events'
    vars_saving = ['bdt_score', 'xs', 'trigger_SF', 'btag_SF', 'norm_weight']
    if output_tree is 'background':
        vars_saving = ['bdt_score', 'bkg_model_w']
    if output_tree is 'data':
        vars_saving = ['bdt_score']
    output_data = input_data[vars_saving]
    output_data.to_root(output_path, key = output_tree, mode = 'a')


def make_hists(data, output_tree):
    print '... making ' + output_tree + ' histogram'
    hfile = ROOT.TFile(output_tree + '.root', 'recreate')
    hist = ROOT.TH1F('Histogram', output_tree, 20, 0, 1)
    data = data[output_tree].pandas.df()
    for index, row in data.iterrows():
        weight = 1
        if output_tree is 'signal':
            weight = row ['btag_SF'] * lumi * row['xs'] * row['norm_weight'] # * row['trigger_SF'] 
        if output_tree is 'background':
            weight = row['bkg_model_w']
        hist.Fill(row['bdt_score'], weight)
    print '... saving ' + output_tree + ' histogram'
    hfile.Write()
    
    
signal_path = '../analysis/objects_gg_HH_bbbb_SM.root'
signal_path = save_bdt(signal_path)

background_path = '../background/data_3btag_with_weights_AR.root'
background_path = save_bdt(background_path)

data_path = '../background/data_4btag.root'
data_path = save_bdt(data_path)

output_path = 'bdt_plots.root'
if os.path.isfile(output_path):
    os.remove(output_path)

selection(signal_path, output_path, 'signal')
selection(background_path, output_path, 'background')
selection(data_path, output_path, 'data')

data = uproot.open(output_path)
make_hists(data,'signal') 
#make_hists(data, 'background')
#make_hists(data, 'data')
