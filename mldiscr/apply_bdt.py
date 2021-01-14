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

infile = open('bdt_training.pkl', 'rb')
in_data = pickle.load(infile)
infile.close()
    
print '... opened model, variables are', in_data['vars']
model         = in_data['model']
vars_training = in_data['vars']

def apply_bdt(input_path, output_path = ""):
    input_arrs = uproot.open(input_path)['bbbbTree']
    input_data = input_arrs.pandas.df(vars_training)
    
    print '... applying bdt score'
    pred = model.predict_proba(input_data)[:,-1]
    
    filename = input_path.split('/')[-1]
    filename = filename.replace('.root','_mva.root')
    
    print '... saving bdt score'
    output_arrs = input_arrs.pandas.df()
    output_arrs["bdt_score"]=pred
    output_arrs.to_root(output_path + filename, key="bbbbTree")

input_path='../analysis/objects_gg_HH_bbbb_SM.root'
apply_bdt(input_path)
