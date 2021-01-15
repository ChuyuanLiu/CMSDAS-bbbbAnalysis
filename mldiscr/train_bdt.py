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

## open sig and bkg datasets

input_bkg  = '../background/data_3btag_with_weights_AR.root'
input_sig  = '../analysis/objects_gg_HH_bbbb_SM.root'

print "... opening input files"

arrs_bkg  = uproot.open(input_bkg)['bbbbTree']
arrs_sig  = uproot.open(input_sig)['bbbbTree']

## convert to dataframes
vars_training = ['H1_costhetaCM' , 'H1_m', 'H2_m', 'H1_pt', 'H2_pt', 'HH_m', 'HH_pt', 'H1_b1_m', 'H1_b2_m', 'H2_b1_m', 'H2_b2_m', 'H1_b1_pt', 'H1_b2_pt', 'H2_b1_pt', 'H2_b2_pt', 'H1_eta', 'H2_eta', 'HH_eta', 'H1_b1_eta', 'H1_b2_eta', 'H2_b1_eta','H2_b2_eta']

# extra variables needed for preselections
all_vars = vars_training + ['H1_m', 'H2_m', 'n_btag']
all_vars = list(set(all_vars))

print "... converting to pandas"

# data_bkg = arrs_bkg.pandas.df(all_vars + ['bkg_model_w'], entrystop = 100000)
# data_sig = arrs_sig.pandas.df(all_vars, entrystop = 100000)

data_bkg = arrs_bkg.pandas.df(all_vars + ['bkg_model_w'])
data_sig = arrs_sig.pandas.df(all_vars)# + ['btag_SF'])

print "... preselecting data"

## apply a selection on the datasets
data_bkg = data_bkg[data_bkg['n_btag'] == 3]
data_sig = data_sig[data_sig['n_btag'] >= 4]

# restrict training to the signal region
data_bkg['chi'] = np.sqrt( (data_bkg['H1_m']-125)*(data_bkg['H1_m']-125)+(data_bkg['H2_m']-120)*(data_bkg['H2_m']-120))
data_sig['chi'] = np.sqrt( (data_sig['H1_m']-125)*(data_sig['H1_m']-125)+(data_sig['H2_m']-120)*(data_sig['H2_m']-120))

data_bkg = data_bkg[data_bkg['chi'] < 30]
data_sig = data_sig[data_sig['chi'] < 30]


## for the signal, add a fake weight column
data_bkg['train_w'] = data_bkg['bkg_model_w']
data_bkg.drop('bkg_model_w', axis=1, inplace=True)
data_sig['train_w'] = 1 #trigger weight 

# normalise the sum of weights to unity
data_bkg['train_w'] = data_bkg['train_w'].multiply(1./data_bkg['train_w'].sum())
data_sig['train_w'] = data_sig['train_w'].multiply(1./data_sig['train_w'].sum())

## label the datasets as bkg (0) and signal (1)
data_bkg['target'] = 0
data_sig['target'] = 1

all_data = pd.concat([data_bkg, data_sig], axis=0, sort=False)

print '... preparing inputs for xgboost'

all_data_train, all_data_test = train_test_split(all_data, test_size=0.33, random_state=123456)

xg_reg = xgb.XGBClassifier(
    base_score        = 0.5,
    booster           = 'gbtree',
    colsample_bylevel = 1,
    colsample_bytree  = 1,
    gamma             = 0,
    learning_rate     = 0.3,
    max_delta_step    = 0,
    max_depth         = 3,
    min_child_weight  = 0.001,
    n_estimators      = 1200,
    n_jobs            = 5,
    nthread           = 10,
    objective         = 'binary:logistic',
    random_state      = 0,
    reg_alpha         = 0.0,
    reg_lambda        = 0.05,
    scale_pos_weight  = 1,
    seed              = 123456,
    silent            = True,
    subsample         = 1,
)

print '... fitting the BDT'
xg_reg.fit(all_data_train[vars_training], all_data_train['target'], sample_weight = all_data_train['train_w'])

#######  Compute ROC curve and ROC area
def plot_ROC_curve(y_true, y_pred):

    # fpr, tpr, _ = roc_curve(X_test[target].values, check)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc     = auc(fpr, tpr)

    print '... ROC AUC : ', roc_auc
    # print fpr
    # print tpr

    print '... drawing ROC curve'

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig("bdt_ROC.png")

y_true = all_data_test['target']
y_pred = xg_reg.predict_proba(all_data_test[vars_training])[:,1] ## prob to be of class '1'
plot_ROC_curve(y_true, y_pred)

def plot_classifier_output():

    print '... drawing discriminator output'

    sig_train = all_data_train[all_data_train.target == 1][vars_training]
    bkg_train = all_data_train[all_data_train.target == 0][vars_training]
    
    sig_test  = all_data_test[all_data_test.target == 1][vars_training]
    bkg_test  = all_data_test[all_data_test.target == 0][vars_training]

    ## these are already normalised to unity
    weights_sig_train = all_data_train[all_data_train.target == 1]['train_w']
    weights_bkg_train = all_data_train[all_data_train.target == 0]['train_w']    
    
    weights_sig_test  = all_data_test[all_data_test.target == 1]['train_w']
    weights_bkg_test  = all_data_test[all_data_test.target == 0]['train_w']

    ## but before the splitting, so re-normalise
    weights_sig_train = weights_sig_train.multiply(1./weights_sig_train.sum())
    weights_bkg_train = weights_bkg_train.multiply(1./weights_bkg_train.sum())
    
    weights_sig_test = weights_sig_test.multiply(1./weights_sig_test.sum())
    weights_bkg_test = weights_bkg_test.multiply(1./weights_bkg_test.sum())

    #if n_classses > 2 sig proba is the last one (in the way the code is written) 
    Y_pred_sig_train = xg_reg.predict_proba(sig_train)[:,xg_reg.n_classes_-1]
    Y_pred_bkg_train = xg_reg.predict_proba(bkg_train)[:,xg_reg.n_classes_-1]
    Y_pred_sig_test  = xg_reg.predict_proba(sig_test)[:,xg_reg.n_classes_-1]
    Y_pred_bkg_test  = xg_reg.predict_proba(bkg_test)[:,xg_reg.n_classes_-1]

    # This will be the min/max of our plots
    c_max = max(np.max(d) for d in np.concatenate([Y_pred_sig_train,Y_pred_bkg_train,Y_pred_sig_test,Y_pred_bkg_test]))
    c_min = min(np.min(d) for d in np.concatenate([Y_pred_sig_train,Y_pred_bkg_train,Y_pred_sig_test,Y_pred_bkg_test]))

    # Get histograms of the classifiers
    Histo_training_S = np.histogram(Y_pred_sig_train,bins=40,range=(c_min,c_max),weights=weights_sig_train)
    Histo_training_B = np.histogram(Y_pred_bkg_train,bins=40,range=(c_min,c_max),weights=weights_bkg_train)
    Histo_testing_S = np.histogram(Y_pred_sig_test,bins=40,range=(c_min,c_max),weights=weights_sig_test)
    Histo_testing_B = np.histogram(Y_pred_bkg_test,bins=40,range=(c_min,c_max),weights=weights_bkg_test)
    
    # Lets get the min/max of the Histograms
    AllHistos = [Histo_training_S,Histo_training_B,Histo_testing_S,Histo_testing_B]
    h_max     = max([histo[0].max() for histo in AllHistos])*1.2
    h_min     = min([histo[0].min() for histo in AllHistos])
    
    # Get the histogram properties (binning, widths, centers)
    bin_edges = Histo_training_S[1]
    bin_centers = ( bin_edges[:-1] + bin_edges[1:]  ) /2.
    bin_widths = (bin_edges[1:] - bin_edges[:-1])
    
    # To make error bar plots for the data, take the Poisson uncertainty sqrt(N)
    ErrorBar_testing_S = np.sqrt(Histo_testing_S[0]/Y_pred_sig_test.size)
    ErrorBar_testing_B = np.sqrt(Histo_testing_B[0]/Y_pred_bkg_test.size)
    
    plt.clf() 
    # Draw objects
    ax1 = plt.subplot(111)
    
    # Draw solid histograms for the training data
    ax1.bar(bin_centers,Histo_training_S[0],facecolor='blue',linewidth=0,width=bin_widths,label='S (Train)',alpha=0.5)
    ax1.bar(bin_centers,Histo_training_B[0],facecolor='red',linewidth=0,width=bin_widths,label='B (Train)',alpha=0.5)
    
    # # Draw error-bar histograms for the testing data
    ax1.errorbar(bin_centers, Histo_testing_S[0], yerr=ErrorBar_testing_S, xerr=None, ecolor='blue',c='blue',fmt='o',label='S (Test)')
    ax1.errorbar(bin_centers, Histo_testing_B[0], yerr=ErrorBar_testing_B, xerr=None, ecolor='red',c='red',fmt='o',label='B (Test)')
    
    # Make a colorful backdrop to show the clasification regions in red and blue
    ax1.axvspan(0.5, c_max, color='blue',alpha=0.08)
    ax1.axvspan(c_min,0.5, color='red',alpha=0.08)

    # Adjust the axis boundaries (just cosmetic)
    ax1.axis([c_min, c_max, h_min, h_max])

    # Make labels and title
    plt.title("Classification with scikit-learn")
    plt.xlabel("Classifier output")
    plt.ylabel("Normalized Yields")
    
    # Make legend with smalll font
    legend = ax1.legend(loc='upper center', shadow=True,ncol=2)
    for alabel in legend.get_texts():
        alabel.set_fontsize('small')

    # Save the result to png
    plt.savefig("bdt_score_output.png")

plot_classifier_output()
# plot_classifier_output(xg_reg, all_data_train[vars_training], all_data_test[vars_training], all_data_train['target'], all_data_test['target'])

## save the model as pickle
outname = 'bdt_training.pkl'
print '... pickling the output as', outname
outfile = open(outname, 'wb')

to_save = {
    'vars'  : vars_training,
    'model' : xg_reg,
}

pickle.dump (to_save, outfile)
outfile.close()
