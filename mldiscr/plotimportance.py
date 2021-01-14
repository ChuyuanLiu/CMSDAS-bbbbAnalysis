import pickle
import pandas as pd
from matplotlib import pyplot as plt

infile   = open('bdt_training.pkl', 'rb')
in_data   = pickle.load(infile)
infile.close()

def get_xgb_imp(xgb):
    from numpy import array
    imp_dict = xgb.get_booster().get_score(importance_type='weight')
    total = array(imp_dict.values()).sum()
    return {k:float(v)/total for k,v in imp_dict.items()}

importance = get_xgb_imp(in_data['model'])
keys = list(importance.keys())
values = list(importance.values())

data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=True)
fig = data.plot(kind='barh').get_figure()
fig.savefig('importance.png')
print(importance)
