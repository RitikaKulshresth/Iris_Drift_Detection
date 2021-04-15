import alibi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
#from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input

from alibi_detect.cd import ChiSquareDrift, TabularDrift
from alibi_detect.utils.saving import save_detector, load_detector


df = pd.read_csv('iris.csv')
#dropping irrelevant columns
df = df.drop(['Id'], axis = 1)

#print(df.dtypes)

#idx =  len(df)
col1 = (len(df.columns))
X ,Y= df.iloc[:,0:col1-1],df.iloc[:,-1:] 
#print(X.columns,Y.columns)


data  = df.drop(['Species'], axis = 1)
feature_names = data
# print('Feature Names' ,feature_names)
#print(data.dtypes)

for f in feature_names :
    if data[f].dtype == 'object':
       print('Categorical Features are present')
        #print(f)
    else:
        print('Categorical Features are not present')
        #print([f])


# only return data values
class_names = Y['Species'].unique()
#print(class_names)

target_Values = Y.values 
#print(target_Values)


rows = len(df.axes[0])
cols = len(df.axes[1])
#print(rows,cols)


n_ref = 25
n_test = 25

#print(X.values)
X=X.values
X_ref, X_t0, X_t1 = X[:n_ref], X[n_ref:n_ref + n_test], X[n_ref + n_test:n_ref + 2 * n_test]
X_ref.shape, X_t0.shape, X_t1.shape

print(X_ref.shape, X_t0.shape, X_t1.shape)


categories_per_feature = {0: None, 1: None, 2: None, 3: None}

cd = TabularDrift(X_ref, p_val=.05, categories_per_feature=categories_per_feature)


# filepath = 'my_path'  # change to directory where detector is saved
# save_detector(cd, filepath)
# cd = load_detector(filepath)
print(X_t0)
preds = cd.predict(X_t0)
labels = ['No!', 'Yes!']
print('Drift? {}'.format(labels[preds['data']['is_drift']]))



feature_names = ['SepalLengthCm',  'SepalWidthCm',  'PetalLengthCm',  'PetalWidthCm']
print(feature_names)

for f in range(cd.n_features):
    stat = 'K-S'
    fname = feature_names[f]
    stat_val, p_val = preds['data']['distance'][f], preds['data']['p_val'][f]
    print(f'{fname} -- {stat} {stat_val:.3f} -- p-value {p_val:.3f}')



print(preds['data']['threshold'])

fpreds = cd.predict(X_t0, drift_type='feature')

print(fpreds)

for f in range(cd.n_features):
    stat = 'K-S'
    fname = feature_names[f]
    is_drift = fpreds['data']['is_drift'][f]
    stat_val, p_val = fpreds['data']['distance'][f], fpreds['data']['p_val'][f]
    print(f'{fname} -- Drift? {labels[is_drift]} -- {stat} {stat_val:.3f} -- p-value {p_val:.3f}')



