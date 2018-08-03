# TS_CLASS_3
# 
# Chang Zhaoxin, Su Rui, Zhang Wenjie
# 
# Ensemble two models to get a better performance

import pickle
import pandas as pd
import numpy as np

# models to ensemble
models = ['rf', 'net']

# load data
data = pd.read_pickle('data.pkl')
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = np.expand_dims(
    data[data.date_block_num == 33]['item_cnt_month'].values, 0)
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
X_valid = X_valid.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

# load vaild pred for models
vaild_data = []
for modelname in models:
    curdata = pickle.load(
        open(modelname + '_train.pickle', 'rb'))
    vaild_data.append(np.expand_dims(curdata, 0))
    X_valid = pd.concat([X_valid, pd.Series(
        curdata.astype(np.float32)).rename(modelname + '_predict')], axis=1)
vaild_data = np.concatenate(vaild_data, axis=0)

# load test data for models
Y_test = []
for modelname in models:
    curdata = pickle.load(
        open(modelname + '_test.pickle', 'rb'))
    Y_test.append(np.expand_dims(curdata, 0))
    X_test = pd.concat([X_test, pd.Series(curdata.astype(
        np.float32)).rename(modelname + '_predict')], axis=1)
Y_test = np.concatenate(Y_test, axis=0)
Y_test = np.transpose(Y_test, [1, 0])

# generate minimum of each item
vaild_data = np.abs(vaild_data - Y_valid)
Y_valid = np.argmin(vaild_data, axis=0)

# train a model to determine which model to use
from sklearn.ensemble import RandomForestClassifier

X_valid = X_valid.fillna(0)
X_test = X_test.fillna(0)

rf = RandomForestClassifier(
    n_jobs=-1,
    n_estimators=100,
    verbose=100
)
rf.fit(X_valid, Y_valid)

# generate final output

CLASS_test = rf.predict(X_test)

vecs = [
    np.array([1, 0], dtype=np.float32),
    np.array([0.9, 0.1], dtype=np.float32)
]

Y_test = np.array(
    [np.sum(value * vecs[int(label)])
     for label, value in zip(CLASS_test, Y_test)]
)

submission = pd.DataFrame({
    "ID": list(range(len(Y_test))),
    "item_cnt_month": Y_test
})
submission.to_csv('ensemble_submission.csv', index=False)
