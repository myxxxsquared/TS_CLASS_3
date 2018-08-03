# TS_CLASS_3
# 
# Chang Zhaoxin, Su Rui, Zhang Wenjie
# Train window-based model using random forest regressor
#

import time
import sys
import gc
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

# load data and select columns

data = pd.read_pickle('data.pkl')

data = data[[
    'date_block_num',
    'shop_id',
    'item_id',
    'item_cnt_month',
    'city_code',
    'item_category_id',
    'type_code',
    'subtype_code',
    'item_cnt_month_lag_1',
    'item_cnt_month_lag_2',
    'item_cnt_month_lag_3',
    'item_cnt_month_lag_6',
    'item_cnt_month_lag_12',
    'date_avg_item_cnt_lag_1',
    'date_item_avg_item_cnt_lag_1',
    'date_item_avg_item_cnt_lag_2',
    'date_item_avg_item_cnt_lag_3',
    'date_item_avg_item_cnt_lag_6',
    'date_item_avg_item_cnt_lag_12',
    'date_shop_avg_item_cnt_lag_1',
    'date_shop_avg_item_cnt_lag_2',
    'date_shop_avg_item_cnt_lag_3',
    'date_shop_avg_item_cnt_lag_6',
    'date_shop_avg_item_cnt_lag_12',
    'date_cat_avg_item_cnt_lag_1',
    'date_shop_cat_avg_item_cnt_lag_1',
    #'date_shop_type_avg_item_cnt_lag_1',
    #'date_shop_subtype_avg_item_cnt_lag_1',
    'date_city_avg_item_cnt_lag_1',
    'date_item_city_avg_item_cnt_lag_1',
    #'date_type_avg_item_cnt_lag_1',
    #'date_subtype_avg_item_cnt_lag_1',
    'delta_price_lag',
    'month',
    'days',
    'item_shop_last_sale',
    'item_last_sale',
    'item_shop_first_sale',
    'item_first_sale',
]]


# generate train, validation and test data

X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

del data
gc.collect()

# random forest model

rf = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=30,
    max_features='sqrt',
    oob_score=True,
    n_jobs=20,
    verbose=100
)

rf.fit(X_train, Y_train)

with open('rf.pickle', 'wb') as f:
    pickle.dump(rf, f)


Y_pred = rf.predict(X_valid).clip(0, 20)
Y_test = rf.predict(X_test).clip(0, 20)

# save predictions for an ensemble
pickle.dump(Y_pred, open('rf_train.pickle', 'wb'))
pickle.dump(Y_test, open('rf_test.pickle', 'wb'))
