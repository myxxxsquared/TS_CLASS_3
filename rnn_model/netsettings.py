# TS_CLASS_3
# 
# Chang Zhaoxin, Su Rui, Zhang Wenjie
# 
# Select columns to use in neural network

singlefeatures = [
    'shop_id',
    'city_code',
    'item_category_id',
    'type_code',
    'subtype_code',
    'month'
]
singlefeatures_count = [
    60,
    31,
    84,
    20,
    65,
    12
]
iter_lag_features = [
    ['item_cnt_month', 'shop_id'],
    ['item_cnt_month', 'item_id'],
    ['item_cnt_month', 'item_category_id'],
    ['item_cnt_month', 'type_code'],
    ['item_cnt_month', 'city_code'],
    ['item_cnt_month', 'shop_id', 'item_category_id'],
    ['item_cnt_month', 'shop_id', 'type_code'],
    ['item_cnt_month', 'shop_id', 'item_id'],
    # ['item_price', 'shop_id', 'item_id'],
    # ['item_price', 'shop_id', 'item_category_id'],
    # ['item_price', 'item_category_id']
]
time_lag = 24
time_predict = 34
num_layers = 3
