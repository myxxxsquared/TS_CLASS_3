# TS_CLASS_3
# 
# Chang Zhaoxin, Su Rui, Zhang Wenjie
# 
# Preprocessing data

import time

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from itertools import product
from netsettings import time_predict

def load_data():
    print("load_data()")

    # load data from file
    print("load_data() : load from file")
    items = pd.read_csv('../input/items.csv.gz')
    shops = pd.read_csv('../input/shops.csv.gz')
    cats = pd.read_csv('../input/item_categories.csv.gz')
    train = pd.read_csv('../input/sales_train.csv.gz')
    test = pd.read_csv('../input/test.csv.gz').set_index('ID')

    # Outiers
    print("load_data() : outliers")
    train = train[train.item_price < 100000]
    train = train[train.item_cnt_day < 1001]
    median = train[(train.shop_id == 32) & (train.item_id == 2973) & (
        train.date_block_num == 4) & (train.item_price > 0)].item_price.median()
    train.loc[train.item_price < 0, 'item_price'] = median

    # Duplicates of shops
    print("load_data() : dup shops")
    train.loc[train.shop_id == 0, 'shop_id'] = 57
    test.loc[test.shop_id == 0, 'shop_id'] = 57
    train.loc[train.shop_id == 1, 'shop_id'] = 58
    test.loc[test.shop_id == 1, 'shop_id'] = 58
    train.loc[train.shop_id == 10, 'shop_id'] = 11
    test.loc[test.shop_id == 10, 'shop_id'] = 11

    # extract city of shop
    print("load_data() : extract shop")
    shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"',
              'shop_name'] = 'СергиевПосад ТЦ "7Я"'
    shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
    shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
    shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
    shops = shops[['shop_id', 'city_code']]

    # extract type and subtype of cat
    print("load_data() : extract type")
    cats['split'] = cats['item_category_name'].str.split('-')
    cats['type'] = cats['split'].map(lambda x: x[0].strip())
    cats['type_code'] = LabelEncoder().fit_transform(cats['type'])
    cats['subtype'] = cats['split'].map(
        lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
    cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])
    cats = cats[['item_category_id', 'type_code', 'subtype_code']]

    items.drop(['item_name'], axis=1, inplace=True)

    # monthly sales
    print("load_data() : monthly sales")
    matrix = []
    cols = ['date_block_num', 'shop_id', 'item_id']
    for i in range(time_predict):
        sales = train[train.date_block_num == i]
        matrix.append(np.array(list(
            product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype=np.int))
    matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
    matrix['date_block_num'] = matrix['date_block_num']
    matrix['shop_id'] = matrix['shop_id']
    matrix['item_id'] = matrix['item_id']
    matrix.sort_values(cols, inplace=True)
    group = train.groupby(['date_block_num', 'shop_id', 'item_id']).agg(
        {'item_cnt_day': ['sum'], 'item_price': ['mean']})
    group.columns = ['item_cnt_month', 'item_price']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=cols, how='left')

    matrix['item_cnt_month'] = (matrix['item_cnt_month']
                                .fillna(0)
                                .clip(0, 20)  # NB clip target here
                                .astype(np.int32)
                                )

    # print(len(matrix))

    # test data
    test['date_block_num'] = time_predict
    test['date_block_num'] = test['date_block_num']
    test['shop_id'] = test['shop_id']
    test['item_id'] = test['item_id']

    # merge all together
    print("load_data() : merge all")
    matrix = pd.concat([matrix, test], ignore_index=True,
                       sort=False, keys=cols)
    matrix.fillna(0, inplace=True)
    matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')
    matrix = pd.merge(matrix, items, on=['item_id'], how='left')
    matrix = pd.merge(matrix, cats, on=['item_category_id'], how='left')

    matrix['month'] = matrix['date_block_num'] % 12

    return matrix


def main():

    t0 = time.time()

    matrix = load_data()
    print("pickle matrix")
    matrix.to_hdf('matrix.h5', key='matrix')

    print("genmatrix: {:.03f}s".format(time.time() - t0))


if __name__ == '__main__':
    main()
