# TS_CLASS_3
# 
# Chang Zhaoxin, Su Rui, Zhang Wenjie
# 
# Preprocessing data

import gzip
import pickle
import time

import numpy as np
import pandas as pd

from tqdm import tqdm
from netsettings import (singlefeatures, singlefeatures_count,
                         iter_lag_features, time_lag, num_layers, time_predict)


def gendata(matrix, groupset):
    print("gendata()")

    data = []

    for row in tqdm(matrix.itertuples(), "gendata() : processing", total=len(matrix)):
        date_block_num = row.date_block_num
        data_item_f = np.array(list(getattr(row, f)
                                    for f in singlefeatures), dtype=np.int32)
        data_item_l = []
        for i, (lab, *f) in enumerate(iter_lag_features):
            index = tuple(getattr(row, x) for x in f)
            index = index[0] if len(index) == 1 else index

            dat = groupset[i]
            if index in dat:
                data_item_l.append(dat[index][date_block_num -
                                                        time_lag: date_block_num])
            else:
                data_item_l.append(np.zeros([time_lag], dtype=np.float32))
                # print("no")
        data_item_l = np.transpose(
            np.array(data_item_l, dtype=np.float32), [1, 0])
        data_item_label = np.float32(row.item_cnt_month)

        data.append((data_item_f, data_item_l, data_item_label))
    return data


def main():
    t0 = time.time()

    matrix = pd.read_hdf('matrix.h5', key='matrix')

    with gzip.open('groupset.pkl.gz', 'rb') as pf:
        groupset = pickle.load(pf)

    traindata = gendata(matrix[(matrix['date_block_num'] >= time_lag) & (
        matrix['date_block_num'] < time_predict - 1)], groupset)
    validationdata = gendata(
        matrix[matrix['date_block_num'] == time_predict - 1], groupset)
    print("pickle traindata")
    with gzip.open('traindata.pkl.gz', 'wb') as pf:
        pickle.dump((traindata, validationdata), pf)

    testdata = gendata(
        matrix[matrix['date_block_num'] == time_predict], groupset)
    print("pickle testdata")
    with gzip.open('testdata.pkl.gz', 'wb') as pf:
        pickle.dump(testdata, pf)

    print("gentraindata: {:.03f}s".format(time.time() - t0))

if __name__ == '__main__':
    main()
