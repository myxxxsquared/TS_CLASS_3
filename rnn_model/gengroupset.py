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


def groupby(matrix, lab, cols):
    print("groupby() : '{}' : {}".format(lab, cols))

    group = matrix.groupby(
        cols + ['date_block_num']).agg({lab: ['mean']})
    group.columns = ['meanval']
    group.reset_index(inplace=True)

    group = group.pivot_table(index=cols, columns='date_block_num', values='meanval', fill_value=0.0)
    assert(group.columns.tolist() == list(range(34)))

    result = {}
    for i, v in zip(group.index, group.values):
        result[i] = v

    return result


def groupdata(matrix):
    print("groupdata()")
    matrix = matrix[matrix['date_block_num'] < time_predict]

    groupset = []
    for lab, *f in iter_lag_features:
        groupset.append(groupby(matrix, lab, f))
    return groupset


def main():
    t0 = time.time()

    matrix = pd.read_hdf('matrix.h5', key='matrix')

    groupset = groupdata(matrix)
    print("pickle groupset")
    with gzip.open('groupset.pkl.gz', 'wb') as pf:
        pickle.dump(groupset, pf)

    print("gengroupset: {:.03f}s".format(time.time() - t0))

if __name__ == '__main__':
    main()
