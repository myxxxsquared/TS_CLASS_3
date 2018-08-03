# TS_CLASS_3
# 
# Chang Zhaoxin, Su Rui, Zhang Wenjie
# 
# Test RNN model

import gzip
import pickle
import os
import sys

import numpy as np

from tsnetwork import TSNetwork, genbatch

def test():
    with gzip.open('traindata.pkl.gz', 'rb') as pf:
        _, validdata = pickle.load(pf)
    with gzip.open('testdata.pkl.gz', 'rb') as pf:
        testdata = pickle.load(pf)

    validdata = genbatch(validdata, 4096, True) 
    testdata = genbatch(testdata, 4096, True)

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    network = TSNetwork("train")
    network.saver.restore(network.sess, "train/best/" + os.listdir("train/best/")[0].split('.')[0])

    valid_result = []
    for dat in validdata:
        valid_result.append(network.predict(dat))
    valid_result = np.concatenate(valid_result).clip(0, 20)

    test_result = []
    for dat in testdata:
        test_result.append(network.predict(dat))
    test_result = np.concatenate(test_result).clip(0, 20)

    pickle.dump(valid_result, open('net_train.pickle', 'wb'))
    pickle.dump(test_result, open('net_test.pickle', 'wb'))

if __name__ == '__main__':
    test()
