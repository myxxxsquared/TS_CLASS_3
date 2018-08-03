# TS_CLASS_3
# 
# Chang Zhaoxin, Su Rui, Zhang Wenjie
# 
# Train RNN model

import gzip
import random
import os
import pickle

from tsnetwork import TSNetwork

def train():
    with gzip.open('traindata.pkl.gz', 'rb') as pf:
        traindata = pickle.load(pf)

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    network = TSNetwork("train")

    traindata, validationdata = traindata

    network.fit(traindata, validationdata, 4096, 30, 1000)

if __name__ == '__main__':
    train()
