#!/bin/sh

# TS_CLASS_3
# Chang Zhaoxin, Su Rui, Zhang Wenjie
# Train model and get predection

cd window_based_mode
python generate_traindata.py
python train_model.py
cd ..

cd rnn_model
python genmatrix.py
python gengroupset.py
python gentraindata.py
python train.py
python test.py
cd ..

mv window_based_mode/*.pickle ..
mv rnn_model/*.pickle ..

python ensemble.py

