# TS_CLASS_3
# 
# Chang Zhaoxin, Su Rui, Zhang Wenjie
# 
# RNN Model

import random
import os
import sys
import time
import logging

import tensorflow as tf
import numpy as np

from tqdm import tqdm
from netsettings import (singlefeatures, singlefeatures_count,
                         iter_lag_features, time_lag, num_layers, time_predict)


def genbatch(data, batchsize=256, test=False):
    batch_data = []
    for i in range(0, len(data) - batchsize + 1, batchsize):
        batch_data.append(tuple(map(np.array, zip(*data[i:i+batchsize]))))
    if test:
        if i + batchsize != len(data):
            batch_data.append(tuple(map(np.array, zip(*data[i+batchsize:]))))
    return batch_data


class TSNetwork:

    def __init__(self, outputdir, isTest=False):

        self.is_test = isTest

        self.build_network()

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.global_step_inc = tf.assign_add(self.global_step, 1)

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True  # pylint: disable=E1101
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        self.outputdir = outputdir

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.166666)
        self.summary = tf.summary.FileWriter(os.path.join(outputdir, "train"))
        self.summary_eval = tf.summary.FileWriter(
            os.path.join(outputdir, "eval"))
        self.summary_avg = tf.summary.FileWriter(
            os.path.join(outputdir, "avg"))
        self.summary.add_graph(self.sess.graph)

        self.best_score = 100

        self.logger = logger = logging.getLogger('tsnetwork')
        logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(os.path.join(outputdir, 'train.log'))
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

    def build_network(self):
        with tf.variable_scope('tsnetwork'):
            self.ph_f = tf.placeholder(dtype=np.int32, shape=[
                                       None, len(singlefeatures)], name="ph_f")
            self.ph_l = tf.placeholder(dtype=np.float32, shape=[
                                       None, 24, len(iter_lag_features)], name="ph_l")
            self.ph_r = tf.placeholder(
                dtype=np.float32, shape=[None], name="ph_r")

            self.ph_dropout = tf.placeholder(
                dtype=np.float32, shape=[], name='ph_dropout')

            self.output = self.forward(self.ph_f, self.ph_l, self.ph_dropout)
            self.output_loss = self.loss(self.output, self.ph_r)
            self.optimizer = tf.train.AdamOptimizer()
            self.train_step = self.optimizer.minimize(self.output_loss)

    def forward(self, f, l, dropout):
        f = tf.split(f, len(singlefeatures), 1)
        f_embedded = []
        for inputlabel, fname, flength in zip(f, singlefeatures, singlefeatures_count):
            inputlabel = tf.squeeze(inputlabel, axis=[1])
            embedding = tf.get_variable('embedding_{}'.format(
                fname), shape=[flength, 32], dtype=tf.float32, initializer=tf.random_normal_initializer)
            f_embedded.append(tf.nn.embedding_lookup(embedding, inputlabel))
        f_embedded = tf.concat(f_embedded, 1)

        f_embedded = tf.nn.dropout(f_embedded, dropout)
        f_embedded = tf.layers.dense(
            f_embedded, 128, activation=tf.nn.relu, name="f_1")
        f_embedded = tf.nn.dropout(f_embedded, dropout)
        f_info = f_embedded = tf.layers.dense(
            f_embedded, 64, activation=tf.nn.relu, name="f_2")
        f_embedded = tf.nn.dropout(f_embedded, dropout)
        f_embedded = tf.layers.dense(
            f_embedded, 32 * num_layers, activation=tf.nn.relu, name="f_3")

        cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.GRUCell(32, dtype=tf.float32) for _ in range(num_layers)])

        hidden = tf.split(f_embedded, num_layers, 1)

        l = tf.split(l, 24, 1)
        for lt in l:
            lt = tf.squeeze(lt, [1])
            lt = tf.pad(lt, [[0, 0], [0, 32 - lt.shape[1]]])
            output, hidden = cell(lt, hidden)

        output = tf.layers.dense(output, 32, activation=tf.nn.relu, name="f_4")
        output = tf.concat([f_info, output], axis=1)
        output = tf.nn.dropout(output, dropout)
        output = tf.layers.dense(output, 16, activation=tf.nn.relu, name="f_5")
        output = tf.nn.dropout(output, dropout)
        output = tf.layers.dense(output, 16, activation=tf.nn.relu, name="f_6")
        output = tf.layers.dense(output, 1, name="f_7")
        output = tf.squeeze(output, [1])

        return output

    def loss(self, output, label):
        return tf.reduce_mean(tf.square(output - label))

    def train(self, data):
        f, l, r = data
        _, step = self.sess.run([self.global_step_inc, self.global_step])
        _, trainloss = self.sess.run([self.train_step, self.output_loss], feed_dict={
            self.ph_f: f, self.ph_l: l, self.ph_r: r, self.ph_dropout: np.float32(0.8)})
        return trainloss, step

    def predict(self, data):
        f, l, _ = data
        output = self.sess.run(self.output, feed_dict={
            self.ph_f: f, self.ph_l: l, self.ph_dropout: np.float32(1.0)})
        return output

    def eval(self, data):
        f, l, r = data
        loss = self.sess.run(self.output_loss, feed_dict={
            self.ph_f: f, self.ph_l: l, self.ph_r: r, self.ph_dropout: np.float32(1.0)})
        return loss

    def fit(self, traindata, validationdata, batchsize, epoch, validationstep):
        t0 = 0
        step = 0
        sumloss = 0.0

        validationdata = genbatch(validationdata, batchsize=batchsize)
        for i in range(epoch):
            self.logger.info("fit() epoch {}".format(i))
            random.shuffle(traindata)
            batch_train_data = genbatch(traindata, batchsize=batchsize)
            for dat in batch_train_data:
                if step % validationstep == 0:
                    t0 = time.time()
                    sumloss = 0.0
                loss, step = self.train(dat)
                self.summary.add_summary(tf.Summary(
                    value=[tf.Summary.Value(tag="loss", simple_value=np.sqrt(loss))]), step)
                sumloss += np.sqrt(loss)
                if step % validationstep == 0:
                    timeuse = (time.time() - t0) / validationstep

                    t0 = time.time()
                    validation_loss = np.average(
                        list(self.eval(x) for x in validationdata))
                    timevail = (time.time() - t0)

                    self.summary_avg.add_summary(tf.Summary(
                        value=[tf.Summary.Value(tag="loss", simple_value=sumloss / validationstep)]), step)
                    self.summary_eval.add_summary(tf.Summary(
                        value=[tf.Summary.Value(tag="loss", simple_value=np.sqrt(validation_loss))]), step)
                    self.saver.save(self.sess, os.path.join(
                        self.outputdir, "model"), step)
                    self.summary.flush()
                    self.summary_avg.flush()
                    self.summary_eval.flush()

                    best = False
                    if validation_loss < self.best_score:
                        self.best_score = validation_loss
                        bestdir = os.path.join(self.outputdir, 'best')
                        os.system("rm -rf {}".format(bestdir))
                        os.system("mkdir -p {}".format(bestdir))
                        os.system(
                            "cp {}-{}* {}".format(os.path.join(self.outputdir, "model"), step, bestdir))
                        best = True

                    self.logger.info("fit() : step {} {}".format(
                        step, "best" if best else ""))
                    self.logger.info(
                        "fit() : step time {:.03f}s".format(timeuse))
                    self.logger.info(
                        "fit() : vaild time {:.03f}s".format(timevail))
                    self.logger.info("fit() : loss {} {}".format(
                        sumloss / validationstep, np.sqrt(validation_loss)))
