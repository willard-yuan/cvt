#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf
from dnn_ctr_v2 import Data, split_list
import numpy as np, deepdish as dd, glob, pickle, random


data_parts = 1
imgfeats_path = './imgfeats_data/*.pickle'
userfeats_path = './userfeats_data/*.h5'
label_path = './labels.txt'
Dator = Data(imgfeats_path, userfeats_path, label_path)

random.shuffle(Dator.userfeats_path)
userfeats_files_parts = split_list(Dator.userfeats_path, data_parts)

sess = tf.Session()
saver = tf.train.import_meta_graph('./ckpt/ctr-model.ckpt-19.meta')
model_file = tf.train.latest_checkpoint('./ckpt/')
saver.restore(sess, model_file)

graph = tf.get_default_graph()
inputs = graph.get_tensor_by_name("input/train_batch_samples:0")
Y_true = graph.get_tensor_by_name("input/train_batch_labels:0")
training = graph.get_tensor_by_name("input/training:0")
Y_pred = graph.get_tensor_by_name("forward/output_layer/BiasAdd:0")

retfile = open('ret_ctr.txt', 'w')
for userfeats_files_part in userfeats_files_parts:
    Dator.loadUserData(userfeats_files_part)
    for i in range(Dator.n_batch):
        (batch_data, batch_label, batch_upids) = Dator.getbatch(i)
        if batch_data is None or batch_label is None or batch_upids is None:
            print "batch_data or batch_label is None"
            continue

        result = sess.run(Y_pred, feed_dict = {inputs: batch_data, Y_true: batch_label, training: False})         
        result = sess.run(tf.nn.softmax(result, 1))
        labels_pred = sess.run(tf.argmax(result, 1))
        labels_true = sess.run(tf.argmax(batch_label, 1))
        for j, upid in enumerate(batch_upids):
            print upid, labels_true[j], labels_pred[j], result[j, labels_pred[j]]
            retfile.write(str(upid) + ' ' + str(labels_true[j]) + ' ' + str(labels_pred[j]) + ' ' + str(result[j, labels_pred[j]]) + '\n')
