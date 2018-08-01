#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf
import ctrNet
import numpy as np


class Extractor(object):
    def __init__(self):
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph('../model/ctr-model.ckpt-29.meta')
        self.model_file = tf.train.latest_checkpoint('../model/')
        self.saver.restore(self.sess, self.model_file)
        self.graph = tf.get_default_graph()
        self.inputs = self.graph.get_tensor_by_name("input/train_batch_samples:0")
        self.Y_true = self.graph.get_tensor_by_name("input/train_batch_labels:0")
        self.training = self.graph.get_tensor_by_name("input/training:0")
        self.Y_pred = self.graph.get_tensor_by_name("forward/imghidden2_layer/BiasAdd:0")
        self.d_dlfeat = 1024
        self.d_userfeat = 128
        self.user_feat = [0.0]*self.d_userfeat
        self.batchsize = 512
        self.d = self.d_userfeat + self.d_dlfeat
        self.batch_label = np.eye(2)[[0]*self.batchsize]

    def compute(self, data):
        if len(data) % self.d_dlfeat != 0:
            print "input size error"
            return None
        num_dlfeat = len(data)/self.d_dlfeat
        if num_dlfeat > self.batchsize:
            print "input size is bigger than batch size"
        batch_data = np.zeros((self.batchsize, self.d), dtype=np.float32)
        for i in range(num_dlfeat):
            batch_data[i, self.d_userfeat:self.d] = data[i*self.d_dlfeat:(i+1)*self.d_dlfeat]
        feats = self.sess.run(self.Y_pred, feed_dict = {self.inputs: batch_data, self.Y_true: self.batch_label, self.training: False})
        if num_dlfeat < self.batchsize:
            feats = feats[0:num_dlfeat, :]
        return feats


if __name__ == '__main__':

    import math
    extor = Extractor()
    embedding_file = open('../smalldata/1000_128d_feats.txt', 'w')
    feats_file = open('../smalldata/1000_1024d_feats.txt', 'r')
    content = feats_file.readlines()

    datas = []
    ids = []
    num_batchs = math.ceil(len(content)/512)
    count = 0
    for k, line in enumerate(content):
        line = line.strip().split(' ')
        id_ = line[1]
        tmp_img_feat = [float(value) for value in line[2:]]
        if len(tmp_img_feat) != 1024:
            continue
        datas.append(tmp_img_feat)
        ids.append(id_)
        if len(datas) == 512 or (k == (len(content)-1)):
            print "%dth patch" % count
            count = count + 1
            data = [item for sublist in datas for item in sublist]
            feats = extor.compute(data)
            for j, id_ in enumerate(ids):
                tmp_str = ''
                embedding_feat = list(feats[j, :])
                if len(embedding_feat) != 128:
                    continue
                embedding = [str(value) for value in embedding_feat]
                tmp_str = ' '.join(embedding)
                embedding_file.write(id_ + ' ' + tmp_str + '\n')
            datas = []
            ids = []

    if ((k+1)%10000 == 0):
        print 'processed %d' % (k+1)
