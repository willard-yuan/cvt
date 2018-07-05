#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import division
import numpy as np, deepdish as dd, glob, tensorflow as tf, pickle, random


train_batch_size = 512


class Data(object):

    def __init__(self, imgfeats_path, userfeats_path, label_path):
        self.imgfeats_dict = {}
        self.userfeats_dict = {}
        self.labels = {}
        self.userphotoids = []
        self.imgfeats_path = glob.glob(imgfeats_path)
        self.userfeats_path = glob.glob(userfeats_path)
        self.batch_userphotoids = []
        self.loadImgData()
        self.loadLabel(label_path)
        self.n_batch = 0

    def loadImgData(self):
        self.imgfeats_path.sort()
        for i, imgfeat_path in enumerate(self.imgfeats_path):
            print 'loading image data feature: %s' % imgfeat_path
            handle = open(imgfeat_path, 'rb')
            imgdict0 = pickle.load(handle)
            self.imgfeats_dict = dict(self.imgfeats_dict, **imgdict0)
            handle.close()
        print 'finished loading'
        imgdict0 = None

    def loadUserData(self, tmp_userfeats_path):
        for i, userfeat_path in enumerate(tmp_userfeats_path):
            print 'loading user feature: %s' % userfeat_path
            userdict0 = dd.io.load(userfeat_path)
            self.userfeats_dict = dict(self.userfeats_dict, **userdict0)
        print 'finished loading'
        userdict0 = None
        self.userphotoids = self.userfeats_dict.keys()
        random.shuffle(self.userphotoids)
        self.n_batch = int(np.ceil(len(self.userphotoids)/train_batch_size))

    def loadLabel(self, label_path):
        print 'loading label ......'
        handle = open(label_path, 'r')
        content = handle.readlines()
        for line in content:
            line = line.strip().split(' ')
            self.labels[line[0]] = line[1]
        print 'finished loading'
        handle.close()

    def releaseUserData(self):
        self.userfeats_dict = {}
        self.userphotoids = []
        self.n_batch = 0

    def getbatch(self, ith):
        if (ith + 1) * train_batch_size > len(self.userphotoids):
            print 'iterate at the end of data, ignore the batch'
            return (None,None)
        self.batch_userphotoids = self.userphotoids[ith * train_batch_size:(ith + 1) * train_batch_size]
        batch_data = []
        batch_label = []
        for userphotoid in self.batch_userphotoids:
            try:
                tmp_user_feat = self.userfeats_dict.get(userphotoid)
                tmp_img_feat = self.imgfeats_dict.get('p' + userphotoid.split('p')[1])
                tmp_label = int(self.labels.get(userphotoid))
                if tmp_user_feat is None or tmp_img_feat is None or (tmp_label != 0 and tmp_label != 1):
                    return (None,None)
                tmp_feat = tmp_user_feat + tmp_img_feat
            except:
                print 'fail to get batch data'
                return (None,None)
            batch_data.append(tmp_feat)
            batch_label.append(tmp_label)
        batch_data = np.asanyarray(batch_data)
        batch_label = np.asanyarray(batch_label)
        batch_label = np.array([batch_label, -(batch_label - 1)]).T
        self.batch_userphotoids = []
        return (batch_data, batch_label)


class Network(object):
    def __init__(self):
        self.d_img_feat = 1024
        self.d_img_hidden = 128
        self.d_user_feat = 128
        self.d_hidden1 = 256
        self.d_hidden2 = 128
        self.d_hidden3 = 64
        self.d_outputs = 2
        self.train_batch_size = train_batch_size
        self.baseRate = 0.00001
        self.global_step = 0
        self.decay_steps = 25000
        self.learning_rate_decay_factor = 0.33
        self.learn_rate = tf.train.exponential_decay(self.baseRate, self.global_step, self.decay_steps, self.learning_rate_decay_factor, True)

        with tf.name_scope("input"):
            self.training = tf.placeholder(dtype=tf.bool, shape=(), name='training')
            self.train_batch_samples = tf.placeholder(dtype=tf.float32, shape=(self.train_batch_size, self.d_img_feat + self.d_user_feat), name='train_batch_samples')
            self.train_batch_labels = tf.placeholder(dtype=tf.float32, shape=(self.train_batch_size, 2), name='train_batch_labels')

        with tf.name_scope("forward"):
            samples = tf.layers.batch_normalization(self.train_batch_samples, training=self.training)
            samples = tf.nn.relu(samples)
            imgfeat_inputs = samples[:, 0:self.d_img_feat]
            imghidden = tf.layers.dense(imgfeat_inputs, self.d_img_hidden, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003), name='imgfeat_layer')
            user_feats = samples[:, self.d_img_feat:self.d_img_feat + self.d_user_feat]
            fusion_feat = tf.concat([imghidden, user_feats], 1,name='concat')
            bn = tf.layers.batch_normalization(fusion_feat, training=self.training)
            bn = tf.nn.relu(bn)
            hidden1 = tf.layers.dense(bn, self.d_hidden1, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003), name='hidden1')
            bn1 = tf.layers.batch_normalization(hidden1, training=self.training)
            bn1 = tf.nn.relu(bn1)
            hidden2 = tf.layers.dense(bn1, self.d_hidden2, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003), name='hidden2')
            bn2 = tf.layers.batch_normalization(hidden2, training=self.training)
            bn2 = tf.nn.relu(bn2)
            hidden3 = tf.layers.dense(bn2, self.d_hidden3, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003), name='hidden3')
            self.scores = tf.layers.dense(hidden3, self.d_outputs, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003), name='output')
        with tf.name_scope("loss"):
            self.train_batch_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels=self.train_batch_labels))
        with tf.name_scope('optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_batch_optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.train_batch_loss)
        with tf.name_scope("accuracy"):
            self.correct_prediction = tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.train_batch_labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] for i in range(wanted_parts) ]


if __name__ == '__main__':

    data_parts = 1 # 160
    training_epochs = 5000
    imgfeats_path = './imgfeats_data/*.pickle'
    userfeats_path = './usersfeat_data/*.h5'
    label_path = 'labels.txt'
    Dator = Data(imgfeats_path, userfeats_path, label_path)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        sess = tf.Session(config=session_conf)
        sess = tf.Session()
        with sess.as_default():
            # Initialize class and all variables
            DNN = Network()

            loss_summary = tf.summary.scalar("loss", DNN.train_batch_loss)
            acc_summary = tf.summary.scalar("accuracy", DNN.accuracy)
            train_summary_op = tf.summary.merge_all()
            train_summary_writer = tf.summary.FileWriter(logdir='./log', graph=sess.graph)

            init = tf.global_variables_initializer()
            sess.run(init)

            def train_step(epoch, ith, batch_data, batch_label, count):
                _, loss, prediction, accuracy, summaries = sess.run(fetches=[DNN.train_batch_optimizer,
                                                                    DNN.train_batch_loss,
                                                                    DNN.correct_prediction,
                                                                    DNN.accuracy, train_summary_op],
                                                                    feed_dict={DNN.train_batch_samples: batch_data,
                                                                    DNN.train_batch_labels: batch_label, DNN.training: True})
                print 'epoch: %d, batch: %d, loss: %f, accuray: %f' %(epoch, ith, loss, accuracy)
                train_summary_writer.add_summary(summaries, count)

            count = 0
            for epoch in range(training_epochs):
                random.shuffle(Dator.userfeats_path)
                userfeats_files_parts = split_list(Dator.userfeats_path, data_parts) #分块读取
                for userfeats_files_part in userfeats_files_parts:
                    Dator.loadUserData(userfeats_files_part) # 读入3个区块的行为数据
                    for i in range(Dator.n_batch):
                        (batch_data, batch_label) = Dator.getbatch(i) # 获取每个batch
                        if batch_data is None or batch_label is None:
                            print "batch_data or batch_label is None"
                            continue
                        train_step(epoch, i, batch_data, batch_label, count)
                        count = count + 1
                        # 保存模型
                        if i == (Dator.n_batch-1):
                            modelName = '%d-ctr-model' % epoch
                            #saver.save(sess, modelName, global_step=ith_batch)
                    Dator.releaseUserData() # 释放数据
