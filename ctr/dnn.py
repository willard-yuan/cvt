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
            return (None,None,None)
        self.batch_userphotoids = self.userphotoids[ith * train_batch_size:(ith + 1) * train_batch_size]
        batch_data = []
        batch_label = []
        batch_upids = []
        for userphotoid in self.batch_userphotoids:
            try:
                tmp_user_feat = self.userfeats_dict.get(userphotoid)
                tmp_img_feat = self.imgfeats_dict.get('p' + userphotoid.split('p')[1])
                tmp_label = int(self.labels.get(userphotoid))
                if tmp_user_feat is None or tmp_img_feat is None or (tmp_label != 0 and tmp_label != 1):
                    return (None,None,None)
                tmp_feat = tmp_user_feat + tmp_img_feat
            except:
                print 'fail to get batch data'
                return (None,None,None)
            batch_data.append(tmp_feat)
            batch_label.append(tmp_label)
            batch_upids.append(userphotoid)
        batch_data = np.asanyarray(batch_data)
        batch_label = np.asanyarray(batch_label)
        batch_label = np.eye(2)[batch_label]
        self.batch_userphotoids = []
        return (batch_data, batch_label, batch_upids)


class Network(object):
    def __init__(self):
        self.d_img_feat = 1024
        self.d_img_hidden1 = 512
        self.d_img_hidden2 = 128
        self.d_user_feat = 128
        self.d_hidden1 = 256
        self.d_hidden2 = 128
        self.d_outputs = 2
        self.train_batch_size = train_batch_size
        self.baseRate = 0.001
        self.global_step = 0
        self.decay_steps = 25000
        self.l2_reg = 0.01
        self.learning_rate_decay_factor = 0.33
        self.learn_rate = tf.train.exponential_decay(self.baseRate, self.global_step, self.decay_steps, self.learning_rate_decay_factor, True)
        self.l2_regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg)
        self.he_init = tf.contrib.layers.variance_scaling_initializer()

        with tf.name_scope("input"):
            self.training = tf.placeholder(dtype=tf.bool, shape=(), name='training')
            self.train_batch_samples = tf.placeholder(dtype=tf.float32, shape=(self.train_batch_size, self.d_user_feat + self.d_img_feat), name='train_batch_samples')
            self.train_batch_labels = tf.placeholder(dtype=tf.float32, shape=(self.train_batch_size, 2), name='train_batch_labels')

        with tf.name_scope("forward"):
            samples = self.train_batch_samples
            # 用户特征
            user_feats = samples[:, 0:self.d_user_feat]
            # 图像特征
            imgfeat_inputs = samples[:, self.d_user_feat:self.d_user_feat + self.d_img_feat]
            imghidden1 = tf.layers.dense(imgfeat_inputs, self.d_img_hidden1, kernel_initializer = self.he_init, kernel_regularizer = self.l2_regularizer, name='imghidden1_layer')
            imghidden1 = tf.layers.batch_normalization(imghidden1, training=self.training)
            imghidden1 = tf.nn.relu(imghidden1)
            imghidden2 = tf.layers.dense(imghidden1, self.d_img_hidden2, kernel_initializer = self.he_init, kernel_regularizer = self.l2_regularizer, name='imghidden2_layer')
            # 拼接起来
            fusion_feat = tf.concat([user_feats, imghidden2], 1, name='concat_layer')
            # BN+relu
            fusion_feat = tf.layers.batch_normalization(fusion_feat, training=self.training)
            fusion_feat = tf.nn.relu(fusion_feat)
            # fc1
            hidden1 = tf.layers.dense(fusion_feat, self.d_hidden1, kernel_initializer = self.he_init, kernel_regularizer = self.l2_regularizer, name='fusion_hidden1_layer')
            hidden1 = tf.layers.batch_normalization(hidden1, training=self.training)
            hidden1 = tf.nn.relu(hidden1)
            # fc2
            hidden2 = tf.layers.dense(hidden1, self.d_hidden2, kernel_initializer = self.he_init, kernel_regularizer = self.l2_regularizer, name='fusion_hidden2_layer')
            self.scores = tf.layers.dense(hidden2, self.d_outputs, kernel_initializer = self.he_init, kernel_regularizer = self.l2_regularizer, name='output_layer')
            probs_predct = tf.nn.softmax(self.scores, name="softmax_layer")
        with tf.name_scope("loss"):
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            cross_entropy_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.train_batch_labels))
            self.train_batch_loss = tf.add_n([cross_entropy_loss] + reg_losses)
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
    training_epochs = 100
    ckpt_path = './ckpt/ctr-model.ckpt'
    imgfeats_path = './imgfeats_data/*.pickle'
    userfeats_path = './usersfeat_data/*.h5'
    label_path = './labels.txt'
    Dator = Data(imgfeats_path, userfeats_path, label_path)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(device_count={"CPU": 32},
                                      allow_soft_placement=True, 
                                      log_device_placement=True)
        sess = tf.Session(config=session_conf)
        sess = tf.Session()
        with sess.as_default():
            # Initialize class and all variables
            DNN = Network()

            loss_summary = tf.summary.scalar("loss", DNN.train_batch_loss)
            accu_summary = tf.summary.scalar("accuracy", DNN.accuracy)
            train_summary_op = tf.summary.merge_all()
            train_summary_writer = tf.summary.FileWriter(logdir='./log', graph=sess.graph)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=5)
            
            sess.run(tf.global_variables_initializer())
            
            def train_step(epoch, ith, batch_data, batch_label, count):
                _, loss, prediction, accuracy, summaries = sess.run(
                                                                    fetches=[DNN.train_batch_optimizer, 
                                                                             DNN.train_batch_loss,
                                                                             DNN.correct_prediction, 
                                                                             DNN.accuracy,
                                                                             train_summary_op], 
                                                                    feed_dict={DNN.train_batch_samples: batch_data, 
                                                                             DNN.train_batch_labels: batch_label, 
                                                                             DNN.training: True})
                print 'epoch: %d, batch: %d, loss: %f, accuray: %f' %(epoch, ith, loss, accuracy)
                train_summary_writer.add_summary(summaries, count)
 
            count = 0
            for epoch in range(training_epochs):
                random.shuffle(Dator.userfeats_path)
                userfeats_files_parts = split_list(Dator.userfeats_path, data_parts)
                for userfeats_files_part in userfeats_files_parts:
                    Dator.loadUserData(userfeats_files_part)
                    for i in range(Dator.n_batch):
                        (batch_data, batch_label, _) = Dator.getbatch(i) # 获取每个batch
                        if batch_data is None or batch_label is None:
                            print "batch_data or batch_label is None"
                            continue
                        train_step(epoch, i, batch_data, batch_label, count)
                        count = count + 1
                    Dator.releaseUserData() # 释放数据
                # 保存模型
                if (epoch+1) % 10 == 0:
                    model_path = saver.save(sess, ckpt_path, global_step=epoch)
                    print("Model saved in file: %s" % model_path)
