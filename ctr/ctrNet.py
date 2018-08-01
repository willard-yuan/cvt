#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import division
import numpy as np, deepdish as dd, glob, tensorflow as tf, pickle, random


train_batch_size = 512

class Network(object):
    def __init__(self):
        self.d_img_feat = 1024
        self.d_img_hidden1 = 512
        self.d_img_hidden2 = 128
        self.d_user_feat = 128
        self.d_hidden1 = 256
        self.d_hidden2 = 128
        self.d_hidden3 = 64
        self.d_outputs = 2
        self.train_batch_size = train_batch_size
        self.baseRate = 0.0001
        self.global_step = 0
        self.decay_steps = 25000
        self.l2_reg = 0.008
        self.learning_rate_decay_factor = 0.33
        self.learn_rate = tf.train.exponential_decay(self.baseRate, self.global_step, self.decay_steps, self.learning_rate_decay_factor, True)
        self.l2_regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg)
        self.he_init = tf.contrib.layers.variance_scaling_initializer() #He initialization, 如果使用relu，则最好使用he initial

        with tf.name_scope("input"):
            self.training = tf.placeholder(dtype=tf.bool, shape=(), name='training')
            self.train_batch_samples = tf.placeholder(dtype=tf.float32, shape=(self.train_batch_size, self.d_user_feat + self.d_img_feat), name='train_batch_samples')
            self.train_batch_labels = tf.placeholder(dtype=tf.float32, shape=(self.train_batch_size, 2), name='train_batch_labels')

        with tf.name_scope("forward"):
            samples = self.train_batch_samples
            # 用户特征
            user_feats = samples[:, 0:self.d_user_feat]

            # test2
            user_feats = tf.layers.dense(user_feats, self.d_user_feat, kernel_initializer = self.he_init, kernel_regularizer = self.l2_regularizer, name='userhidden1_layer')
            user_feats = tf.layers.batch_normalization(user_feats, training=self.training)
            user_feats = tf.nn.relu(user_feats)
            # end

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
            
            # test
            hidden2 = tf.layers.batch_normalization(hidden2, training=self.training)
            hidden2 = tf.nn.relu(hidden2)
            hidden3 = tf.layers.dense(hidden2, self.d_hidden3, kernel_initializer = self.he_init, kernel_regularizer = self.l2_regularizer, name='fusion_hidden3_layer')
            # end
            

            self.scores = tf.layers.dense(hidden3, self.d_outputs, kernel_initializer = self.he_init, kernel_regularizer = self.l2_regularizer, name='output_layer')
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
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float")) # 每一个batch的准确率

