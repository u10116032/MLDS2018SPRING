#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 16:58:02 2018

@author: halley
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import Model
import numpy as np
import utils


train_data_size = 1000

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_x = mnist.train.images
train_y = mnist.train.labels

random_order = np.arange(train_x.shape[0])
np.random.shuffle(random_order)
train_x = train_x[random_order]
train_x = train_x[0:train_data_size]

train_y = train_y[random_order][0:train_data_size]

#%%
graph = tf.Graph()
with graph.as_default():
    input = tf.placeholder(tf.float32, (None, 784), 'input')
    labels = tf.placeholder(tf.float32, (None, 10), 'label')
    model = Model()
    logits = model(input)
    
    cross_entropy = tf.reduce_sum(- labels * utils.safe_log(logits))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.02)
    train_step = optimizer.minimize(cross_entropy)
    
    accuracy = tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                                tf.arg_max(logits,dimension= 1),
                                tf.arg_max(labels,dimension= 1)
                        ),
                        tf.float32
                    )
                )
                    
EPOCH = 30
batch_size = 200

weights_record = []
train_acc_record = []

random_order = np.arange(train_data_size)
for i in range(8):
    with tf.Session(graph= graph) as sess:
        sess.run(tf.global_variables_initializer())
        temp_weights_record = []
        temp_acc_record = []
        for epoch in range(1, EPOCH + 1):
            total_loss = 0
            for idx in range(train_data_size // batch_size):
                x = train_x[idx*batch_size : (idx+1) * batch_size]
                y = train_y[idx*batch_size : (idx+1) * batch_size]
                feed_dict = {input: x, labels: y}
                _, loss = sess.run([train_step, cross_entropy], feed_dict= feed_dict)
                total_loss += loss / batch_size
            
            
            feed_dict = {input: train_x, labels: train_y}
            train_acc = sess.run(accuracy, feed_dict= feed_dict)
            print('epoch:', epoch, ',loss:', loss, 'train_acc:', train_acc)
            
            np.random.shuffle(random_order)
            train_x = train_x[random_order]
            train_y = train_y[random_order]
            
            if epoch % 3 == 0:
                variables = np.asarray([])
                for v in tf.trainable_variables():
                    variables = np.concatenate((variables, sess.run(tf.reshape(v, [-1]))))
                temp_weights_record.append(variables)
                temp_acc_record.append(train_acc)
                
    weights_record.append(temp_weights_record)
    train_acc_record.append(temp_acc_record)
    
#%%
from numpy import linalg as LA
points_record = []
for i in range(8):
    weights = np.asarray(weights_record[i])
    S = (weights -  np.mean(weights, axis= 0)).transpose()
    U = np.dot(S, S.transpose())
    eig_val, eig_vec = LA.eigh(U)
    
    sort_order = np.sort(eig_val)[::-1]
    W = (eig_vec[sort_order])[:,2]
    
    points = np.dot(weights, W)
    points_record.append(points)
    
#%%
import matplotlib.pyplot as plt
colormap = plt.cm.gist_ncar 
colorst = [colormap(i) for i in np.linspace(0, 0.9,8)]

fig = plt.figure(1)
ax = fig.add_subplot(111)

for i in range(8):
    x, y = points_record[i].T
    ax.scatter(x, y, alpha=0.5, c= colorst[i])
    for idx in range(len(x)):
        ax.annotate(str(train_acc_record[i][idx]), (x[idx], y[idx]))








