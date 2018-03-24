# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 14:43:06 2018

@author: Halley
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from deep import Deep
from shallow import Shallow
import os

def safe_log(x, eps= 1e-12):
    return tf.log( x + eps)

EPOCH = 1
batch_size= 500

mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
train_x = np.asarray([np.reshape(image, (28,28,1)) for image in mnist.train.images])
train_y = mnist.train.labels

loss_record = {}
acc_record = {}
#%%
graph = tf.Graph()
with graph.as_default():
    x_placeholder = tf.placeholder(tf.float32, (None, 28, 28, 1), 'x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, (None, 10), 'y_placeholder')
    
    deep = Deep()
    logits = deep(x_placeholder)
    cross_entropy = tf.reduce_sum(- y_placeholder * safe_log(logits))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1= 0.5)
    train_step = optimizer.minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_placeholder, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session(graph= graph) as sess:
    sess.run(tf.global_variables_initializer())
      
    print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    
    temp_loss_record = []
    temp_acc_record = []
    for epoch in range(1, EPOCH + 1):
        total_loss = 0
        #for idx in range(train_x.shape[0] // batch_size):
            #x = train_x[idx * batch_size: (idx + 1) * batch_size]
            #y = train_y[idx * batch_size: (idx + 1) * batch_size]
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = np.asarray([np.reshape(image, (28,28,1)) for image in batch_x])
        feed_dict = {x_placeholder:batch_x, y_placeholder:batch_y}
        _, loss, train_acc = sess.run([train_step, cross_entropy, acc], feed_dict=feed_dict)
        total_loss += loss / batch_size
        temp_loss_record.append(total_loss)
        temp_acc_record.append(train_acc)
        print('epoch:',epoch,',loss:',total_loss)
    saver = tf.train.Saver()
    save_path = './Deep_model/'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    saver.save(sess, save_path + 'deep.ckpt')    
    loss_record['deep'] = temp_loss_record
    acc_record['deep'] = temp_acc_record   
#%%
graph = tf.Graph()
with graph.as_default():
    x_placeholder = tf.placeholder(tf.float32, (None, 28, 28, 1), 'x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, (None, 10), 'y_placeholder')
    
    shallow = Shallow()
    logits = shallow(x_placeholder)
    cross_entropy = tf.reduce_sum(- y_placeholder * safe_log(logits))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1= 0.5)
    train_step = optimizer.minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_placeholder, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session(graph= graph) as sess:
    sess.run(tf.global_variables_initializer())
      
    print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    
    temp_loss_record = []
    temp_acc_record = []
    for epoch in range(1, EPOCH + 1):
        total_loss = 0
        #for idx in range(train_x.shape[0] // batch_size):
            #x = train_x[idx * batch_size: (idx + 1) * batch_size]
            #y = train_y[idx * batch_size: (idx + 1) * batch_size]
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = np.asarray([np.reshape(image, (28,28,1)) for image in batch_x])
        feed_dict = {x_placeholder:batch_x, y_placeholder:batch_y}
        _, loss, train_acc = sess.run([train_step, cross_entropy, acc], feed_dict=feed_dict)
        total_loss += loss / batch_size
        temp_loss_record.append(total_loss)
        temp_acc_record.append(train_acc)
        print('epoch:',epoch,',loss:',total_loss)
    saver = tf.train.Saver()
    save_path = './Shallow_model/'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    saver.save(sess, save_path + 'shallow.ckpt')    
    loss_record['shallow'] = temp_loss_record
    acc_record['shallow'] = temp_acc_record

#%%
import matplotlib.pyplot as plt
x_axis = np.arange(EPOCH) + 1
plt.figure(1)
plt.title('Loss Comparison')
plt.plot(x_axis, loss_record['shallow'])
plt.plot(x_axis, loss_record['deep'])
plt.xlabel('Epoch')
plt.ylabel('Cross entropy')
plt.legend(['shallow', 'deep'], loc= 'upper right')
plt.savefig('ggg.jpg')      

#%%
x_axis = np.arange(EPOCH) + 1
plt.figure(2)
plt.title('Accuracy Comparison')
plt.plot(x_axis, acc_record['shallow'])
plt.plot(x_axis, acc_record['deep'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['shallow', 'deep'], loc= 'upper left')
plt.savefig('ggg.jpg')    