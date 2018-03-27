#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 21:04:55 2018

@author: root
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import utils
from model import Model, SimulateFunctionModel
import numpy as np

EPOCH = 150
batch_size = 200
train_data_size = 5000

#%%
mnist = input_data.read_data_sets('./MNIST_data', one_hot= True)
train_x = mnist.train.images[0:train_data_size]
train_y = mnist.train.labels[0:train_data_size]

random_order = np.arange(train_data_size)
np.random.shuffle(random_order)

graph = tf.Graph()
with graph.as_default():
    input = tf.placeholder(tf.float32, shape= (None, 784), name= 'input')
    labels = tf.placeholder(tf.float32, shape= (None, 10), name= 'labels')
    
    model = Model()
    logits = model(input)
    
    cross_entropy = tf.reduce_sum(-labels * utils.safe_log(logits))
    optimizer = tf.train.AdamOptimizer(learning_rate= 0.002)
    train_step = optimizer.minimize(cross_entropy)
    
    accuracy = tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            tf.arg_max(labels, dimension= 1),
                            tf.arg_max(logits, dimension= 1)
                        ),
                        tf.float32
                    )
                )

grad_record = []
loss_record = []

with tf.Session(graph= graph) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1, EPOCH + 1):
        np.random.shuffle(random_order)
        train_x = train_x[random_order]
        train_y = train_y[random_order]
        
        
        total_loss = 0
        for idx in range(train_data_size // batch_size):
            x = train_x[idx*batch_size : (idx+1) * batch_size]
            y = train_y[idx*batch_size : (idx+1) * batch_size]
            feed_dict = {input: x, labels: y}
            _, loss = sess.run([train_step, cross_entropy], feed_dict= feed_dict)
            total_loss += loss / batch_size
        
        feed_dict = {input: train_x, labels: train_y}
        train_acc = sess.run(accuracy, feed_dict= feed_dict)
        print('epoch:', epoch, ',loss:', total_loss, 'train_acc:', train_acc)
        
        grad_all = 0
        for variable in tf.trainable_variables():
            [grad] = sess.run(tf.gradients(ys= cross_entropy, xs = variable), feed_dict= feed_dict)
            grad_norm = np.sum(grad**2)
            grad_all += grad_norm
        grad_all = grad_all**0.5
        grad_record.append(grad_all)
        loss_record.append(total_loss)

#%%
import matplotlib.pyplot as plt
fig1= plt.figure(1)
plt.suptitle('MNIST')
plt.subplot(2, 1, 1)
plt.xlabel('Epoch')
plt.ylabel('Gradient')
xlabels = np.arange(1, EPOCH + 1)
plt.plot(xlabels, grad_record, linestyle= '-')

plt.subplot(2, 1, 2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
xlabels = np.arange(1, EPOCH + 1)
plt.plot(xlabels, loss_record, linestyle= '-')

fig1.savefig('Visualize Gradient Norm_MNIST')
plt.close()
#%%

def objective_function(x):
    return x**5 - x**4 + x**3 - x**2 + x - 1

train_x = np.random.normal(scale= 10, size= (train_data_size, 1))
train_y = objective_function(train_x)
random_order = np.arange(train_data_size)

graph = tf.Graph()
with graph.as_default():
    x_placeholder = tf.placeholder(tf.float32, shape= (None, 1), name= 'x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, shape= (None, 1), name= 'y_placeholder')
    
    model = SimulateFunctionModel()
    prediction = model(x_placeholder)
    
    mse_loss = tf.reduce_mean(tf.squared_difference(y_placeholder, prediction))
    optimizer = tf.train.AdamOptimizer(learning_rate= 0.02)
    train_step = optimizer.minimize(mse_loss)
    

grad_record = []
loss_record = []

with tf.Session(graph= graph) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1, EPOCH + 1):
        np.random.shuffle(random_order)
        train_x = train_x[random_order]
        train_y = train_y[random_order]
        
        
        total_loss = 0
        for idx in range(train_data_size // batch_size):
            x = train_x[idx*batch_size : (idx+1) * batch_size]
            y = train_y[idx*batch_size : (idx+1) * batch_size]
            feed_dict = {x_placeholder: x, y_placeholder: y}
            _, loss = sess.run([train_step, mse_loss], feed_dict= feed_dict)
            total_loss += loss / batch_size
        
        feed_dict = {x_placeholder: train_x, y_placeholder: train_y}
        print('epoch:', epoch, ',loss:', total_loss)
        
        grad_all = 0
        for variable in tf.trainable_variables():
            [grad] = sess.run(tf.gradients(ys= mse_loss, xs = variable), feed_dict= feed_dict)
            grad_norm = np.sum(grad**2)
            grad_all += grad_norm
        grad_all = grad_all**0.5
        grad_record.append(grad_all)
        loss_record.append(total_loss)

#%%
import matplotlib.pyplot as plt
fig2= plt.figure(2)
plt.suptitle('$x^{5}-x^{4}+x^{3}-x^{2}+x-1$')
plt.subplot(2, 1, 1)
plt.xlabel('Epoch')
plt.ylabel('Gradient')
xlabels = np.arange(1, EPOCH + 1)
plt.plot(xlabels, grad_record, linestyle= '-')

plt.subplot(2, 1, 2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
xlabels = np.arange(1, EPOCH + 1)
plt.plot(xlabels, loss_record, linestyle= '-')

fig2.savefig('Visualize Gradient Norm_Function')
plt.close()