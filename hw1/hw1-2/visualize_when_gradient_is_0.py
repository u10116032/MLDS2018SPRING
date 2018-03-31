#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 17:18:57 2018

@author: halley
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from numpy import linalg as LA
import utils
from model import Model

train_data_size = 1000
batch_size= 200
EPOCH = 200
gradient_threshold = 1e-5

mnist = input_data.read_data_sets('MNIST_data/', one_hot= True)
train_x = mnist.train.images[0:train_data_size]
train_y = mnist.train.labels[0:train_data_size]

random_order = np.arange(train_data_size)
#%%
graph = tf.Graph()
with graph.as_default():
    x_placeholder = tf.placeholder(tf.float32, (None, 784), name= 'x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, (None, 10), name= 'y_placeholder')
    
    model = Model()
    logits = model(x_placeholder)
    
    cross_entropy = tf.reduce_sum(-1 * y_placeholder * utils.safe_log(logits))
    train_optimizer = tf.train.AdamOptimizer(learning_rate=0.002)
    train_step = train_optimizer.minimize(cross_entropy)
    
    
    total_grad_norm = tf.constant(0, dtype= tf.float32)
    for variable in tf.trainable_variables():
        [grad] = tf.gradients(ys= logits, xs= variable)
        total_grad_norm += tf.reduce_sum(grad**2)
    total_grad_norm = tf.sqrt(total_grad_norm)
    grad_optimizer = tf.train.AdamOptimizer(learning_rate= 0.002)
    min_grad = grad_optimizer.minimize(total_grad_norm)
    

    accuracy = tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            tf.arg_max(y_placeholder, dimension= 1),
                            tf.arg_max(logits, dimension= 1)
                        ),
                        tf.float32
                    )
                )
    writer = tf.summary.FileWriter("TensorBoard/", graph = graph)


with tf.Session(graph= graph) as sess:
    sess.run(tf.global_variables_initializer())
    
    minimal_ratio_record = []
    loss_record = []
    for epoch in range(1, EPOCH+1):
        np.random.shuffle(random_order)
        train_x = train_x[random_order]
        train_y = train_y[random_order]
    
        total_loss = 0
        for idx in range(train_data_size//batch_size):
            x = train_x[idx * batch_size : (idx+1) * batch_size]
            y = train_y[idx * batch_size : (idx+1) * batch_size]
            
            feed_dict= {x_placeholder: x, y_placeholder:y}
            _, loss= sess.run([train_step, cross_entropy], feed_dict= feed_dict)
            
            total_loss += loss / batch_size
        feed_dict= {x_placeholder: train_x, y_placeholder:train_y}
        train_acc, current_grad = sess.run([accuracy, total_grad_norm], feed_dict= feed_dict)
        loss_record.append(total_loss)
        print('epoch:', epoch, 'loss:', total_loss, 'train_acc:', train_acc)

        # Find where gradient is 0
        while True:
            feed_dict= {x_placeholder: train_x, y_placeholder:train_y}
            _, gradient_norm = sess.run([min_grad, total_grad_norm], feed_dict= feed_dict)
            
            print('gradient_norm:', gradient_norm)
            
            if gradient_norm <= gradient_threshold:
                break
        # Calculate minima ratio w.r.t eigen values which are positive
        eigen_values = []
        for variable in tf.trainable_variables():
            hess = sess.run(tf.hessians(cross_entropy, variable), feed_dict= feed_dict)
            eigen_values.append(LA.eigvals(hess))
        eigen_values = np.asarray(eigen_values).reshape((-1,))
        minimal_ratio = np.sum(eigen_values > 0) / np.len(eigen_values)
        minimal_ratio_record.append(minimal_ratio)
        
#%%
import matplotlib.pyplot as plt
fig1 = plt.figure(1)
plt.title('What happens when gradient is 0?')
plt.ylabel('Loss')
plt.xlabel('Minima Ratio')

plt.scatter(minimal_ratio_record, loss_record)
        
fig1.savefig('Visualize When Gradient is 0.png')
        
