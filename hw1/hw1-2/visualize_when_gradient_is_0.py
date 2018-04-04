#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 17:18:57 2018

@author: halley
"""

import tensorflow as tf
import numpy as np
from numpy import linalg as LA
from model import SimulateFunctionModel

def objective_function(x):
    return x**2 + x - 1

train_data_size = 1000
batch_size= 100
EPOCH = 1000
gradient_threshold = 0.05

train_x = np.random.normal(scale= 10, size= (train_data_size, 1))
train_y = objective_function(train_x)

random_order = np.arange(train_data_size)
#%%
minimal_ratio_record = []
loss_record = []
for time in range(100):
    graph = tf.Graph()
    with graph.as_default():
        x_placeholder = tf.placeholder(tf.float32, (None, 1), name= 'x_placeholder')
        y_placeholder = tf.placeholder(tf.float32, (None, 1), name= 'y_placeholder')
        
        model = SimulateFunctionModel()
        prediction = model(x_placeholder)
        
        mse_loss = tf.reduce_mean(tf.squared_difference(y_placeholder, prediction))
        train_optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
        train_step = train_optimizer.minimize(mse_loss)
        
        
        total_grad_norm = tf.constant(0, dtype= tf.float32)
        for variable in tf.trainable_variables():
            [grad] = tf.gradients(ys= mse_loss, xs= variable)
            total_grad_norm += tf.reduce_sum(grad**2)
        total_grad_norm = tf.sqrt(total_grad_norm)
        grad_optimizer = tf.train.AdamOptimizer(learning_rate= 1)
        min_grad = grad_optimizer.minimize(total_grad_norm)
        
        writer = tf.summary.FileWriter("TensorBoard/", graph = graph)
    
    with tf.Session(graph= graph) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, EPOCH+1):
            np.random.shuffle(random_order)
            train_x = train_x[random_order]
            train_y = train_y[random_order]
        
            total_loss = 0
            for idx in range(train_data_size//batch_size):
                x = train_x[idx * batch_size : (idx+1) * batch_size]
                y = train_y[idx * batch_size : (idx+1) * batch_size]
                
                feed_dict= {x_placeholder: x, y_placeholder:y}
                _, loss= sess.run([train_step, mse_loss], feed_dict= feed_dict)
                
                total_loss += (loss / batch_size)
            print('epoch:', epoch, 'loss:', total_loss)
        loss_record.append(total_loss)
        # Find where gradient is 0
        while True:
            feed_dict= {x_placeholder: train_x, y_placeholder:train_y}
            _, gradient_norm = sess.run([min_grad, total_grad_norm], feed_dict= feed_dict)  
            print('gradient_norm:', gradient_norm)
            
            if gradient_norm <= gradient_threshold:
                break
            
        # Calculate minima ratio w.r.t eigen values which are positive
        eigen_values = np.asarray([])
        for variable in tf.trainable_variables():
            hess = sess.run(tf.hessians(mse_loss, variable), feed_dict= feed_dict)
            eigen_values = np.append(eigen_values, (LA.eigvals(hess).reshape(-1,)))
        
        minimal_ratio = np.sum(eigen_values > 0) / np.prod(eigen_values.shape)
        minimal_ratio_record.append(minimal_ratio)
        
#%%
import matplotlib.pyplot as plt
fig1 = plt.figure(1)
plt.title('What happens when gradient is 0?')
plt.ylabel('Loss')
plt.xlabel('Minima Ratio')

plt.scatter(minimal_ratio_record, loss_record)
        
fig1.savefig('Visualize When Gradient is 0.png')
        
