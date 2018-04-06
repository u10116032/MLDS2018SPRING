# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 00:12:01 2018

@author: Halley
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import utils
from models import SimpleDNN
import numpy as np

train_dataset_counts = 55000

mnist = input_data.read_data_sets('./MNIST_data', one_hot= True)
train_x = mnist.train.images[0:train_dataset_counts, :]
train_y = mnist.train.labels[0:train_dataset_counts, :]

test_x = mnist.test.images
test_y = mnist.test.labels

random_order = np.arange(train_dataset_counts)

graphs = []

learning_rate = 1e-3
graph = tf.Graph()
with graph.as_default():
    x_placeholder = tf.placeholder(tf.float32, (None, 784), 'x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, (None, 10), 'y_placeholder')
    
    model = SimpleDNN()
    logits = model(x_placeholder)
    
    cross_entropy = tf.reduce_mean(tf.reduce_sum(-y_placeholder * utils.safe_log(logits), 1), name= 'cross_entropy')
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(cross_entropy, name= 'train_step')
    
    accuracy = tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            tf.arg_max(y_placeholder, dimension= 1),
                            tf.arg_max(logits, dimension= 1)
                        ),
                        tf.float32
                    ),
                    name= 'accuracy'
                )
    graphs.append(graph)

learning_rate = 1e-2                     
graph = tf.Graph()
with graph.as_default():
    x_placeholder = tf.placeholder(tf.float32, (None, 784), 'x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, (None, 10), 'y_placeholder')
    
    model = SimpleDNN()
    logits = model(x_placeholder)
    
    cross_entropy = tf.reduce_mean(tf.reduce_sum(-y_placeholder * utils.safe_log(logits), 1), name= 'cross_entropy')
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(cross_entropy, name= 'train_step')
    
    accuracy = tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            tf.arg_max(y_placeholder, dimension= 1),
                            tf.arg_max(logits, dimension= 1)
                        ),
                        tf.float32
                    ),
                    name= 'accuracy'
                )
    graphs.append(graph)

#%%

weights_record = []

EPOCH = 500
batch_size = 64
for graph in graphs:
    with tf.Session(graph= graph) as sess:
        sess.run(tf.global_variables_initializer())
        
        x_placeholder = graph.get_tensor_by_name('x_placeholder:0')
        y_placeholder = graph.get_tensor_by_name('y_placeholder:0')
        cross_entropy = graph.get_tensor_by_name('cross_entropy:0')
        train_step = graph.get_operation_by_name('train_step')
        accuracy = graph.get_tensor_by_name('accuracy:0')
        
        for epoch in range(1, EPOCH+1, 1):
            
            np.random.shuffle(random_order)
            train_x = train_x[random_order]
            train_y = train_y[random_order]
            
            total_loss = 0.0
            for idx in range(train_dataset_counts//batch_size):
                x = train_x[idx*batch_size : (idx+1)*batch_size]
                y = train_y[idx*batch_size : (idx+1)*batch_size]       
                
                feed_dict = {x_placeholder:x, y_placeholder:y}
                _, loss = sess.run([train_step, cross_entropy], feed_dict= feed_dict)
                total_loss += (loss / train_dataset_counts * batch_size)
            feed_dict = {x_placeholder:train_x, y_placeholder:train_y}   
            train_acc = sess.run(accuracy, feed_dict= feed_dict)
            print('epoch:', epoch, ',loss:', total_loss, ',train_acc:', train_acc)

        weights = []
        for variable in tf.trainable_variables():
            weights.append(variable.eval())
        weights_record.append(weights)
        
        batch_size = 1024

alpha_records = []
train_acc_record = []
test_acc_record = []
train_loss_record = []
test_loss_record = []

for alpha in range(-100, 201, 1):
    alpha = alpha * 1e-2
    alpha_records.append(alpha)
    with tf.Session(graph= graph) as sess:
        sess.run(tf.global_variables_initializer())
        for idx, variable in enumerate(tf.trainable_variables()):
            new_weight = alpha * weights_record[0][idx] + (1-alpha) * weights_record[1][idx]
            assign = tf.assign(variable, new_weight)
            sess.run(assign)
            
        x_placeholder = graph.get_tensor_by_name('x_placeholder:0')
        y_placeholder = graph.get_tensor_by_name('y_placeholder:0')
        cross_entropy = graph.get_tensor_by_name('cross_entropy:0')
        accuracy = graph.get_tensor_by_name('accuracy:0')
        
        feed_dict = {x_placeholder:train_x, y_placeholder:train_y}   
        train_acc, train_loss = sess.run([accuracy, cross_entropy], feed_dict= feed_dict)
        train_acc_record.append(train_acc)
        train_loss_record.append(train_loss)
        
        feed_dict = {x_placeholder:test_x, y_placeholder:test_y}   
        test_acc, test_loss = sess.run([accuracy, cross_entropy], feed_dict= feed_dict)
        test_acc_record.append(test_acc)
        test_loss_record.append(test_loss)

#%%
import matplotlib.pyplot as plt
    
ax1 = plt.plot()
plt.plot(alpha_records, train_loss_record, linestyle= '-', color= 'blue', label= 'train_loss')
plt.plot(alpha_records, test_loss_record, linestyle= '--', color= 'blue', label= 'test_loss')
plt.xlabel('alpha')
plt.ylabel('cross entropy',color='blue')
legend = plt.legend(loc= 'upper left')
legend.get_frame().set_alpha(0.5)

ax2 = plt.gca().twinx()
plt.plot(alpha_records, train_acc_record, linestyle= '-', color= 'red', label= 'train_acc')
plt.plot(alpha_records, test_acc_record, linestyle= '--', color= 'red', label= 'test_acc')
plt.ylabel('accuracy', color='red')
legend = plt.legend(loc= 'upper right')
legend.get_frame().set_alpha(0.5)

plt.savefig('./Flatness v.s. Generalization - part1.png')


