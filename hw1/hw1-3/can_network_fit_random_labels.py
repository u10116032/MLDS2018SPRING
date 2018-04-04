# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 15:10:49 2018

@author: Halley
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import utils
from models import SimpleDNN
import numpy as np

train_dataset_counts = 1000
learning_rate = 0.003
EPOCH = 5000
batch_size = 100

mnist = input_data.read_data_sets('./MNIST_data', one_hot= True)
train_x = mnist.train.images[0:train_dataset_counts, :]
train_y = mnist.train.labels[0:train_dataset_counts, :]

test_x = mnist.train.images[0:train_dataset_counts, :]
test_y = mnist.train.labels[0:train_dataset_counts, :]

random_order = np.arange(train_dataset_counts)
np.random.shuffle(random_order)
train_y = train_y[random_order]

graph = tf.Graph()
with graph.as_default():
    x_placeholder = tf.placeholder(tf.float32, (None, 784), 'x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, (None, 10), 'y_placeholder')
    
    model = SimpleDNN()
    logits = model(x_placeholder)
    
    cross_entropy = tf.reduce_mean(tf.reduce_sum(-y_placeholder * utils.safe_log(logits), 1))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(cross_entropy)
    
    accuracy = tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            tf.arg_max(y_placeholder, dimension= 1),
                            tf.arg_max(logits, dimension= 1)
                        ),
                        tf.float32
                    )  
                )

train_loss_record = []
test_loss_record = []
with tf.Session(graph= graph) as sess:
    sess.run(tf.global_variables_initializer())
    
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
        
        feed_dict = {x_placeholder:test_x, y_placeholder:test_y}   
        test_loss = sess.run(cross_entropy, feed_dict= feed_dict)
        
        train_loss_record.append(total_loss)
        test_loss_record.append(test_loss)

import matplotlib.pyplot as plt
fig1 = plt.figure(1)
x_labels = np.arange(1, EPOCH+1, 1)
plt.title('Can Network Fit Random Labels?')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(x_labels, train_loss_record, linestyle= '-', label="train_loss")            
plt.plot(x_labels, test_loss_record, linestyle= '-', label="test_loss")
plt.legend(loc='upper left')      
fig1.savefig('Can Network Fit Random Labels.png')