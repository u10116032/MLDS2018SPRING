#%%
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 16:40:15 2018
@author: Halley

Objective Function: y = x**3 + x**2 + x + 1
    
Goal: Design two DNN with same parameter counts to fit objective function,
        and record the training process. 
"""

import tensorflow as tf
import numpy as np
from deep import Deep
from shallow import Shallow
import os

EPOCH = 10000
BATCH_SIZE = 200

def objective_function(x):
    return x**5-x**4+x**3-x**2+x-1

train_x = np.random.normal(scale= 10, size= (1000,1))
train_y = np.asarray([objective_function(entry) for entry in train_x], dtype= np.float)

loss_record = {}

#%%
tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default():
    x_placeholder = tf.placeholder(tf.float32, (None, 1))
    y_placeholder = tf.placeholder(tf.float32, (None, 1))

    shallow_model = Shallow()
    loss = tf.reduce_mean(tf.reduce_sum((y_placeholder - shallow_model(x_placeholder))**2, axis= 1))
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0005, beta1= 0.5)
    train_process = optimizer.minimize(loss)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("./tensorboard/", graph)    


random_order = np.arange(train_x.shape[0])
with tf.Session(graph= graph) as sess:
    sess.run(tf.global_variables_initializer())
    loss_temp_record = []
    for epoch in range(1, EPOCH+1):
        total_loss = 0
        for index in range(train_x.shape[0]//BATCH_SIZE):
            x = np.reshape(train_x[index:index+BATCH_SIZE], (BATCH_SIZE, 1))
            y = np.reshape(train_y[index:index+BATCH_SIZE], (BATCH_SIZE, 1))
            _, loss_value = sess.run([train_process, loss], 
                    feed_dict= {x_placeholder:x, y_placeholder:y})
            total_loss = total_loss + loss_value
        total_loss = total_loss/train_x.shape[0]  
        loss_temp_record.append(total_loss)
        print("epoch:", str(epoch), ", loss:", str(total_loss))        
        summary = sess.run(summary_op, feed_dict={x_placeholder:train_x, y_placeholder: train_y})
        summary_writer.add_summary(summary, epoch)
    
    loss_record['shallow'] = loss_temp_record
    dir_name = './Shallow_Model'
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    saver = tf.train.Saver()
    saver.save(sess, dir_name + './shallow.ckpt')    
            
#%%   
tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default():
    x_placeholder = tf.placeholder(tf.float32, (None, 1))
    y_placeholder = tf.placeholder(tf.float32, (None, 1))

    deep_model = Deep()
    loss = tf.reduce_mean(tf.reduce_sum((y_placeholder - deep_model(x_placeholder))**2, axis= 1))
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0005, beta1= 0.5)
    train_process = optimizer.minimize(loss)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("./tensorboard/", graph)    


random_order = np.arange(train_x.shape[0])
with tf.Session(graph= graph) as sess:
    sess.run(tf.global_variables_initializer())
    loss_temp_record = []
    for epoch in range(1, EPOCH+1):
        total_loss = 0
        for index in range(train_x.shape[0]//BATCH_SIZE):
            x = np.reshape(train_x[index:index+BATCH_SIZE], (BATCH_SIZE, 1))
            y = np.reshape(train_y[index:index+BATCH_SIZE], (BATCH_SIZE, 1))
            _, loss_value = sess.run([train_process, loss], 
                    feed_dict= {x_placeholder:x, y_placeholder:y})
            total_loss = total_loss + loss_value
        total_loss = total_loss/train_x.shape[0]  
        loss_temp_record.append(total_loss)
        print("epoch:", str(epoch), ", loss:", str(total_loss))
        summary = sess.run(summary_op, feed_dict={x_placeholder:train_x, y_placeholder: train_y})
        summary_writer.add_summary(summary, epoch)
    
    loss_record['deep'] = loss_temp_record
    dir_name = './Deep_Model'
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    saver = tf.train.Saver()
    saver.save(sess, dir_name + './deep.ckpt')
#%%
import matplotlib.pyplot as plt
x_axis = np.arange(EPOCH) + 1
plt.plot(x_axis, loss_record['shallow'])
plt.plot(x_axis, loss_record['deep'])
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend(['shallow', 'deep'], loc= 'upper right')
plt.savefig('loss_compare.jpg')
#%%
training_range = np.arange(-10, 10+0.25, 0.25)
plt.figure(num=1)
plt.plot(training_range, objective_function(training_range),'bs')

tf.reset_default_graph()
graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    new_saver = tf.train.import_meta_graph("./Shallow_Model/shallow.ckpt.meta")
    new_saver.restore(sess, tf.train.latest_checkpoint('./Shallow_Model/'))

    x_pl = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
    predict = tf.get_default_graph().get_tensor_by_name('Shallow/output/dense/BiasAdd:0')
    prediction = sess.run(predict, feed_dict={x_pl:np.reshape(training_range, (-1,1))})
    plt.plot(training_range, np.asarray(prediction),'r--')
    
tf.reset_default_graph()
graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    new_saver = tf.train.import_meta_graph("./Deep_Model/deep.ckpt.meta")
    new_saver.restore(sess, tf.train.latest_checkpoint('./Deep_Model/'))
    
    x_pl = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
    predict = tf.get_default_graph().get_tensor_by_name('Deep/output/dense/BiasAdd:0')
    prediction = sess.run(predict, feed_dict={x_pl:np.reshape(training_range, (-1,1))})
    plt.plot(training_range, np.asarray(prediction),'g^')    

plt.xlabel('x')
plt.ylabel('y')
plt.legend(['ground truth','shallow','deep'], loc='upper right')
plt.title(r'$y=x^{5}-x^{4}+x^{3}-x^{2}+x-1$')
plt.savefig('validation_compare')