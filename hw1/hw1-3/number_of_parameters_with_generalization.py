# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 16:26:27 2018

@author: Halley
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import utils
from models import SimpleDNN_1, SimpleDNN_2, SimpleDNN_3, SimpleDNN_4, SimpleDNN_5, SimpleDNN_6, SimpleDNN_7, SimpleDNN_8, SimpleDNN_9, SimpleDNN_10
import numpy as np

train_dataset_counts = 1000
learning_rate = 0.003
EPOCH = 2
batch_size = 100

mnist = input_data.read_data_sets('./MNIST_data', one_hot= True)
train_x = mnist.train.images[0:train_dataset_counts, :]
train_y = mnist.train.labels[0:train_dataset_counts, :]

test_x = mnist.train.images[0:train_dataset_counts, :]
test_y = mnist.train.labels[0:train_dataset_counts, :]

random_order = np.arange(train_dataset_counts)

graphs = []

graph = tf.Graph()
with graph.as_default():
    x_placeholder = tf.placeholder(tf.float32, (None, 784), 'x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, (None, 10), 'y_placeholder')
    
    model = SimpleDNN_1()
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
graphs.append(graph)

graph = tf.Graph()
with graph.as_default():
    x_placeholder = tf.placeholder(tf.float32, (None, 784), 'x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, (None, 10), 'y_placeholder')
    
    model = SimpleDNN_2()
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
graphs.append(graph)

graph = tf.Graph()
with graph.as_default():
    x_placeholder = tf.placeholder(tf.float32, (None, 784), 'x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, (None, 10), 'y_placeholder')
    
    model = SimpleDNN_3()
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
graphs.append(graph)

graph = tf.Graph()
with graph.as_default():
    x_placeholder = tf.placeholder(tf.float32, (None, 784), 'x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, (None, 10), 'y_placeholder')
    
    model = SimpleDNN_4()
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
graphs.append(graph)

graph = tf.Graph()
with graph.as_default():
    x_placeholder = tf.placeholder(tf.float32, (None, 784), 'x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, (None, 10), 'y_placeholder')
    
    model = SimpleDNN_5()
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
graphs.append(graph)

graph = tf.Graph()
with graph.as_default():
    x_placeholder = tf.placeholder(tf.float32, (None, 784), 'x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, (None, 10), 'y_placeholder')
    
    model = SimpleDNN_6()
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
graphs.append(graph)

graph = tf.Graph()
with graph.as_default():
    x_placeholder = tf.placeholder(tf.float32, (None, 784), 'x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, (None, 10), 'y_placeholder')
    
    model = SimpleDNN_7()
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
graphs.append(graph)

graph = tf.Graph()
with graph.as_default():
    x_placeholder = tf.placeholder(tf.float32, (None, 784), 'x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, (None, 10), 'y_placeholder')
    
    model = SimpleDNN_8()
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
graphs.append(graph)

graph = tf.Graph()
with graph.as_default():
    x_placeholder = tf.placeholder(tf.float32, (None, 784), 'x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, (None, 10), 'y_placeholder')
    
    model = SimpleDNN_9()
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
graphs.append(graph)

graph = tf.Graph()
with graph.as_default():
    x_placeholder = tf.placeholder(tf.float32, (None, 784), 'x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, (None, 10), 'y_placeholder')
    
    model = SimpleDNN_10()
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
graphs.append(graph)

#%%

train_loss_record = []
test_loss_record = []
train_acc_record = []
test_acc_record = []
parameters_record = []

for each_graph in graphs:
    with tf.Session(graph= each_graph) as sess:
        sess.run(tf.global_variables_initializer())
        
        parameters = 0
        for variable in tf.trainable_variables():
            parameters += np.prod(variable.get_shape().as_list())
        parameters_record.append(parameters)
        
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
        
        feed_dict = {x_placeholder:train_x, y_placeholder:train_y}   
        train_acc, train_loss = sess.run([accuracy, cross_entropy], feed_dict= feed_dict)
        
        feed_dict = {x_placeholder:test_x, y_placeholder:test_y}   
        test_acc, test_loss = sess.run([accuracy, cross_entropy], feed_dict= feed_dict)
        
        train_loss_record.append(train_loss)
        test_loss_record.append(test_loss)
        train_acc_record.append(train_acc)
        test_acc_record.append(test_acc)
#%%
import matplotlib.pyplot as plt
fig1 = plt.figure(1)
plt.xlabel('# of Parameters')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.scatter(parameters_record, train_loss_record, label="train_loss")            
plt.scatter(parameters_record, test_loss_record, label="test_loss")
plt.legend(loc='upper right')      
fig1.savefig('Model Loss.png')
plt.close()

fig2 = plt.figure(2)
plt.xlabel('# of Parameters')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.scatter(parameters_record, train_acc_record, label="train_acc")            
plt.scatter(parameters_record, test_acc_record, label="test_acc")
plt.legend(loc='upper left')      
fig2.savefig('Model Accuracy.png')
plt.close()