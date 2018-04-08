# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 21:21:16 2018

@author: Halley
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from models import SimpleDNN
import utils

mnist = input_data.read_data_sets('MNIST_data', one_hot= True)

train_dataset_count = 55000

train_x = mnist.train.images[0:train_dataset_count]
train_y = mnist.train.labels[0:train_dataset_count]

test_x = mnist.test.images
test_y = mnist.test.labels

random_order = np.arange(train_dataset_count)

#%%
graph = tf.Graph()
with graph.as_default():
    x_placeholder = tf.placeholder(tf.float32, (None, 784), name= 'x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, (None, 10), name= 'y_placeholder')

    model = SimpleDNN()
    logits = model(x_placeholder)

    cross_entropy = tf.reduce_mean(
                        tf.reduce_sum(
                            -y_placeholder * utils.safe_log(logits),
                            axis= 1
                        ),
                    axis= 0,
                    name= 'cross_entropy'
                )
    optimizer = tf.train.AdamOptimizer(learning_rate= 1e-2)
    train_step = optimizer.minimize(cross_entropy, name= 'train_step')

    accuracy = tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        tf.arg_max(y_placeholder, dimension= 1),
                        tf.arg_max(logits, dimension= 1)
                    ),
                    tf.float32
                ),
                axis= 0,
                name= 'accuracy'
            )

#%%
batch_size = 500
EPOCH = 200

train_acc_record = []
test_acc_record = []
train_loss_record = []
test_loss_record = []
sensitivity_record = []
batch_size_record = []

for _ in range(10): # train 10 different structure
    with tf.Session(graph= graph) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, EPOCH+1, 1):
            np.random.shuffle(random_order)
            train_x = train_x[random_order]
            train_y = train_y[random_order]

            train_loss = 0
            for idx in range(train_dataset_count // batch_size):
                x = train_x[idx * batch_size : (idx + 1) * batch_size]
                y = train_y[idx * batch_size : (idx + 1) * batch_size]

                feed_dict = {x_placeholder:x, y_placeholder:y}
                _, loss = sess.run([train_step, cross_entropy], feed_dict= feed_dict)
                train_loss += (loss * (batch_size / train_dataset_count))
            feed_dict = {x_placeholder: train_x, y_placeholder: train_y}
            train_acc = sess.run(accuracy, feed_dict= feed_dict)
            print('epoch:', epoch, ', loss:', train_loss, ',train_acc:', train_acc)

        feed_dict = {x_placeholder: train_x, y_placeholder: train_y}
        train_acc, train_loss = sess.run([accuracy, cross_entropy], feed_dict= feed_dict)

        feed_dict = {x_placeholder: test_x, y_placeholder: test_y}
        test_acc, test_loss = sess.run([accuracy, cross_entropy], feed_dict= feed_dict)

        batch_size_record.append(batch_size)
        batch_size = batch_size + 500
        train_acc_record.append(train_acc)
        test_acc_record.append(test_acc)
        train_loss_record.append(train_loss)
        test_loss_record.append(test_loss)

        random_seed = np.random.randint(train_dataset_count, size=None)
        jacobian_x = train_x[random_seed].reshape(1, -1)

        jacobian_elements = []
        for yi in tf.unstack(logits, axis= 1):
            grad = sess.run(tf.gradients(yi, x_placeholder), feed_dict= {x_placeholder: jacobian_x})[0]
            print(grad.shape)
            jacobian_elements.append(grad)
        jacobian_elements = np.asarray(jacobian_elements)
        sensitivity = np.sum(jacobian_elements**2)**0.5

        sensitivity_record.append(sensitivity)

#%%
import matplotlib.pyplot as plt

ax1 = plt.plot()
plt.plot(batch_size_record, train_loss_record, linestyle= '-', color= 'blue', label= 'train_loss')
plt.plot(batch_size_record, test_loss_record, linestyle= '--', color= 'blue', label= 'test_loss')
plt.xlabel('batch size')
plt.ylabel('cross entropy',color='blue')
legend = plt.legend(loc= 'upper left')
legend.get_frame().set_alpha(0.5)

ax2 = plt.gca().twinx()
plt.plot(batch_size_record, sensitivity_record, linestyle= '-', color= 'red', label= 'sensitivity')
plt.ylabel('sensitivity', color='red')
legend = plt.legend(loc= 'upper right')
legend.get_frame().set_alpha(0.5)

plt.savefig('./Flatness v.s. Generalization - part2-1.png')
plt.close()

ax1 = plt.plot()
plt.plot(batch_size_record, train_acc_record, linestyle= '-', color= 'blue', label= 'train_loss')
plt.plot(batch_size_record, test_acc_record, linestyle= '--', color= 'blue', label= 'test_loss')
plt.xlabel('batch size')
plt.ylabel('accuracy',color='blue')
legend = plt.legend(loc= 'upper left')
legend.get_frame().set_alpha(0.5)

ax2 = plt.gca().twinx()
plt.plot(batch_size_record, sensitivity_record, linestyle= '-', color= 'red', label= 'sensitivity')
plt.ylabel('sensitivity', color='red')
legend = plt.legend(loc= 'upper right')
legend.get_frame().set_alpha(0.5)

plt.savefig('./Flatness v.s. Generalization - part2-2.png')
plt.close()


