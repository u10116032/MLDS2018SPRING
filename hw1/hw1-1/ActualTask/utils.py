# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 14:17:40 2018

@author: Halley
"""
import tensorflow as tf
import numpy as np

def conv2d(input, kernel_shape, name, reuse, activation='relu'):
    with tf.variable_scope(name, reuse= reuse):
        shape = np.asarray(kernel_shape)
        weight = tf.get_variable('weight',
                kernel_shape,
                initializer= tf.random_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias',
                shape[len(shape) - 1],
                initializer= tf.constant_initializer(0))
        output = tf.nn.conv2d(input, weight, [1,1,1,1], padding='SAME', name= 'conv2d')
        output = output + bias
        if activation == 'relu':
            output = tf.nn.relu(output)

        return output

def pooling(input, type, kernel, stride, name):
    if type == 'max':
        output = tf.nn.max_pool(input, kernel, stride, padding='SAME', name=name)
    if type == 'average':
        output = tf.nn.avg_pool(input, kernel, stride, padding='SAME', name=name)
        
    return output

def dense(input, unit, name, reuse, activation='relu'):
    with tf.variable_scope(name, reuse= reuse):
        kernel_initializer = None#tf.random_normal_initializer(0.01)
        output = tf.layers.dense(input, unit, name='dense', kernel_initializer=kernel_initializer)
        if activation == 'relu':
            output = tf.nn.relu(output)
        if activation == 'softmax':
            output = tf.nn.softmax(output)
            
        return output
    
