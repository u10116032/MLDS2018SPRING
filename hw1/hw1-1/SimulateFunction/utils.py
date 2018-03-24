# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 16:44:28 2018

@author: Halley
"""

import tensorflow as tf

def dense_layer(input, width, reuse= False, activation= 'relu', name= "dense"):
    """
    Args:
        input: 4D tensor
        width: integer, number of neuron
        reuse: boolean
        activation: string, 'relu' or 'tanh'
        name: string
    return:
        4D tensor
        
    """
    with tf.variable_scope(name, reuse= reuse):
        
        if activation == 'relu':
            output = tf.layers.dense(input, width,activation= tf.nn.relu, kernel_initializer= tf.truncated_normal_initializer(stddev=0.002), name='dense')
        if activation == 'tanh':
            output = tf.layers.dense(input, width,activation= tf.nn.tanh, kernel_initializer= tf.truncated_normal_initializer(stddev=0.002), name='dense')
        if activation == 'sigmoid':
            output = tf.layers.dense(input, width,activation= tf.nn.sigmoid, kernel_initializer= tf.truncated_normal_initializer(stddev=0.002), name='dense')
        if activation == 'linear':
            output = tf.layers.dense(input, width,activation= None, kernel_initializer= tf.truncated_normal_initializer(stddev=0.002), name='dense')
        
        return output