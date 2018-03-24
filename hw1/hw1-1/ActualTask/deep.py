# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 14:44:50 2018

@author: Halley
"""

import tensorflow as tf
import utils
import numpy as np

class Deep:
    def __init__(self):
        self.name = 'Deep'
        self.reuse = False
    
    def __call__(self, input):
        with tf.variable_scope(self.name, reuse= self.reuse):
            conv1_1 = utils.conv2d(input, [5,5,1,6], 'conv1_1', reuse= self.reuse)
            conv1_2 = utils.conv2d(conv1_1, [5,5,6,6], 'conv1_2', reuse= self.reuse)
            #max_pool1 = utils.pooling(conv1_2, 'max', [1,2,2,1], [1,2,2,1], name= 'max_pool1')
            conv2_1 = utils.conv2d(conv1_2, [5,5,6,16], 'conv2_1', reuse= self.reuse)
            conv2_2 = utils.conv2d(conv2_1, [5,5,16,16], 'conv2_2', reuse= self.reuse)
            #max_pool2 = utils.pooling(conv2_2, 'max', [1,2,2,1], [1,2,2,1], name= 'max_pool2')
            #conv3_1 = utils.conv2d(max_pool2, [5,5,16,32], 'conv3_1', reuse= self.reuse)
            #conv3_2 = utils.conv2d(conv3_1, [5,5,32,32], 'conv3_2', reuse= self.reuse)
            #max_pool3 = utils.pooling(conv3_2, 'max', [1,2,2,1], [1,2,2,1], name= 'max_pool3')
            featuremap_size = np.prod(np.asarray(conv2_2.get_shape())[1:4])
            flatten = tf.reshape(conv2_2, (-1, featuremap_size), name= 'flatten')
            #dense1 = utils.dense(flatten, 120, 'dense1', reuse=self.reuse)
            #dense2 = utils.dense(dense1, 86, 'dense2', reuse=self.reuse)
            output = utils.dense(flatten, 10, 'output', reuse=self.reuse, activation= 'softmax')
        
        self.reuse = True
        return output