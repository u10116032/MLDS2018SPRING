# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 15:47:41 2018

@author: Halley
"""

import tensorflow as tf
import utils
import numpy as np

class Shallow:
    def __init__(self):
        self.name = 'Shallow'
        self.reuse = False
    
    def __call__(self, input):
        with tf.variable_scope(self.name, reuse= self.reuse):
            conv1_1 = utils.conv2d(input, [5,5,1,17], 'conv1_1', reuse= self.reuse)
            #max_pool1 = utils.pooling(conv1_1, 'max', [1,2,2,1], [1,2,2,1], name= 'max_pool1')
            featuremap_size = np.prod(conv1_1.get_shape()[1:4])
            flatten = tf.reshape(conv1_1, (-1, featuremap_size), name= 'flatten')
            output = utils.dense(flatten, 10, 'output', reuse=self.reuse, activation= 'softmax')
        
        self.reuse = True
        return output