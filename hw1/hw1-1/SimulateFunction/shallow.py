# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 21:50:39 2018

@author: Halley
"""

import tensorflow as tf
import utils

class Shallow:
    def __init__(self):
        self.reuse = False
        self.name = 'Shallow'        
        
    def __call__(self, input):
        with tf.variable_scope(self.name):
            d1 = utils.dense_layer(input, 6833, self.reuse, 'relu', name='layer1')
            output = utils.dense_layer(d1, 1, self.reuse, 'linear', name='output')
        self.reuse = True
        
        return output