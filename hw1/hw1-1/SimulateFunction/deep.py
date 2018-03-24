# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 18:29:20 2018

@author: Halley
"""
import tensorflow as tf
import utils

class Deep:
    def __init__(self):
        self.reuse = False
        self.name = 'Deep'        
        
    def __call__(self, input):
        with tf.variable_scope(self.name):
            d1 = utils.dense_layer(input, 100, self.reuse, 'relu', name='layer1')
            d2 = utils.dense_layer(d1, 100, self.reuse, 'relu', name='layer2')
            d3 = utils.dense_layer(d2, 100, self.reuse, 'relu', name='layer3')
            output = utils.dense_layer(d3, 1, self.reuse, 'linear', name='output')
        self.reuse = True
        
        return output