#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 16:42:26 2018

@author: halley
"""

import tensorflow as tf
import ops

class Model:
    def __init__(self):
        self.reuse = False
        self.name = 'Model'
        
    def __call__(self, input):
        with tf.variable_scope(self.name, reuse= self.reuse):
            input_shape = input.get_shape().as_list()
            dense1 = ops.dense(input, [input_shape[-1], 20], 'dense1', reuse= self.reuse)
            dense2 = ops.dense(dense1, [20, 20], 'dense2', reuse= self.reuse)
            dense3 = ops.dense(dense2, [20, 20], 'dense3', reuse= self.reuse)
            output = ops.dense(dense3, [20, 10], 'output', activation='softmax', reuse= self.reuse)
            
        self.reuse = True
        return output
    
    
class SimulateFunctionModel:
    def __init__(self):
        self.reuse = False
        self.name = 'SimulateFunctionModel'
        
    def __call__(self, input):
        with tf.variable_scope(self.name, reuse= self.reuse):
            input_shape = input.get_shape().as_list()
            dense1 = ops.dense(input, [input_shape[-1], 20], 'dense1', reuse= self.reuse)
            dense2 = ops.dense(dense1, [20, 20], 'dense2', reuse= self.reuse)
            dense3 = ops.dense(dense2, [20, 20], 'dense3', reuse= self.reuse)
            output = ops.dense(dense3, [20, 1], 'output', activation='linear', reuse= self.reuse)
            
        self.reuse = True
        return output
    
            