# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 15:00:22 2018

@author: Halley
"""

import tensorflow as tf
import ops

class SimpleDNN:
    def __init__(self):
        self.reuse = False
        self.name = 'SimpleDNN'
        
    def __call__(self, input):
        with tf.variable_scope(self.name, reuse= self.reuse):
            input_shape = input.get_shape().as_list()
            dense1 = ops.dense(input, (input_shape[-1], 20), 'dense1', reuse= self.reuse)
            dense2 = ops.dense(dense1, (20, 20), 'dense2', reuse= self.reuse)
            dense3 = ops.dense(dense2, (20, 20), 'dense3', reuse= self.reuse)
            output = ops.dense(dense3, (20, 10), 'output', reuse= self.reuse, activation= 'softmax')
            
        self.reuse = True
        return output

class SimpleDNN_1:
    def __init__(self):
        self.reuse = False
        self.name = 'SimpleDNN_1'
        
    def __call__(self, input):
        with tf.variable_scope(self.name, reuse= self.reuse):
            input_shape = input.get_shape().as_list()
            dense1 = ops.dense(input, (input_shape[-1], 5), 'dense1', reuse= self.reuse)
            dense2 = ops.dense(dense1, (5, 5), 'dense2', reuse= self.reuse)
            dense3 = ops.dense(dense2, (5, 5), 'dense3', reuse= self.reuse)
            output = ops.dense(dense3, (5, 10), 'output', reuse= self.reuse, activation= 'softmax')
            
        self.reuse = True
        return output
    
class SimpleDNN_2:
    def __init__(self):
        self.reuse = False
        self.name = 'SimpleDNN_2'
        
    def __call__(self, input):
        with tf.variable_scope(self.name, reuse= self.reuse):
            input_shape = input.get_shape().as_list()
            dense1 = ops.dense(input, (input_shape[-1], 10), 'dense1', reuse= self.reuse)
            dense2 = ops.dense(dense1, (10, 10), 'dense2', reuse= self.reuse)
            dense3 = ops.dense(dense2, (10, 10), 'dense3', reuse= self.reuse)
            output = ops.dense(dense3, (10, 10), 'output', reuse= self.reuse, activation= 'softmax')
            
        self.reuse = True
        return output
    
class SimpleDNN_3:
    def __init__(self):
        self.reuse = False
        self.name = 'SimpleDNN_3'
        
    def __call__(self, input):
        with tf.variable_scope(self.name, reuse= self.reuse):
            input_shape = input.get_shape().as_list()
            dense1 = ops.dense(input, (input_shape[-1], 20), 'dense1', reuse= self.reuse)
            dense2 = ops.dense(dense1, (20, 20), 'dense2', reuse= self.reuse)
            dense3 = ops.dense(dense2, (20, 20), 'dense3', reuse= self.reuse)
            output = ops.dense(dense3, (20, 10), 'output', reuse= self.reuse, activation= 'softmax')
            
        self.reuse = True
        return output
    
class SimpleDNN_4:
    def __init__(self):
        self.reuse = False
        self.name = 'SimpleDNN_4'
        
    def __call__(self, input):
        with tf.variable_scope(self.name, reuse= self.reuse):
            input_shape = input.get_shape().as_list()
            dense1 = ops.dense(input, (input_shape[-1], 30), 'dense1', reuse= self.reuse)
            dense2 = ops.dense(dense1, (30, 30), 'dense2', reuse= self.reuse)
            dense3 = ops.dense(dense2, (30, 30), 'dense3', reuse= self.reuse)
            output = ops.dense(dense3, (30, 10), 'output', reuse= self.reuse, activation= 'softmax')
            
        self.reuse = True
        return output
    
class SimpleDNN_5:
    def __init__(self):
        self.reuse = False
        self.name = 'SimpleDNN_5'
        
    def __call__(self, input):
        with tf.variable_scope(self.name, reuse= self.reuse):
            input_shape = input.get_shape().as_list()
            dense1 = ops.dense(input, (input_shape[-1], 40), 'dense1', reuse= self.reuse)
            dense2 = ops.dense(dense1, (40, 40), 'dense2', reuse= self.reuse)
            dense3 = ops.dense(dense2, (40, 40), 'dense3', reuse= self.reuse)
            output = ops.dense(dense3, (40, 10), 'output', reuse= self.reuse, activation= 'softmax')
            
        self.reuse = True
        return output
    
class SimpleDNN_6:
    def __init__(self):
        self.reuse = False
        self.name = 'SimpleDNN_6'
        
    def __call__(self, input):
        with tf.variable_scope(self.name, reuse= self.reuse):
            input_shape = input.get_shape().as_list()
            dense1 = ops.dense(input, (input_shape[-1], 50), 'dense1', reuse= self.reuse)
            dense2 = ops.dense(dense1, (50, 50), 'dense2', reuse= self.reuse)
            dense3 = ops.dense(dense2, (50, 50), 'dense3', reuse= self.reuse)
            output = ops.dense(dense3, (50, 10), 'output', reuse= self.reuse, activation= 'softmax')
            
        self.reuse = True
        return output
    
class SimpleDNN_7:
    def __init__(self):
        self.reuse = False
        self.name = 'SimpleDNN_7'
        
    def __call__(self, input):
        with tf.variable_scope(self.name, reuse= self.reuse):
            input_shape = input.get_shape().as_list()
            dense1 = ops.dense(input, (input_shape[-1], 60), 'dense1', reuse= self.reuse)
            dense2 = ops.dense(dense1, (60, 60), 'dense2', reuse= self.reuse)
            dense3 = ops.dense(dense2, (60, 60), 'dense3', reuse= self.reuse)
            output = ops.dense(dense3, (60, 10), 'output', reuse= self.reuse, activation= 'softmax')
            
        self.reuse = True
        return output
    
class SimpleDNN_8:
    def __init__(self):
        self.reuse = False
        self.name = 'SimpleDNN_8'
        
    def __call__(self, input):
        with tf.variable_scope(self.name, reuse= self.reuse):
            input_shape = input.get_shape().as_list()
            dense1 = ops.dense(input, (input_shape[-1], 70), 'dense1', reuse= self.reuse)
            dense2 = ops.dense(dense1, (70, 70), 'dense2', reuse= self.reuse)
            dense3 = ops.dense(dense2, (70, 70), 'dense3', reuse= self.reuse)
            output = ops.dense(dense3, (70, 10), 'output', reuse= self.reuse, activation= 'softmax')
            
        self.reuse = True
        return output
    
class SimpleDNN_9:
    def __init__(self):
        self.reuse = False
        self.name = 'SimpleDNN_9'
        
    def __call__(self, input):
        with tf.variable_scope(self.name, reuse= self.reuse):
            input_shape = input.get_shape().as_list()
            dense1 = ops.dense(input, (input_shape[-1], 80), 'dense1', reuse= self.reuse)
            dense2 = ops.dense(dense1, (80, 80), 'dense2', reuse= self.reuse)
            dense3 = ops.dense(dense2, (80, 80), 'dense3', reuse= self.reuse)
            output = ops.dense(dense3, (80, 10), 'output', reuse= self.reuse, activation= 'softmax')
            
        self.reuse = True
        return output
    
class SimpleDNN_10:
    def __init__(self):
        self.reuse = False
        self.name = 'SimpleDNN_10'
        
    def __call__(self, input):
        with tf.variable_scope(self.name, reuse= self.reuse):
            input_shape = input.get_shape().as_list()
            dense1 = ops.dense(input, (input_shape[-1], 90), 'dense1', reuse= self.reuse)
            dense2 = ops.dense(dense1, (90, 90), 'dense2', reuse= self.reuse)
            dense3 = ops.dense(dense2, (90, 90), 'dense3', reuse= self.reuse)
            output = ops.dense(dense3, (90, 10), 'output', reuse= self.reuse, activation= 'softmax')
            
        self.reuse = True
        return output