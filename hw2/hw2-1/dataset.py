#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 12:13:17 2018

@author: halley
"""

import numpy as np

class DataSet:
    def __init__(self, data_path, captions, vocab_size, BOS_tag, EOS_tag):
        self._feat = []
        self._label = []
        self._caption = []
        self._max_seq_len = 0
        
        for idx, caption in enumerate(captions):
            x = np.load(data_path + '/' + caption['id'] + '.npy')
            self._feat.append(x)
            
            for sentence in caption['caption']:
                self._label.append(idx)
                sentence.append(EOS_tag)
                sentence = [BOS_tag] + sentence
                self._caption.append(sentence)
                if self._max_seq_len < len(sentence):
                    self._max_seq_len = len(sentence)
        
        self._feat = np.asarray(self._feat)
        self._label = np.asarray(self._label)
        self._caption = np.asarray(self._caption)
        self._datalen = len(self._caption)
        self._feat_timestep = len(self._feat[0]) #80
        self._feat_dim = len(self._feat[0][0]) #4096
        self._vocab_size = vocab_size
        self._index_in_epoch = 0
        self._N_epoch = 1
        
        self.shuffle_data()
        
    def next_batch(self, batch_size = 1):
        x = []
        y = []
        
        for _ in range(batch_size):
            if self._index_in_epoch >= self._datalen:
                self.shuffle_data()
                
                self._index_in_epoch = 0
                self._N_epoch += 1
            
            x.append(self._feat[self._label[self._index_in_epoch]])
            y.append(self._caption[self._index_in_epoch])
            self._index_in_epoch += 1
        
        return np.asarray(x), np.asarray(y)
    
    def shuffle_data(self):
        random_order = np.arange(self._datalen)
        np.random.shuffle(random_order)
        self._label = self._label[random_order]
        self._caption = self._caption[random_order]
    
    @property
    def feat(self):
        return self._feat

    @property
    def label(self):
        return self._label

    @property
    def caption(self):
        return self._caption

    @property
    def max_seq_len(self):
        return self._max_seq_len
    
    @property
    def datalen(self):
        return self._datalen

    @property
    def feat_timestep(self):
        return self._feat_timestep

    @property
    def feat_dim(self):
        return self._feat_dim

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def N_epoch(self):
        return self._N_epoch
            