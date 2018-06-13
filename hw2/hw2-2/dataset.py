#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 11:41:11 2018

@author: halley
"""

import numpy as np

class DataSet:
  def __init__(self, captions, vocab_size, BOS_tag, EOS_tag):
    self._caption = []
    self._train_x = []
    self._train_y = []
    self._max_seq_len = 0
    
    current_idx = 0
    for caption in captions:
      caption_length = len(caption)
      for idx, sentence in enumerate(caption):
        sentence.append(EOS_tag)
        sentence = [BOS_tag] + sentence
        self._caption.append(np.asarray(sentence))
        if self._max_seq_len < len(sentence):
          self._max_seq_len = len(sentence)
        
        if idx >= (caption_length - 1):
          current_idx += 1
          break
        self._train_x.append(current_idx)
        self._train_y.append(current_idx + 1)
        current_idx += 1
      
    self._caption = np.asarray(self._caption)
    self._train_x = np.asarray(self._train_x)
    self._train_y = np.asarray(self._train_y)
    self._datalen = len(self._train_x)
    
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
      x.append(np.asarray(self._caption[self._train_x[self._index_in_epoch]]))
      y.append(np.asarray(self._caption[self._train_y[self._index_in_epoch]]))
      self._index_in_epoch += 1
    
    return np.asarray(x), np.asarray(y)
  
  
  def shuffle_data(self):
    random_order = np.arange(self.datalen)
    np.random.shuffle(random_order)
    self._train_x = self._train_x[random_order]
    self._train_y = self._train_y[random_order]
    
    
  @property
  def caption(self):
    return self._caption
    
  @property
  def train_x(self):
    return self._train_x

  @property
  def train_y(self):
    return self._train_y

  @property
  def max_seq_len(self):
    return self._max_seq_len
  
  @property
  def datalen(self):
    return self._datalen

  @property
  def vocab_size(self):
    return self._vocab_size

  @property
  def index_in_epoch(self):
    return self._index_in_epoch

  @property
  def N_epoch(self):
    return self._N_epoch


def test_main():
  import data_processing as DP
  
  captions = DP.read_train('./translated_training_label.json')
  dataset = DataSet(captions, 1024, '<BOS_tag>', '<EOS_tag>')
  assert len(dataset.train_x) == len(dataset.train_y), 'train_x, train_y length non-equal'
  assert len(dataset.caption) > len(dataset.train_x), 'train label extraction error, caption len:{}, train_x len:{}'.format(dataset.datalen, len(dataset.train_x))
  assert np.amax(dataset.train_x) == (len(dataset.caption)-2), 'idx error, max_label:{}, caption length:{}'.format(np.amax(dataset.train_x), len(dataset.caption))
  assert np.amax(dataset.train_y) == (len(dataset.caption)-1), 'idx error, max_label:{}, caption length:{}'.format(np.amax(dataset.train_y), len(dataset.caption))
  
if __name__ == '__main__':
  test_main()
  