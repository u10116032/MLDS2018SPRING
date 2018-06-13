#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 11:42:52 2018

@author: halley
"""

import numpy as np
import argparse
import re
import json
from collections import Counter


def build_counter(data_path):
  counter = Counter()
  
  with open(data_path, 'r') as data:
    for sentence in data:
      sentence = re.sub(r'\n', '', sentence) 
      if sentence == '+++$+++':
        continue
      words = sentence.split()
      counter.update(words)
      
  return counter
          

def denoise(counter, word_frequency= None):
  if word_frequency is None:
    return counter.most_common()
  
  cut_index = 0
  sorted_counter = counter.most_common()
  for idx, item in enumerate(sorted_counter):
    if item[1] < word_frequency:
      cut_index = idx
      break
  sorted_counter = counter.most_common(cut_index)
  counter = Counter(dict(sorted_counter))
  return counter
  

def build_dict(counter, dictionary_path= None):
  dictionary = {}
  sorted_counter = counter.most_common()
  
  for idx, item in enumerate(sorted_counter):
    dictionary[item[0]] = idx
  
  if dictionary_path is not None:
    with open(dictionary_path, 'w') as f:
      for idx, item in enumerate(sorted_counter):
        f.write(str(idx) + ' ' + item[0] + ' ' + str(item[1]) + '\n')
  return dictionary


def read_dictionary(filepath):
  dictionary = {}
  with open(filepath, 'r') as f:
    for line in f:
      tokens = line.split()
      dictionary[tokens[1]] = int(tokens[0])
  return dictionary


def translate_label(dictionary, train_data_path, train_label_path = None):
  with open(train_data_path, 'r') as train_data:
    label_data = []
    caption = []
    for sentence in train_data:
      sentence = re.sub(r'\n', '', sentence) 
      if sentence == '+++$+++':
        label_data.append(caption)
        caption = []
        continue   
      words = sentence.split()
      translated_words = [dictionary[word] if word in dictionary else dictionary['<UNK>'] for word in words]
      caption.append(translated_words)
    if not len(caption) == 0:
      label_data.append(caption)
      
  if train_label_path is not None:
    with open(train_label_path, 'w') as train_label:
      json.dump(label_data, train_label, indent=4)
  
  return label_data


def read_train(filename):
  if filename[-3:] == 'npy':
    data = np.load(filename)
  elif filename[-4:] == 'json':
    with open(filename, 'r') as f:
      data = json.load(f)
  else:
    data = []

  return data


def main():
  parser = argparse.ArgumentParser(description= 'Create dictionary.txt and process training_label')
  parser.add_argument('--train_data',
                      type= str,
                      help= 'train data path',
                      required= True)
  parser.add_argument('--dictionary_path',
                      type= str,
                      help= 'dictionary path',
                      required= True)
  args = parser.parse_args()
  
  counter = build_counter(args.train_data)
  counter = denoise(counter, word_frequency= 3)
  counter.update(['<EOS>', '<BOS>', '<UNK>'])
  dictionary = build_dict(counter, args.dictionary_path)
  translate_label(dictionary, args.train_data, 'translated_training_label.json')
#%%

if __name__ == '__main__':
  main()