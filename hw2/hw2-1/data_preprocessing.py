#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 17:52:24 2018

@author: halley
"""

import re
from collections import Counter
import json
import argparse

def build_counter(train_label_path):
    counter = Counter()
    
    with open(train_label_path, 'r') as f:
        data = json.load(f)
    
    for captions in data:
        caption = captions['caption']
        for sentence in caption:
            sentence = sentence.lower()
            sentence = re.sub(r'\b', ' ', sentence)
            sentence = re.sub('[^a-z0-9 ,.]', '', sentence)
            words = sentence.split()
            counter.update(words)
            
    return counter

def denoise(counter, word_frequency= None):
    if word_frequency is None:
        return counter
    
    cut_index = 0
    sorted_counter = counter.most_common()
    for idx, item in enumerate(sorted_counter):
        if item[1] < word_frequency:
            cut_index = idx
            break
    sorted_counter = counter.most_common(cut_index)
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
        
    
def main():
    parser = argparse.ArgumentParser(description= 'Create dictionary.txt and process training_label')
    parser.add_argument('--train_label_path',
                        type= str,
                        help= 'train_label file path',
                        required= True)
    parser.add_argument('--dictionary_path',
                        type= str,
                        help= 'dictionary path',
                        required= True)
    args = parser.parse_args()
    
    counter = build_counter(args.train_label_path)
    counter = denoise(counter, word_frequency= 3)
    counter.update(['<EOS>', '<BOS>', '<UNK>'])
    dictionary = build_dict(counter, args.dictionary_path)

if __name__ == '__main__':
    main()    
