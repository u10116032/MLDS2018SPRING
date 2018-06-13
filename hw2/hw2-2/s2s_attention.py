#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 15:11:25 2018

@author: halley
"""

import tensorflow as tf
from rnn_models import RnnModel_Attention
import numpy as np
from dataset import DataSet
import data_processing as DP
import argparse
import json
import os
import re


##### Training file path #####

dict_file = 'dictionary.txt'
train_label_file = './translated_training_label.json'

test_file = './mlds_hw2_2_data/test_input.txt'
model_file = './s2s_attention/model.ckpt'

######### Constants ##########

EOS_tag = '<EOS>'
BOS_tag = '<BOS>'
UNK_tag = '<UNK>'

######### Parameters #########

batch_size = 100
N_hidden = 256
N_epoch = 1000
max_seq_len = 30
save_step = 20

params = {}
params['cell_type'] = 'lstm'
params['batch_size'] = batch_size
params['learning_rate'] = 0.001
params['hidden_layers'] = 1
params['dropout'] = 0.1

##############################

def run_train():
  # Inputs
  dictionary = DP.read_dictionary(dict_file)
  train_data = DP.read_train(train_label_file)
  train = DataSet(train_data, len(dictionary), dictionary[BOS_tag], dictionary[EOS_tag])

  N_input = train.datalen
  N_iter = N_input * N_epoch // batch_size
  print('Total training steps: %d' % N_iter)

  # Model
  graph = tf.Graph()
  with graph.as_default():
    train_model = RnnModel_Attention(
                is_training= True,
                vocab_size = train.vocab_size,
                N_hidden = N_hidden,
                N_caption_step = train.max_seq_len,
                **params)
    tf_encoder_input, tf_decoder_input, tf_decoder_target, tf_decoder_mask, loss, train_step = train_model.build_train_model()

  with tf.Session(graph= graph) as sess:
    sess.run(tf.global_variables_initializer())
    step = train_model.restore_model(sess, model_file)

    while step < N_iter:
      batch_x, batch_y = train.next_batch(batch_size=batch_size)

      x = np.full((batch_size, train.max_seq_len), dictionary[EOS_tag])
      y = np.full((batch_size, train.max_seq_len), dictionary[EOS_tag])
      y_mask = np.zeros((batch_size, train.max_seq_len - 1))
      for i in range(batch_size):
        x[i,:len(batch_x[i])] = batch_x[i]
        y[i,:len(batch_y[i])] = batch_y[i]
        y_mask[i, :len(batch_y[i])] = 1.0

      feed_dict = {tf_encoder_input: x,
                   tf_decoder_input: y[:, :-1],
                   tf_decoder_target: y[:, 1:],
                   tf_decoder_mask: y_mask}
      _, train_loss = sess.run([train_step, loss], feed_dict=feed_dict)
      step += 1
      print('step: %d, train_loss: %f' % (step, train_loss))

      if (step % save_step) == 0:
        train_model.save_model(sess, model_file, step)
        print('----- Saving Model -----')

    train_model.save_model(sess, model_file, step)
    print('----- Saving Model -----')


def run_test(sampling):
  # Inputs
  dictionary = DP.read_dictionary(dict_file)
  inverse_dictionary = {dictionary[key]:key for key in dictionary}

  # Test Data Processing
  test = []
  test_max_seq_length = 0
  with open(test_file) as data:
    for sentence in data:
      sentence = re.sub(r'\n', '', sentence) 
      words = sentence.split()
      translated_words = [dictionary[word] if word in dictionary else dictionary['<UNK>'] 
                        for word in words]
      if len(translated_words) > test_max_seq_length:
        test_max_seq_length = len(translated_words)
      test.append(translated_words)    
  
  # Parameters
  N_input = len(test)
  vocab_size = len(dictionary)
  print('Total testing steps: %d' % N_input)

  graph = tf.Graph()
  with graph.as_default():
    params['batch_size'] = 1
    test_model = RnnModel_Attention(
              is_training= False,
              vocab_size = vocab_size,
              N_hidden = N_hidden,
              N_caption_step = test_max_seq_length,
              **params)

    tf_encoder_input, tf_decoder_input, captions = test_model.build_test_model(sampling)

  with tf.Session(graph= graph) as sess:
    sess.run(tf.global_variables_initializer())
    step = test_model.restore_model(sess, model_file)
    print('Restore the model with step %d' % (step))
    
    result = []
    for idx, sentence in enumerate(test):
      caption = [BOS_tag]  
      x = np.full((1, test_max_seq_length), dictionary[EOS_tag])
      x[0,:len(sentence)] = sentence
      
      begin = np.array([dictionary[caption[0]]]).reshape(1, 1)
      feed_dict = {tf_encoder_input: x, tf_decoder_input: begin}
      predictions = sess.run(captions, feed_dict= feed_dict)
      for word_idx in predictions:
        word = inverse_dictionary[word_idx]
        if word == EOS_tag:
          break
        caption.append(word)

      caption = caption[1:]
      result.append(caption)
            
    return result


def write_result(data):
  folder = os.path.dirname(model_file) + '/'
  with open(folder + 'result.txt', 'w') as f:
    for sentence in data:
      for word in sentence:
        f.write('%s ' % word)
      f.write('\n')


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description= 'sequence to sequence model')
  parser.add_argument('--train',
              action= 'store_true',
              default= False,
              help= 'train task')
  parser.add_argument('--test',
              action= 'store_true',
              default= False,
              help= 'test task')
  parser.add_argument('--sampling',
              action= 'store_true',
              default= False,
              help= 'test task')

  arg = parser.parse_args()
  if arg.train:
    run_train()
  if arg.test:
    result = run_test(arg.sampling)
    write_result(result)
