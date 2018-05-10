#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 00:33:44 2018

@author: halley
"""

import tensorflow as tf
from rnn_models import RnnModel_Attention
import numpy as np
from dataset import DataSet
import data_preprocessing as DP
import argparse
import json
import os

##### Training file path #####

dict_file = 'dictionary.txt'
train_label_file = './translated_training_label.json'
train_path = './MLDS_hw2_1_data/training_data/feat/'
test_path = './MLDS_hw2_1_data/testing_data/feat/'
test_id_path = './MLDS_hw2_1_data/testing_id.txt'
model_file = './s2s_attention/model.ckpt'

##############################

##### Constants #####

EOS_tag = '<EOS>'
BOS_tag = '<BOS>'
UNK_tag = '<UNK>'

#####################

##### Parameters #####

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

######################

def run_train():
    # Inputs
    dictionary = DP.read_dictionary(dict_file)
    train_label = DP.read_train(train_label_file)
    train = DataSet(train_path, train_label, len(dictionary), dictionary[BOS_tag], dictionary[EOS_tag])

    # Parameters
    N_input = train.datalen
    N_iter = N_input * N_epoch // batch_size
    print('Total training steps: %d' % N_iter)

    # Model
    graph = tf.Graph()
    with graph.as_default():
        train_model = RnnModel_Attention(
                        is_training= True,
                        image_dim = train.feat_dim,
                        vocab_size = train.vocab_size,
                        N_hidden = N_hidden,
                        N_video_step = train.feat_timestep,
                        N_caption_step = train.max_seq_len,
                        **params)
        tf_video, tf_decoder_input, tf_decoder_target, loss, train_step = train_model.build_train_model()

    with tf.Session(graph= graph) as sess:
        sess.run(tf.global_variables_initializer())
        step = train_model.restore_model(sess, model_file)

        while step < N_iter:
            batch_x, batch_y = train.next_batch(batch_size=batch_size)

            y = np.full((batch_size, train.max_seq_len), dictionary[EOS_tag])
            for i, caption in enumerate(batch_y):
                y[i,:len(caption)] = caption

            feed_dict = {tf_video: batch_x, tf_decoder_input: y[:, :-1], tf_decoder_target: y[:, 1:]}
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

    ID = []
    with open(test_id_path) as f:
        for line in f:
            ID.append(line[:-1])

    features = []
    for id in ID:
        feature = np.load(test_path + id + '.npy')
        features.append(feature)
    features = np.asarray(features)
    
    # Parameters
    N_input = len(ID)
    feat_timestep = features.shape[1]
    feat_dim = features.shape[2]
    vocab_size = len(dictionary)
    print('Total testing steps: %d' % N_input)

    graph = tf.Graph()
    with graph.as_default():
        params['batch_size'] = 1
        test_model = RnnModel_Attention(
                        is_training= False,
                        image_dim = feat_dim,
                        vocab_size = vocab_size,
                        N_hidden = N_hidden,
                        N_video_step = feat_timestep,
                        N_caption_step = max_seq_len,
                        **params)

        tf_video, tf_decoder_input, captions = test_model.build_test_model(sampling)

    with tf.Session(graph= graph) as sess:
        sess.run(tf.global_variables_initializer())
        step = test_model.restore_model(sess, model_file)
        print('Restore the model with step %d' % (step))
        
        result = []
        for idx, x in enumerate(features):
            caption = {}
            caption['caption'] = [BOS_tag]
            caption['id'] = ID[idx]

            x = x.reshape(1, feat_timestep, feat_dim)
            
            begin = np.array([dictionary[caption['caption'][0]]]).reshape(1, 1)
            feed_dict = {tf_video: x, tf_decoder_input: begin}
            predictions = sess.run(captions, feed_dict= feed_dict)
            for word_idx in predictions:
                word = inverse_dictionary[word_idx]
                if word == EOS_tag:
                    break
                caption['caption'].append(word)

            caption['caption'] = caption['caption'][1:]
            result.append(caption)
            
        return result

def write_result(data):
    folder = os.path.dirname(model_file) + '/'
    with open(folder + 'result.json', 'w') as f:
        json.dump(data, f, sort_keys= True, indent= 4)

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
