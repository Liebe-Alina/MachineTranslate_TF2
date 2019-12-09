#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:  Chenyang WANG
@license: MIT Licence 
@file: Try_tokenizer_corpus.py 
@time: 2019/12
@contact: wcy1705@outlook.com
@software: PyCharm 
@description:

         ,'~~^- -~~\
        (          ,,)
         \  ''    .|_
         ` C       .-'
          ` ,    _ '
            ,--~,
           /~    \       
          /     . ~/--,  ___No BUG!___
         ,    .__~-_--__[     WCY     |
         |___/ ,/\ /~|_______________|
         \_____-\///~  ||         ||
                 ~~
"""

import tensorflow_datasets as tfds
from bpemb import BPEmb
import regex
import codecs
import numpy as np
import tensorflow as tf
from hyperparams import Hparams


"""
BUFFER_SIZE = 20000
BATCH_SIZE = 64
bpemb_en = BPEmb(lang="en", dim=50)
bpemb_de = BPEmb(lang='de', dim=50)
sample = 'Machine Translate is awesome!'
encoded_sen = bpemb_en.encode(sample)
encoded = bpemb_en.encode_ids(sample)
for i in range(len(encoded_sen)):
    print('{} --> {}'.format(encoded_sen[i], encoded[i]))
"""

parser = Hparams.parser
hp = parser.parse_args()


def create_datasets(lang_src, lang_tar, bpemb_de, bpemb_en):

    assert len(lang_src) == len(lang_tar)
    src_list = []
    tar_list = []
    Sources, Targets = [], []
    for src, tar in zip(lang_src, lang_tar):
        x = [bpemb_de.vocab_size] + bpemb_de.encode_ids(src) + [bpemb_de.vocab_size + 1]
        y = [bpemb_en.vocab_size] + bpemb_en.encode_ids(tar) + [bpemb_en.vocab_size + 1]
        if max(len(x), len(y)) <= hp.maxlen:
            src_list.append(np.array(x))
            tar_list.append(np.array(y))
            Sources.append(src)
            Targets.append(tar)

    # Padding
    X = np.zeros([len(src_list), hp.maxlen], np.int32)
    Y = np.zeros([len(tar_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(src_list, tar_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))

    return X, Y, Sources, Targets, bpemb_de.vocab_size, bpemb_en.vocab_size


def load_data(source_dir, tar_dir):
    de_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(source_dir, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    en_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(tar_dir, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    x_train, y_train, _, _, a, b = create_datasets(de_sents, en_sents, hp.tokenizer_de, hp.tokenizer_en)
    return x_train, y_train, a, b


def get_batch_data():
    Src, Tar, a, b = load_data('./data/train.tags.de-en.de', './data/train.tags.de-en.en')
    Tar = tf.convert_to_tensor(Tar, tf.int32)
    Src = tf.convert_to_tensor(Src, tf.int32)
    input_queues = tf.data.Dataset.from_tensor_slices((Src, Tar))
    train_ds = input_queues.batch(64)
    return train_ds


def get_test_data():
    Src, Tar, a, b = load_data('./data/newstest2013.de', './data/newstest2013.en')
    Tar = tf.convert_to_tensor(Tar, tf.int32)
    Src = tf.convert_to_tensor(Src, tf.int32)
    input_queues = tf.data.Dataset.from_tensor_slices(Src)
    test_data = input_queues.batch(64)
    return test_data