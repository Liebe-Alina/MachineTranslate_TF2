#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:  Chenyang WANG
@license: MIT Licence 
@file: hyperparams.py 
@time: 2019/11
@contact: wcy1705@outlook.com
@software: PyCharm 
@description: Hyper-parameters of Model

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


import argparse
from bpemb import BPEmb

class Hparams:
    parser = argparse.ArgumentParser()

    bpemb_en = BPEmb(lang="en", dim=50)
    bpemb_de = BPEmb(lang='de', dim=50)

    # preprocess
    parser.add_argument('--BUFFER_SIZE', default=10000)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--maxlen', default=40, help='max length of sentences')
    parser.add_argument('--tokenizer_de', default=bpemb_de, help='encoding method')
    parser.add_argument('--tokenizer_en', default=bpemb_en, help='decoding method')

    # train
    parser.add_argument('--num_layers', default=4, help='blocks number of encoder and decoder')
    parser.add_argument('--d_model', default=128)
    parser.add_argument('--dff', default=512)
    parser.add_argument('--num_heads', default=8)
    parser.add_argument('--dropout_rate', default=0.1)
    parser.add_argument('--checkpoint_dir', default='./checkpoints/train')
    parser.add_argument('--checkpoint_dir_de', default='./checkpoints/de_en')
    parser.add_argument('--epochs', default=10)
