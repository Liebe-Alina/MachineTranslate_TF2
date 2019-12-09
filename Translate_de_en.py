#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:  Chenyang WANG
@license: MIT Licence 
@file: Translate.py
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


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
from hyperparams import Hparams
from All_de_en import Transformer, create_masks
import os
from nltk.translate.bleu_score import sentence_bleu
import regex
import codecs
from Try_tokenizer_corpus import get_batch_data


parser = Hparams.parser
hp = parser.parse_args()
bpemb_en = hp.tokenizer_en
bpemb_de = hp.tokenizer_de
transformer = Transformer(hp.num_layers, hp.d_model, hp.num_heads, hp.dff,
                          bpemb_de.vocab_size+2, bpemb_en.vocab_size+2,
                          src=bpemb_de.vocab_size+2,
                          target=bpemb_en.vocab_size+2,
                          rate=hp.dropout_rate)
checkpoint_path = hp.checkpoint_dir_de
ckpt = tf.train.Checkpoint(transformer=transformer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

ckpt.restore(ckpt_manager.latest_checkpoint)
print('Latest checkpoint restored!!')


def evaluate(inp_sentence):

    start_token = [bpemb_de.vocab_size]
    end_token = [bpemb_de.vocab_size + 1]

    # inp sentence is German, hence adding the start and end token
    inp_sentence = start_token + bpemb_de.encode_ids(inp_sentence) + end_token
    inp_sentence = tf.convert_to_tensor(inp_sentence)
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [bpemb_en.vocab_size]
    decoder_input = tf.convert_to_tensor(decoder_input)
    output = tf.expand_dims(decoder_input, 0)

    for i in range(40):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer.call(encoder_input,
                                                          output,
                                                          False,
                                                          enc_padding_mask,
                                                          combined_mask,
                                                          dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == bpemb_en.vocab_size + 1:
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def plot_attention_weights(attention, sentence, result, layer):
    fig = plt.figure(figsize=(16, 8))

    sentence = bpemb_de.encode_ids(sentence)

    attention = tf.squeeze(attention[layer], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        # plot the attention weights
        ax.matshow(attention[head][:-1, :], cmap='magma')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(sentence) + 2))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(result) - 1.5, -0.5)

        ax.set_xticklabels(
            ['<start>'] + [bpemb_de.decode_ids([i]) for i in sentence] + ['<end>'],
            fontdict=fontdict, rotation=90)

        ax.set_yticklabels([bpemb_en.decode_ids([i]) for i in result
                            if i < bpemb_en.vocab_size],
                           fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head + 1))

    plt.tight_layout()
    plt.show()


def translate(sentence, plot=''):

    result, attention_weights = evaluate(sentence)
    result = result.numpy()
    result = result.tolist()
    predicted_sentence = bpemb_en.decode_ids([i for i in result
                                              if i < bpemb_en.vocab_size])

    print('\nInput: {}'.format(sentence))
    print('\nPredicted translation: {}'.format(predicted_sentence))
    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    flag = 1
    if flag == 1:
        while flag:
            src_language = input('Please input the sentence you want to translate:')
            translate(src_language)
            flag = int(input('Do you want to continue? \n0 for no & 1 for yes'))
        print("***************************Thanks for your using.***************************")
    elif flag == 0:
        translate('Heute lernte David das vertiefte Lernen für maschinelle Übersetzung und er glaubt, dass es ihm gut geht..', plot='decoder_layer4_block2')
        print('The true Translation is: Today, David learned about deep learning for machine translation, and he thinks he is doing well.')
