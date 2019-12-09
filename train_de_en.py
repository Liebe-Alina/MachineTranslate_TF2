#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:  Chenyang WANG
@license: MIT Licence 
@file: train.py 
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

from All_de_en import create_masks, Transformer, CustomSchedule, loss_function
from hyperparams import Hparams
import tensorflow as tf
import time
from Try_tokenizer_corpus import get_batch_data


def train():

    parser = Hparams.parser
    hp = parser.parse_args()

    num_layers = hp.num_layers
    d_model = hp.d_model
    dff = hp.dff
    num_heads = hp.num_heads
    dropout_rate = hp.dropout_rate
    print('Finish loading hyper-parameters!')

    train_dataset = get_batch_data()
    learning_rate = CustomSchedule(d_model)
    input_vocab_size = hp.tokenizer_de.vocab_size + 2
    target_vocab_size = hp.tokenizer_en.vocab_size + 2
    print('Finish loading word vectors and batch data')

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    transformer = Transformer(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size,
                              src=input_vocab_size,
                              target=target_vocab_size,
                              rate=dropout_rate)
    checkpoint_path = hp.checkpoint_dir_de

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    EPOCHS = hp.epochs

    train_step_signature = [
        tf.TensorSpec(shape=(None, 40), dtype=tf.int32),
        tf.TensorSpec(shape=(None, 40), dtype=tf.int32),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
      tar_inp = tar[:, :-1]
      tar_real = tar[:, 1:]
      enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

      with tf.GradientTape() as tape:
        predictions, _ = transformer.call(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions, loss_object)

      gradients = tape.gradient(loss, transformer.trainable_variables)
      optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

      train_loss(loss)
      train_accuracy(tar_real, predictions)

    print('Start Training')
    for epoch in range(EPOCHS):
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        # inp -> German, tar -> english
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)

            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch,
                                                                             train_loss.result(),
                                                                             train_accuracy.result()))
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                             ckpt_save_path))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                    train_loss.result(),
                                                    train_accuracy.result()))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


if __name__ == '__main__':
    train()
