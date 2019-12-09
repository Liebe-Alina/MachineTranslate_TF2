from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from hyperparams import Hparams
from All_de_en import Transformer, create_masks
from nltk.translate.bleu_score import corpus_bleu
from Try_tokenizer_corpus import get_test_data
import regex
import codecs
import os
import time


def evaluate(inp_sentence_dataset):
    parser = Hparams.parser
    hp = parser.parse_args()
    bpemb_en = hp.tokenizer_en
    bpemb_de = hp.tokenizer_de
    transformer = Transformer(hp.num_layers, hp.d_model, hp.num_heads, hp.dff,
                              bpemb_de.vocab_size + 2, bpemb_en.vocab_size + 2,
                              src=bpemb_de.vocab_size + 2,
                              target=bpemb_en.vocab_size + 2,
                              rate=hp.dropout_rate)
    checkpoint_path = hp.checkpoint_dir_de
    ckpt = tf.train.Checkpoint(transformer=transformer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')
    start_token = [bpemb_de.vocab_size]
    end_token = [bpemb_de.vocab_size + 1]
    result = []
    start_time = time.time()
    for k in range(len(inp_sentence_dataset)):

        # inp sentence is German, hence adding the start and end token
        inp_sentence = start_token + bpemb_de.encode_ids(inp_sentence_dataset[k]) + end_token
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
            predictions, _ = transformer.call(encoder_input,
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
                a = tf.squeeze(output, axis=0).numpy().tolist()
                break
            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.

            output = tf.concat([output, predicted_id], axis=-1)
            a = output.numpy().tolist()
        x = bpemb_en.decode_ids([i for i in a if i < bpemb_en.vocab_size])
        result.append(x)
        if k % 10 == 0:
            print('every 10 sentences use time: {}'.format(time.time()-start_time))
            start_time = time.time()
    return result


def cal_Blue():
    de_sents_test = [regex.sub("[^\s\p{Latin}']", "", line) for line in
                    codecs.open('./data/newstest2013.de', 'r', 'utf-8').read().split("\n") if
                    line and line[0] != "<"]
    en_sents_test = [regex.sub("[^\s\p{Latin}']", "", line) for line in
                    codecs.open('./data/newstest2013.en', 'r', 'utf-8').read().split("\n") if
                    line and line[0] != "<"]
    assert len(de_sents_test) == len(en_sents_test)
    src = de_sents_test
    pre = evaluate(de_sents_test)
    real = en_sents_test
    list_of_refs = []
    hypotheses = []
    if not os.path.exists('results'): os.mkdir('results')
    with codecs.open("results/result", "w", "utf-8") as fout:
        for src_, target, pred in zip(src, real, pre):  # sentence-wise
            fout.write("- source: " + src_ + '\n')
            fout.write("- expected: " + target + "\n")
            fout.write("- got: " + pred + "\n\n")
            fout.flush()
            ref = target.split()
            hypothesis = pred.split()
            if len(ref) > 3 and len(hypothesis) > 3:
                list_of_refs.append([ref])
                hypotheses.append(hypothesis)
        score = corpus_bleu(list_of_refs, hypotheses)
        fout.write("Bleu Score = " + str(100 * score))


if __name__ == '__main__':
    cal_Blue()