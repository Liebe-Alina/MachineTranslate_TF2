from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from hyperparams import Hparams
from All_de_en import Transformer, create_masks
from nltk.translate.bleu_score import corpus_bleu
import tqdm
import regex
import codecs
import os
import time

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
print('Begin Evaluation')


def evaluate(inp_sentence):
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
    out = [i for i in a if isinstance(i, list) == False and i < bpemb_en.vocab_size]
    x = bpemb_en.decode_ids(out)
#    x = bpemb_en.decode_ids(i for i in a if i < bpemb_en.vocab_size)
    result.append(x)
    return x


def cal_Blue():
    def _refine(line):
        line = regex.sub("<[^>]+>", "", line)
        line = regex.sub("[^\s\p{Latin}']", "", line)
        return line.strip()

#    de_sents_test = [_refine(line) for line in codecs.open('./data/IWSLT16.TED.tst2014.de-en.de.xml', 'r', 'utf-8').read().split("\n") if
#                line and line[:4] == "<seg"]
#    en_sents_test = [_refine(line) for line in codecs.open('./data/IWSLT16.TED.tst2014.de-en.en.xml', 'r', 'utf-8').read().split("\n") if
#                line and line[:4] == "<seg"]

    de_sents_test = [regex.sub("[^\s\p{Latin}']", "", line) for line in
                    codecs.open('./data/newstest2016.de', 'r', 'utf-8').read().split("\n") if
                    line and line[0] != "<"]
    en_sents_test = [regex.sub("[^\s\p{Latin}']", "", line) for line in
                    codecs.open('./data/newstest2016.en', 'r', 'utf-8').read().split("\n") if
                    line and line[0] != "<"]
    assert len(de_sents_test) == len(en_sents_test)
    list_of_refs = []
    hypotheses = []
    if not os.path.exists('results'): os.mkdir('results')
    with codecs.open("results/result", "a", "utf-8") as fout:
        for i in tqdm.tqdm(range(len(de_sents_test))):
            pre = evaluate(de_sents_test[i])
            if isinstance(pre, str):
                fout.write("- source: " + de_sents_test[i] + '\n')
                fout.write("- expected: " + en_sents_test[i] + "\n")
                fout.write("- got: " + pre + "\n\n")
                fout.flush()
                ref = en_sents_test[i].split()
                hypothesis = pre.split()
                if len(ref) > 3 and len(hypothesis) > 3:
                    list_of_refs.append([ref])
                    hypotheses.append(hypothesis)
            if (len(de_sents_test) - i) % 100 == 0:
                print("remain: {}".format(len(de_sents_test) - i))
        score = corpus_bleu(list_of_refs, hypotheses)
        fout.write("Bleu Score = " + str(100 * score))


if __name__ == '__main__':
    cal_Blue()

    """
    
    src = de_sents_test[:150]
    pre = evaluate(de_sents_test[:150])
    real = en_sents_test[:150]

        for i in range(64, len(de_sents_test), 64):
        if (i + 64) < len(de_sents_test):
            j = i + 64
        else:
            j = len(de_sents_test)
    """