# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
date: 2019/5/21
mail: cally.maxiong@gmail.com
page: http://www.cnblogs.com/callyblog/
'''
import json
import logging
import os

import jieba

logging.basicConfig(level=logging.INFO)

def calc_num_batches(total_num, batch_size):
    '''Calculates the number of batches.
    total_num: total sample number
    batch_size

    Returns
    number of batches, allowing for remainders.'''
    return total_num // batch_size + int(total_num % batch_size != 0)

def convert_idx_to_token_tensor(inputs, idx2token):
    '''Converts int32 tensor to string tensor.
    inputs: 1d int32 tensor. indices.
    idx2token: dictionary

    Returns
    1d string tensor.
    '''
    import tensorflow as tf
    def my_func(inputs):
        return " ".join(idx2token[elem] for elem in inputs)

    return tf.py_func(my_func, [inputs], tf.string)

def postprocess(hypotheses, idx2token):
    '''Processes translation outputs.
    hypotheses: list of encoded predictions
    idx2token: dictionary

    Returns
    processed hypotheses
    '''
    _hypotheses = []
    for h in hypotheses:
        sent = "".join(idx2token[idx] for idx in h)
        sent = sent.split("</s>")[0].strip()
        sent = sent.replace("▁", " ") # remove bpe symbols
        _hypotheses.append(' '.join(list(sent.strip())))
    return _hypotheses

def save_hparams(hparams, path):
    '''Saves hparams to path
    hparams: argsparse object.
    path: output directory.

    Writes
    hparams as literal dictionary to path.
    '''
    if not os.path.exists(path): os.makedirs(path)
    hp = json.dumps(vars(hparams))
    with open(os.path.join(path, "hparams"), 'w') as fout:
        fout.write(hp)

def load_hparams(parser, path):
    '''Loads hparams and overrides parser
    parser: argsparse parser
    path: directory or file where hparams are saved
    '''
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    d = open(os.path.join(path, "hparams"), 'r').read()
    flag2val = json.loads(d)
    for f, v in flag2val.items():
        parser.f = v

def save_variable_specs(fpath):
    '''Saves information about variables such as
    their name, shape, and total parameter number
    fpath: string. output file path

    Writes
    a text file named fpath.
    '''
    import tensorflow as tf
    def _get_size(shp):
        '''Gets size of tensor shape
        shp: TensorShape

        Returns
        size
        '''
        size = 1
        for d in range(len(shp)):
            size *=shp[d]
        return size

    params, num_params = [], 0
    for v in tf.global_variables():
        params.append("{}==={}".format(v.name, v.shape))
        num_params += _get_size(v.shape)
    print("num_params: ", num_params)
    with open(fpath, 'w') as fout:
        fout.write("num_params: {}\n".format(num_params))
        fout.write("\n".join(params))
    logging.info("Variables info has been saved.")

def get_hypotheses(num_batches, num_samples, sess, tensor, dict):
    '''Gets hypotheses.
    num_batches: scalar.
    num_samples: scalar.
    sess: tensorflow sess object
    tensor: target tensor to fetch
    dict: idx2token dictionary

    Returns
    hypotheses: list of sents
    '''
    hypotheses = []
    for _ in range(num_batches):
        h = sess.run(tensor)
        hypotheses.extend(h.tolist())
    hypotheses = postprocess(hypotheses, dict)

    return hypotheses[:num_samples]

def calc_rouge(references, models, global_step, logdir):
    """
    calculate rouge score
    :param references: reference sentences
    :param models: model sentences
    :param global_step: global step
    :param logdir: log dir
    :return: rouge score
    """
    replaces = [' ', '<s>', '</s>', '<pad>', '<unk>']
    models_ = []
    for model in models:
        for rep in replaces:
            model = model.replace(rep, '')

        models_.append(model)

    rouge1_scores = [rouge_1(model, reference) for model, reference in zip(models, references)]
    rouge2_scores = [rouge_2(model, reference) for model, reference in zip(models, references)]

    rouge1_score = sum(rouge1_scores) / len(rouge1_scores)
    rouge2_score = sum(rouge2_scores) / len(rouge2_scores)

    with open(os.path.join(logdir, 'rouge'), 'a', encoding='utf-8') as f:
        f.write('global step: {}, ROUGE 1: {}, ROUGE 2: {}\n'.format(str(global_step), str(rouge1_score), str(rouge2_score)))

def rouge_1(model, reference):
    """
    calculate rouge 1 score
    :param model: model output
    :param reference: reference
    :return: rouge 1 score
    """
    terms_reference = jieba.cut(reference)
    terms_model = jieba.cut(model)
    grams_reference = list(terms_reference)
    grams_model = list(terms_model)
    temp = 0
    ngram_all = len(grams_reference)
    for x in grams_reference:
        if x in grams_model: temp = temp + 1
    rouge_1 = temp / ngram_all
    return rouge_1

def rouge_2(model, reference):
    """
    calculate rouge 2 score
    :param model: model output
    :param reference: reference
    :return: rouge 2 score
    """
    terms_reference = jieba.cut(reference)
    terms_model = jieba.cut(model)
    grams_reference = list(terms_reference)
    grams_model = list(terms_model)
    gram_2_model = []
    gram_2_reference = []
    temp = 0
    ngram_all = len(grams_reference) - 1
    for x in range(len(grams_model) - 1):
        gram_2_model.append(grams_model[x] + grams_model[x + 1])
    for x in range(len(grams_reference) - 1):
        gram_2_reference.append(grams_reference[x] + grams_reference[x + 1])
    for x in gram_2_model:
        if x in gram_2_reference: temp = temp + 1
    rouge_2 = temp / ngram_all

    return rouge_2

def import_tf(gpu_list):
    """
    import tensorflow, set tensorflow graph load device
    :param gpu_list: GPU list
    :return: tensorflow instance
    """
    import tensorflow as tf
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_list)

    return tf

def split_input(xs, ys, gpu_nums):
    """
    split input
    :param xs:
    :param ys:
    :param gpu_nums:
    :return: split input by gpu numbers
    """
    import tensorflow as tf

    xs = [tf.split(x, gpu_nums, axis=0) for x in xs]
    ys = [tf.split(y, gpu_nums, axis=0) for y in ys]

    return [(xs[0][i], xs[1][i]) for i in range(gpu_nums)], [(ys[0][i], ys[1][i], ys[2][i]) for i in range(gpu_nums)]