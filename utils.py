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
from tqdm import tqdm

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

def postprocess(hypotheses):
    '''Processes translation outputs.
    hypotheses: list of encoded predictions
    idx2token: dictionary

    Returns
    processed hypotheses
    '''
    _hypotheses = []
    for h in hypotheses:
        h = str(h)
        h = h.replace('<s>', '')
        h = h.replace('</s>', '')
        h = h.replace('<pad>', '')
        _hypotheses.append(h)

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

def get_hypotheses(num_batches, num_samples, sess, model, beam_search, tensor, handle_placehoder, handle):
    '''Gets hypotheses.
    num_batches: scalar.
    num_samples: scalar.
    sess: tensorflow sess object
    tensor: target tensor to fetch
    dict: idx2token dictionary

    Returns
    hypotheses: list of sents
    '''
    hypotheses, all_targets = [], []
    for _ in tqdm(range(num_batches)):
        articles, targets = sess.run(tensor, feed_dict={handle_placehoder: handle})
        memories = sess.run(model.enc_output, feed_dict={model.x: articles})
        for article, memory in zip(articles, memories):
            summary = beam_search.search(sess, article, memory)
            summary = postprocess(summary)
            hypotheses.append(summary)
        all_targets.extend([target.decode('utf-8') for target in targets])

    return hypotheses[:num_samples], all_targets[:num_samples]

def calc_rouge(rouge, references, models, global_step, logdir):
    """
    calculate rouge score
    :param references: reference sentences
    :param models: model sentences
    :param global_step: global step
    :param logdir: log dir
    :return: rouge score
    """
    # delete symbol
    references = [reference.replace('</s>', '') for reference in references]

    # calculate rouge score
    rouge1_scores = [_rouge(rouge, model, reference, type='rouge1') for model, reference in zip(models, references)]
    rouge2_scores = [_rouge(rouge, model, reference, type='rouge2') for model, reference in zip(models, references)]
    rougel_scores = [_rouge(rouge, model, reference, type='rougel') for model, reference in zip(models, references)]

    # get rouge score
    rouge1_score = sum(rouge1_scores) / len(rouge1_scores)
    rouge2_score = sum(rouge2_scores) / len(rouge2_scores)
    rougel_score = sum(rougel_scores) / len(rouge2_scores)

    # write result
    with open(os.path.join(logdir, 'rouge'), 'a', encoding='utf-8') as f:
        f.write('global step: {}, ROUGE 1: {}, ROUGE 2: {}, ROUGE L: {}\n'.format(str(global_step), str(rouge1_score),
                                                                                  str(rouge2_score), str(rougel_score)))
    return rouge1_score

def _rouge(rouge, model, reference, type='rouge1'):
    """
    calculate rouge socore
    :param rouge: sumeval instance
    :param model: model prediction, list
    :param reference: reference
    :param type: rouge1, rouge2, rougel
    :return: rouge 1 score
    """
    scores = None
    if type == 'rouge1':
        scores = [rouge.rouge_n(summary=m, references=reference, n=1) for m in model]

    if type == 'rouge2':
        scores = [rouge.rouge_n(summary=m, references=reference, n=2) for m in model]

    if type == 'rougel':
        scores = [rouge.rouge_l(summary=m, references=reference) for m in model]

    return max(scores)

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
    :param xs: articles
    :param ys: summaries
    :param gpu_nums: gpu numbers
    :return: split input by gpu numbers
    """
    import tensorflow as tf
    xs = [tf.split(x, num_or_size_splits=gpu_nums, axis=0) for x in xs]
    ys = [tf.split(y, num_or_size_splits=gpu_nums, axis=0) for y in ys]

    return [(xs[0][i], xs[1][i]) for i in range(gpu_nums)], [(ys[0][i], ys[1][i], ys[2][i]) for i in range(gpu_nums)]