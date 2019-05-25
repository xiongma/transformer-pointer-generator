# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
date: 2019/5/21
mail: cally.maxiong@gmail.com
page: http://www.cnblogs.com/callyblog/
'''

import tensorflow as tf
from utils import calc_num_batches

def _load_vocab(vocab_fpath):
    '''Loads vocabulary file and returns idx<->token maps
    vocab_fpath: string. vocabulary file path.
    Note that these are reserved
    0: <pad>, 1: <unk>, 2: <s>, 3: </s>

    Returns
    two dictionaries.
    '''
    vocab = []
    with open(vocab_fpath, 'r', encoding='utf-8') as f:
        for line in f:
            vocab.append(line.replace('\n', ''))
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}

    return token2idx, idx2token

def _load_data(fpaths, maxlen1, maxlen2):
    '''Loads source and target data and filters out too lengthy samples.
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.

    Returns
    sents1: list of source sents
    sents2: list of target sents
    '''
    sents1, sents2 = [], []
    for fpath in fpaths.split('|'):
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                splits = line.split(',')
                if len(splits) != 2: continue
                sen1 = splits[1].replace('\n', '').strip()
                sen2 = splits[0].strip()
                if len(list(sen1)) + 1 > maxlen1: continue
                if len(list(sen2)) + 1 > maxlen2: continue

                sents1.append(sen1.encode('utf-8'))
                sents2.append(sen2.encode('utf-8'))

    return sents1, sents2

def _encode(inp, token2idx, type):
    '''Converts string to number. Used for `generator_fn`.
    inp: 1d byte array.
    type: "x" (source side) or "y" (target side)
    dict: token2idx dictionary

    Returns
    list of numbers
    '''
    inp = inp.decode('utf-8')
    if type == 'x':
        tokens = list(inp)

        return [token2idx.get(token, token2idx['<unk>']) for token in tokens]

    else:
        inputs = ['<s>'] + list(inp)
        target = list(inp) + ['</s>']

        return [token2idx.get(token, token2idx['<unk>']) for token in inputs], [token2idx.get(token, token2idx['<unk>']) for token in target]

def _generator_fn(sents1, sents2, vocab_fpath):
    '''Generates training / evaluation data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.

    yields
    xs: tuple of
        x: list of source token ids in a sent
        x_seqlen: int. sequence length of x
        sent1: str. raw source (=input) sentence
    labels: tuple of
        decoder_input: decoder_input: list of encoded decoder inputs
        y: list of target token ids in a sent
        y_seqlen: int. sequence length of y
        sent2: str. target sentence
    '''
    token2idx, _ = _load_vocab(vocab_fpath)
    for sent1, sent2 in zip(sents1, sents2):
        x = _encode(sent1, token2idx, "x")

        inputs, targets = _encode(sent2, token2idx, "y")

        yield (x, sent1.decode('utf-8')), (inputs, targets, sent2.decode('utf-8'))

def _input_fn(sents1, sents2, vocab_fpath, batch_size, gpu_nums, shuffle=False):
    '''Batchify data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    '''
    shapes = (([None], ()),
              ([None], [None], ()))
    types = ((tf.int32, tf.string),
             (tf.int32, tf.int32, tf.string))
    paddings = ((0, ''),
                (0, 0, ''))

    dataset = tf.data.Dataset.from_generator(
        _generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(sents1, sents2, vocab_fpath))  # <- arguments for generator_fn. converted to np string arrays

    if shuffle: # for training
        dataset = dataset.shuffle(128*batch_size*gpu_nums)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size*gpu_nums, shapes, paddings)

    return dataset

def get_batch(fpath1, maxlen1, maxlen2, vocab_fpath, batch_size, gpu_nums, shuffle=False):
    '''Gets training / evaluation mini-batches
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    batches
    num_batches: number of mini-batches
    num_samples
    '''
    sents1, sents2 = _load_data(fpath1, maxlen1, maxlen2)
    batches = _input_fn(sents1, sents2, vocab_fpath, batch_size, gpu_nums, shuffle=shuffle)
    num_batches = calc_num_batches(len(sents1), batch_size)
    return batches, num_batches, len(sents1), [sent.decode('utf-8') for sent in sents2]
