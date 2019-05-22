# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
date: 2019/5/21
mail: cally.maxiong@gmail.com
page: http://www.cnblogs.com/callyblog/
'''
import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    # prepro
    parser.add_argument('--vocab_size', default=10598, type=int)

    # train
    ## files
    parser.add_argument('--train', default='data/train.csv',
                             help="data for train")

    parser.add_argument('--eval', default='data/test.csv',
                             help="data for evaluation")
    parser.add_argument('--test', default='data/test_summary.csv',
                             help="data for calculate rouge score")

    ## vocabulary
    parser.add_argument('--vocab', default='vocab',
                        help="vocabulary file path")

    # training scheme
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=32, type=int)

    parser.add_argument('--lr', default=0.0005, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--logdir', default="log/1", help="log directory")
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--evaldir', default="eval/1", help="evaluation dir")

    # model
    parser.add_argument('--d_model', default=512, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff', default=2048, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=6, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")
    parser.add_argument('--maxlen1', default=128, type=int,
                        help="maximum length of a source sequence")
    parser.add_argument('--maxlen2', default=20, type=int,
                        help="maximum length of a target sequence")
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--beam_size', default=4, type=int,
                        help="beam size")
    parser.add_argument('--multi_gpu', default=False, type=bool,
                        help="whether use multi gpu to train model")
    parser.add_argument('--gpu_nums', default=4, type=int,
                        help="gpu amount, which can allow how many gpu to train this model")