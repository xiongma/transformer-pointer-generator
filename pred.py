# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
date: 2019/5/21
mail: cally.maxiong@gmail.com
page: http://www.cnblogs.com/callyblog/
'''
import os

from beam_search import BeamSearch
from data_load import _load_vocab
from hparams import Hparams
from model import Transformer

def import_tf(device_id=-1, verbose=False):
    """
    import tensorflow, set tensorflow graph load device, set tensorflow log level, return tensorflow instance
    :param device_id: GPU id
    :param verbose: tensorflow logging level
    :return: tensorflow instance
    """
    # set visible gpu, -1 is cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if device_id < 0 else str(device_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if verbose else '3'
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.DEBUG if verbose else tf.logging.ERROR)
    return tf

class Prediction:
    def __init__(self, args):
        """
        :param model_dir: model dir path
        :param vocab_file: vocab file path
        """
        self.tf = import_tf(0)

        self.args = args
        self.model_dir = args.logdir
        self.vocab_file = args.vocab
        self.token2idx, self.idx2token = _load_vocab(args.vocab)

        hparams = Hparams()
        parser = hparams.parser
        self.hp = parser.parse_args()

        self.model = Transformer(self.hp)

        self._add_placeholder()
        self._init_graph()

    def _init_graph(self):
        """
        init graph
        """
        self.ys = (self.input_y, None, None)
        self.xs = (self.input_x, None)
        self.memory = self.model.encode(self.xs, False)[0]
        self.logits = self.model.decode(self.xs, self.ys, self.memory, False)[0]

        ckpt = self.tf.train.get_checkpoint_state(self.model_dir).all_model_checkpoint_paths[-1]

        graph = self.logits.graph
        sess_config = self.tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        saver = self.tf.train.Saver()
        self.sess = self.tf.Session(config=sess_config, graph=graph)

        self.sess.run(self.tf.global_variables_initializer())
        self.tf.reset_default_graph()
        saver.restore(self.sess, ckpt)

        self.bs = BeamSearch(self.model,
                             self.hp.beam_size,
                             list(self.idx2token.keys())[2],
                             list(self.idx2token.keys())[3],
                             self.idx2token,
                             self.hp.maxlen2,
                             self.input_x,
                             self.input_y,
                             self.logits)

    def predict(self, content):
        """
        abstract prediction by beam search
        :param content: article content
        :return: prediction result
        """
        input_x = list(content)
        while len(input_x) < self.args.maxlen1: input_x.append('<pad>')
        input_x = input_x[:self.args.maxlen1]

        input_x = [self.token2idx.get(s, self.token2idx['<unk>']) for s in input_x]

        memory = self.sess.run(self.memory, feed_dict={self.input_x: [input_x]})

        return self.bs.search(self.sess, input_x, memory[0])

    def _add_placeholder(self):
        """
        add tensorflow placeholder
        """
        self.input_x = self.tf.placeholder(dtype=self.tf.int32, shape=[None, self.args.maxlen1], name='input_x')
        self.input_y = self.tf.placeholder(dtype=self.tf.int32, shape=[None, None], name='input_y')

if __name__ == '__main__':
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    preds = Prediction(hp)
    content = '2014年，51信用卡管家跟宜信等P2P公司合作，推出线上信贷产品“瞬时贷”，其是一种纯在线操作的信贷模式。51信用卡管家创始人孙海涛说，51目前每天放贷1000万，预计2015年，自营产品加上瞬>时贷，放贷额度将远超'
    result = preds.predict(content)
    for res in result:
        print(res)