# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
date: 2019/5/21
mail: cally.maxiong@gmail.com
page: http://www.cnblogs.com/callyblog/
'''
import logging

import tensorflow as tf
from tqdm import tqdm

from data_load import _load_vocab
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, noam_scheme
from utils import convert_idx_to_token_tensor, split_input

logging.basicConfig(level=logging.INFO)

class Transformer:
    def __init__(self, hp):
        self.hp = hp
        self.token2idx, self.idx2token = _load_vocab(hp.vocab)
        self.embeddings = get_token_embeddings(self.hp.vocab_size, self.hp.d_model, zero_pad=True)

    def encode(self, xs, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            self.x, sents1 = xs

            # embedding
            enc = tf.nn.embedding_lookup(self.embeddings, self.x) # (N, T1, d_model)
            enc *= self.hp.d_model**0.5 # scale

            enc += positional_encoding(enc, self.hp.maxlen1)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)
            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc, _ = multihead_attention(queries=enc,
                                                  keys=enc,
                                                  values=enc,
                                                  num_heads=self.hp.num_heads,
                                                  dropout_rate=self.hp.dropout_rate,
                                                  training=training,
                                                  causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
        self.enc_output = enc
        return self.enc_output, sents1

    def decode(self, xs, ys, memory, training=True):
        '''
        memory: encoder outputs. (N, T1, d_model)

        Returns
        logits: (N, T2, V). float32.
        y: (N, T2). int32
        sents2: (N,). string.
        '''
        self.memory = memory
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            self.decoder_inputs, y, sents2 = ys
            x, _, = xs

            # embedding
            dec = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)  # (N, T2, d_model)
            dec *= self.hp.d_model ** 0.5  # scale

            dec += positional_encoding(dec, self.hp.maxlen2)

            before_dec = dec

            dec = tf.layers.dropout(dec, self.hp.dropout_rate, training=training)

            attn_dists = []
            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Masked self-attention (Note that causality is True at this time)
                    dec, _ = multihead_attention(queries=dec,
                                                 keys=dec,
                                                 values=dec,
                                                 num_heads=self.hp.num_heads,
                                                 dropout_rate=self.hp.dropout_rate,
                                                 training=training,
                                                 causality=True,
                                                 scope="self_attention")
                    # Vanilla attention
                    dec, attn_dist = multihead_attention(queries=dec,
                                                          keys=self.memory,
                                                          values=self.memory,
                                                          num_heads=self.hp.num_heads,
                                                          dropout_rate=self.hp.dropout_rate,
                                                          training=training,
                                                          causality=False,
                                                          scope="vanilla_attention")
                    attn_dists.append(attn_dist)
                    ### Feed Forward
                    dec = ff(dec, num_units=[self.hp.d_ff, self.hp.d_model])

        # Final linear projection (embedding weights are shared)
        weights = tf.transpose(self.embeddings) # (d_model, vocab_size)
        logits = tf.einsum('ntd,dk->ntk', dec, weights) # (N, T2, vocab_size)

        with tf.variable_scope("gen", reuse=tf.AUTO_REUSE):
            gens = tf.layers.dense(tf.concat([before_dec, dec, attn_dists[-1]], axis=-1), units=1, activation=tf.sigmoid,
                                   trainable=training, use_bias=False)

        logits = tf.nn.softmax(logits)

        # final distribution
        self.logits = self._calc_final_dist(x, gens, logits, attn_dists[-1])

        return self.logits, y, sents2

    def _calc_final_dist(self, x, gens, vocab_dists, attn_dists):
        """Calculate the final distribution, for the pointer-generator model

        Args:
          x: encoder input which contain oov number
          gens: the generation, choose vocab from article or vocab
          vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays.
                       The words are in the order they appear in the vocabulary file.
          attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays

        Returns:
          final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
        """
        with tf.variable_scope('final_distribution', reuse=tf.AUTO_REUSE):
            # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
            vocab_dists = gens * vocab_dists
            attn_dists = (1-gens) * attn_dists

            batch_size = tf.shape(attn_dists)[0]
            dec_t = tf.shape(attn_dists)[1]
            attn_len = tf.shape(attn_dists)[2]

            dec = tf.range(0, limit=dec_t) # [dec]
            dec = tf.expand_dims(dec, axis=-1) # [dec, 1]
            dec = tf.tile(dec, [1, attn_len]) # [dec, atten_len]
            dec = tf.expand_dims(dec, axis=0) # [1, dec, atten_len]
            dec = tf.tile(dec, [batch_size, 1, 1]) # [batch_size, dec, atten_len]

            x = tf.expand_dims(x, axis=1) # [batch_size, 1, atten_len]
            x = tf.tile(x, [1, dec_t, 1]) # [batch_size, dec, atten_len]
            x = tf.stack([dec, x], axis=3)

            attn_dists_projected = tf.map_fn(fn=lambda y: tf.scatter_nd(y[0], y[1], [dec_t, self.hp.vocab_size]),
                                             elems=(x, attn_dists), dtype=tf.float32)

            final_dists = attn_dists_projected + vocab_dists

        return final_dists

    def _calc_loss(self, targets, final_dists):
        """
        calculate loss
        :param targets: reference
        :param final_dists:  transformer decoder output add by pointer generator
        :return: loss
        """
        with tf.name_scope('loss'):
            dec = tf.shape(targets)[1]
            batch_nums = tf.shape(targets)[0]
            dec = tf.range(0, limit=dec)
            dec = tf.expand_dims(dec, axis=0)
            dec = tf.tile(dec, [batch_nums, 1])
            indices = tf.stack([dec, targets], axis=2) # [batch_size, dec, 2]

            loss = tf.map_fn(fn=lambda x: tf.gather_nd(x[1], x[0]), elems=(indices, final_dists), dtype=tf.float32)
            loss = tf.log(0.9) - tf.log(loss)

            nonpadding = tf.to_float(tf.not_equal(targets, self.token2idx["<pad>"]))  # 0: <pad>
            loss = tf.reduce_sum(loss * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

            return loss

    def train(self, xs, ys):
        """
        train model
        :param xs: dataset xs
        :param ys: dataset ys
        :return: loss
                 train op
                 global step
                 tensorflow summary
        """
        tower_grads = []
        global_step = tf.train.get_or_create_global_step()
        global_step_ = global_step * self.hp.gpu_nums
        lr = noam_scheme(self.hp.d_model, global_step_, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        losses = []
        xs, ys = split_input(xs, ys, self.hp.gpu_nums)
        with tf.variable_scope(tf.get_variable_scope()):
            for no in range(self.hp.gpu_nums):
                with tf.device("/gpu:%d" % no):
                    with tf.name_scope("tower_%d" % no):
                        memory, sents1 = self.encode(xs[no])
                        logits, y, sents2 = self.decode(xs[no], ys[no], memory)
                        tf.get_variable_scope().reuse_variables()

                        loss = self._calc_loss(y, logits)
                        losses.append(loss)
                        grads = optimizer.compute_gradients(loss)
                        tower_grads.append(grads)

        with tf.device("/cpu:0"):
            grads = self.average_gradients(tower_grads)
            train_op = optimizer.apply_gradients(grads, global_step=global_step)
            loss = sum(losses) / len(losses)
            tf.summary.scalar('lr', lr)
            tf.summary.scalar("train_loss", loss)
            summaries = tf.summary.merge_all()

        return loss, train_op, global_step_, summaries

    def average_gradients(self, tower_grads):
        """
        average gradients of all gpu gradients
        :param tower_grads: list, each element is a gradient of gpu
        :return: be averaged gradient
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expend_g = tf.expand_dims(g, 0)
                grads.append(expend_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

    def eval(self, xs, ys):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        tensorflow summary
        '''
        # decoder_inputs <s> sentences
        decoder_inputs, y, sents2 = ys

        # decoder_inputs shape: [batch_size, 1] [[<s>], [<s>], [<s>], [<s>]]
        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<s>"]
        ys = (decoder_inputs, y, sents2)

        memory, sents1 = self.encode(xs, False)

        y_hat = None
        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(self.hp.maxlen2)):
            logits, y, sents2 = self.decode(xs, ys, memory, False)
            y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

            if tf.reduce_sum(y_hat, 1) == self.token2idx["<pad>"]: break

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, sents2)

        # monitor a random sample
        n = tf.random_uniform((), 0, tf.shape(y_hat)[0]-1, tf.int32)
        sent1 = sents1[n]
        pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)
        sent2 = sents2[n]

        tf.summary.text("sent1", sent1)
        tf.summary.text("pred", pred)
        tf.summary.text("sent2", sent2)
        summaries = tf.summary.merge_all()

        return y_hat, summaries