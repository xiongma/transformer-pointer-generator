# -*- coding: utf-8 -*-
#!/usr/bin/python3
'''
date: 2019/5/21
mail: cally.maxiong@gmail.com
page: http://www.cnblogs.com/callyblog/
'''

import tensorflow as tf
# add self to decode memory
class Hypothesis:
    """
        Defines a hypothesis during beam search.
    """
    def __init__(self, tokens, log_prob, sents, normalize_by_length=True):
        """
        :param tokens: a list, which are ids in vocab
        :param log_prob: log probability, add by beam search
        :param sents: already decode words,
        :param normalize_by_length: sort hypothesis by prob / len, if not, just by prob
        """
        self.tokens = tokens
        self.log_prob = log_prob
        self.normalize_by_length = normalize_by_length
        self.sents = sents

    def extend(self, token, log_prob, word):
        """
        Extend the hypothesis with result from latest step.
        :param token: latest token from decoding
        :param log_prob: log prob of the latest decoded tokens.
        :param word: word piece by transformer decode
        :return: new Hypothesis with the results from latest step.
        """

        return Hypothesis(self.tokens + [token], self.log_prob + log_prob, self.sents + word)

    @property
    def latest_token(self):
        return self.tokens[-1]

    def __str__(self):
        return ''.join(list(self.sents))

class BeamSearch:
    def __init__(self, model, beam_size, start_token, end_token, id2token, max_steps, input_x, input_y, logits,
                 normalize_by_length=False):
        """
        :param model: transformer model
        :param beam_size: beam size
        :param start_token: start token
        :param end_token: end token
        :param id2token: id to token dict
        :param max_steps: max steps in decode
        :param input_x: input x
        :param input_y: input y
        :param logits: logits by decode
        :param normalize_by_length: sort hypothesis by prob / len, if not, just by prob
        """
        # basic params
        self.model = model
        self.beam_size = beam_size
        self.start_token = start_token
        self.end_token = end_token
        self.max_steps = max_steps
        self.id2token = id2token

        # placeholder
        self.input_x = input_x
        self.input_y = input_y

        self.top_k_ = tf.nn.top_k(tf.nn.softmax(logits), k=self.beam_size * 2)

        # This length normalization is only effective for the final results.
        self.normalize_by_length = normalize_by_length

    def search(self, sess, input_x, memory):
        """
        use beam search for decoding
        :param sess: tensorflow session
        :param input_x: article by list, and convert to id by vocab
        :param memory: transformer encode result
        :return: hyps: list of Hypothesis, the best hypotheses found by beam search,
                       ordered by score
        """
        # create a list, which each element is Hypothesis
        hyps = [Hypothesis([self.start_token], 0.0, '')] * self.beam_size

        results = []
        steps = 0
        while steps < self.max_steps and len(results) < self.beam_size:
            top_k = sess.run([self.top_k_], feed_dict={self.model.memory: [memory] * self.beam_size,
                                                       self.input_x: [input_x] * self.beam_size,
                                                       self.input_y: [h.tokens for h in hyps]})
            # print(time.time() - start)
            indices = [list(indice[-1]) for indice in top_k[0][1]]
            probs = [list(prob[-1]) for prob in top_k[0][0]]

            all_hyps = []

            num_orig_hyps = 1 if steps == 0 else len(hyps)
            for i in range(num_orig_hyps):
                h = hyps[i]
                for j in range(self.beam_size*2):
                    new_h = h.extend(indices[i][j], probs[i][j], self.id2token[indices[i][j]])
                    all_hyps.append(new_h)

            # Filter and collect any hypotheses that have the end token
            hyps = []
            for h in self.best_hyps(all_hyps):
                if h.latest_token == self.end_token:
                    # Pull the hypothesis off the beam if the end token is reached.
                    results.append(h)
                else:
                    # Otherwise continue to the extend the hypothesis.
                    hyps.append(h)
                if len(hyps) == self.beam_size or len(results) == self.beam_size:
                    break

            steps += 1

            if steps == self.max_steps:
                results.extend(hyps)

        return self.best_hyps(results)

    def best_hyps(self, hyps):
        """
        Sort the hyps based on log probs and length.
        :param hyps: A list of hypothesis
        :return: A list of sorted hypothesis in reverse log_prob order.
        """
        # This length normalization is only effective for the final results.
        if self.normalize_by_length:
            return sorted(hyps, key=lambda h: h.log_prob / len(h.tokens), reverse=True)
        else:
            return sorted(hyps, key=lambda h: h.log_prob, reverse=True)