# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
date: 2019/5/21
mail: cally.maxiong@gmail.com
page: http://www.cnblogs.com/callyblog/
'''

import logging
import os

from tqdm import tqdm

from data_load import get_batch
from hparams import Hparams
from model import Transformer
from utils import save_hparams, save_variable_specs, get_hypotheses, calc_rouge, import_tf

logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()

# import tensorflow
gpu_list = [str(i) for i in list(range(hp.gpu_nums))]
tf = import_tf(gpu_list)

save_hparams(hp, hp.logdir)

logging.info("# Prepare train/eval batches")
train_batches, num_train_batches, num_train_samples, _ = get_batch(hp.train,
                                                                    hp.maxlen1,
                                                                    hp.maxlen2,
                                                                    hp.vocab,
                                                                    hp.batch_size,
                                                                   hp.gpu_nums,
                                                                    shuffle=True)

eval_batches, num_eval_batches, num_eval_samples, eval_sent2 = get_batch(hp.eval,
                                                                         hp.maxlen1,
                                                                         hp.maxlen2,
                                                                         hp.vocab,
                                                                         hp.batch_size,
                                                                         1,
                                                                         shuffle=False)

# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
xs, ys = iter.get_next()

logging.info('# init data')
train_init_op = iter.make_initializer(train_batches)
eval_init_op = iter.make_initializer(eval_batches)

logging.info("# Load model")
m = Transformer(hp)

loss, train_op, global_step, train_summaries = m.train(xs, ys)

y_hat, eval_summaries, sent2, pred, eval_loss = m.eval(xs, ys)

logging.info("# Session")
saver = tf.train.Saver(max_to_keep=hp.num_epochs)
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    if ckpt is None:
        logging.info("Initializing from scratch")
        sess.run(tf.global_variables_initializer())
        save_variable_specs(os.path.join(hp.logdir, "specs"))
    else:
        saver.restore(sess, ckpt)

    summary_writer = tf.summary.FileWriter(hp.logdir, sess.graph)

    sess.run(train_init_op)
    total_steps = hp.num_epochs * num_train_batches
    _gs = sess.run(global_step)
    for i in tqdm(range(_gs, total_steps+1, hp.gpu_nums)):
        _, _gs, _summary = sess.run([train_op, global_step, train_summaries])
        summary_writer.add_summary(_summary, _gs)
        if _gs % 3000 == 0:
            logging.info("steps {} is done".format(_gs))
            _loss = sess.run(loss) # train loss

            logging.info("# test evaluation")
            _, _eval_summaries, _sent2, _pred, eval_loss_ = sess.run([eval_init_op, eval_summaries, sent2, pred, eval_loss])
            logging.info('origin sentence is {0}, prediction sentence is {1}'.format(_sent2.decode('utf-8'),
                                                                                     _pred.decode('utf-8')))

            summary_writer.add_summary(_eval_summaries, _gs)

            logging.info("# get hypotheses")
            hypotheses = get_hypotheses(num_eval_batches, num_eval_samples, sess, y_hat, m.idx2token)

            logging.info("# write results")
            model_output = "trans_pointer%02dL%.2f" % (_gs, _loss)
            if not os.path.exists(hp.evaldir): os.makedirs(hp.evaldir)
            translation = os.path.join(hp.evaldir, model_output)
            with open(translation, 'w', encoding='utf-8') as fout:
                fout.write("\n".join(hypotheses))

            logging.info("# calc rouge score ")
            calc_rouge(eval_sent2, hypotheses, _gs, hp.evaldir)

            logging.info("# save models")
            ckpt_name = os.path.join(hp.logdir, model_output)
            saver.save(sess, ckpt_name, global_step=_gs)
            logging.info("after training of {} steps, {} has been saved.".format(_gs, ckpt_name))

            logging.info("# fall back to train mode")
            sess.run(train_init_op)
    summary_writer.close()

logging.info("Done")