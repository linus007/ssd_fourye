from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

import sys
import time
import datetime
import os
import os.path as osp

from solver.solver import Solver
from utils.timer import Timer
from config import cfg


class VGG16_Solver(Solver):
    def __init__(self, data_batch, net, imdb):
        self._momentum = 0.9
        self._batch_size = cfg.BATCH_SIZE

        self._data_path = osp.join(cfg.DATA_ROOT, cfg.DATA_SET_USED)
        self._data_output_dir = osp.join(self._data_path, "output")
        if not osp.exists(self._data_output_dir):
            os.makedirs(self._data_output_dir)
        self._train_dir = osp.join(self._data_path, cfg.TRAIN_DIR)
        if not osp.exists(self._train_dir):
            os.makedirs(self._train_dir)
        self._ckpt_file = osp.join(self._data_output_dir, "save.ckpt")

        self._max_iters = cfg.MAX_ITERS
        self._save_iter = cfg.SAVE_ITER
        self._summary_iter = cfg.SUMMARY_ITER

        self._learning_rate = cfg.LEARNING_RATE
        self._decay_steps = cfg.DECAY_STEPS
        self._decay_rate = cfg.DECAY_RATE
        self._stair_case = cfg.STAIR_CASE

        self._data_batch = data_batch

        self._net = net

        self._num_classes = imdb.num_classes

        self._is_pretrain = cfg.IS_PRETRAIN

        self._nums_boxes_used = cfg.NUM_DEFAULT_BOXES

        self.construct_graph()


    def construct_graph(self):
        self.variable_to_restore = self._net.pretrained_collection
        self.variavle_trainable = self._net.trainable_collection

        self.saver = tf.train.Saver(tf.global_variables()
            , max_to_keep=None)

        self._summary_op = tf.summary.merge_all()
        self._writer = tf.summary.FileWriter(self._data_output_dir, flush_secs=60)

        self._global_step = tf.get_variable("global_step"
            , [], initializer=tf.constant_initializer(0)
            , trainable=True)

        self._learning_rate_ed = tf.train.exponential_decay(
            self._learning_rate     \
            , self._global_step, self._decay_steps, self._decay_rate
            , self._stair_case, name="learning_rate")

        self._optimizer = tf.train.MomentumOptimizer(
            self._learning_rate_ed, self._momentum)  \
                .minimize(self._net.total_loss)

        self._ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        self._averages_op = self._ema.apply(tf.trainable_variables())

        with tf.control_dependencies([self._optimizer]):
            self._train_op = tf.group(self._averages_op)

        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)

        self.sess.run(tf.global_variables_initializer())

        if self._is_pretrain:
            self._restorer = tf.train.Saver(self.variavle_trainable
                , max_to_keep=None)
            self._pretrain_model_path = osp.join(self._data_path, cfg.PRETRAIN_MODEL_PATH)

            self._restorer.restore(self.sess, self._pretrain_model_path)

        self._writer.add_graph(self.sess.graph)


    def train(self):
        train_timer = Timer()
        load_timer = Timer()

        for step in range(1, self._max_iters + 1):
            load_timer.tic()
            imgs, labs = self._data_batch.next_batch()
            load_timer.toc()

            feed_dict = {}
            feed_dict[self._net.images] = imgs
            for pl, lab in zip(self._net.lables, labs):
                feed_dict[pl] = lab
            if step % self._summary_iter == 0:
                if step % (self._summary_iter * 10) == 0:
                    train_timer.tic()
                    summary_str, loss, _ = self.sess.run(
                        [self._summary_op, self._net.total_loss, self._train_op]
                        , feed_dict=feed_dict)
                    train_timer.toc()

                    log_str = ('{} Epoch: {}, Step: {}, Learning rate: {}'
                        ' Loss: {:5.3f}\n Speed: {:.3f}s/iter, '
                        'Load: {:.3f}s/iter, Remain: {}').format(
                        datetime.datetime.now().strftime("%m/%d %H:%M:%s")
                        , self._data_batch.epoch
                        , int(step)
                        , round(self._learning_rate_ed.eval(session=self.sess), 6)
                        , loss, train_timer.average_time
                        , load_timer.average_time
                        , train_timer.remain(step, self._max_iters))
                    print(log_str)

                else:
                    train_timer.tic()
                    summary_str, _ = self.sess.run(
                        [self._summary_op, self._train_op]
                        , feed_dict=feed_dict)
                    train_timer.toc()

                self._writer.add_summary(summary_str, step)
            else:
                train_timer.tic()
                self.sess.run(self._train_op, feed_dict=feed_dict)
                train_timer.toc()

            if step % self._save_iter == 0:
                print("{} Saving checkpoint file to: {}"
                    . format(datetime.datetime.now().strftime("%m/%d %H:%M:%S")
                        , self._data_output_dir))

                self.saver.save(self.sess, self._ckpt_file
                    , global_step=self._global_step)
