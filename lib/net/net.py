from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from config import cfg
import tensorflow as tf

class Net(object):
    def __init__(self):
        self.pretrained_collection = []
        self.trainable_collection = []

    def make_variable(self, name, shape, initializer
        , trainable=True, pretrainable=True):
        var = tf.get_variable(name, shape
            , initializer=initializer
            , trainable=trainable)

        if trainable:
            self.trainable_collection.append(var)

        if pretrainable:
            self.pretrained_collection.append(var)
        return var

    def make_weights(self, name, shape, means=0.0
        , stddev=0.01, is_xavier=False, trainable=True
        , pretrainable=True):
        if is_xavier:
            initializer = tf.contrib.layers.xavier_initializer_conv2d()
        else:
            initializer = tf.truncated_normal_initializer(means, stddev=stddev)
        weight = self.make_variable(name, shape, initializer
            , trainable, pretrainable)

        return weight


    def make_biases(self, name, shape, init=0.0
        , is_xavier=False, trainable=True
        , pretrainable=True):
        if is_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.constant_initializer(init)
        bias = self.make_variable(name, shape, initializer
            , trainable, pretrainable)

        return bias

    def conv(self, input, name
        , k_h, k_w, c_o, s_h, s_w
        , padding="SAME", group=1
        , is_xavier=False
        , is_add_bias=True
        , is_bn = False
        , is_activiation=True
        , trainable=True
        , pretrainable=True):

        c_i = int(input.get_shape()[-1])

        assert c_i % group == 0
        assert c_o % group == 0

        convolve = lambda i, k: tf.nn.conv2d(i, k
            , [1, s_h, s_w, 1], padding=padding)

        with tf.variable_scope(name) as scope:
            kernel = self.make_weights("weights"
                , [k_h, k_w, c_i / group, c_o]
                , is_xavier=is_xavier
                , trainable=trainable
                , pretrainable=pretrainable)

            if group == 1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k)
                    for i, k in zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)

            if is_add_bias:
                bias = self.make_biases("biases", [c_o]
                    , trainable=trainable
                    , is_xavier=is_xavier
                    , pretrainable=pretrainable)
                conv = tf.nn.bias_add(conv, bias, name=scope.name)
            if is_bn:
                conv = self.batch_normalization(conv, pretrainable=False)
            if is_activiation:
                conv = self.relu(conv, name="relu")
            return conv

    def batch_normalization(self, input, name="BN"
        , is_training=True, decay=0.9, eps=1e-5
        , trainable=True, pretrainable=False):
        shape = input.get_shape().as_list()
        assert len(shape) in [2, 4]

        out_dim = int(shape[-1])
        with tf.variable_scope(name) as scope:

            if len(shape) == 4:     # flollowed after conv layer
                batch_mean, batch_variance = tf.nn.moments(input, axes=[0, 1, 2])
            else:   # followed after fc layer
                batch_mean, batch_variance = tf.nn.moments(input, [0])

            beta = self.make_variable('beta', [out_dim]
                , initializer=tf.constant_initializer(0.0)
                , trainable=trainable
                , pretrainable=pretrainable)
            gamma = self.make_variable('gamma', [out_dim]
                , initializer=tf.constant_initializer(1.0)
                , trainable=trainable
                , pretrainable=pretrainable)

            ema = tf.train.ExponentialMovingAverage(decay=decay)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_variance])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_variance)

            mean, var = tf.cond(tf.cast(is_training, tf.bool)
                , mean_var_with_update
                ,lambda: (ema.average(batch_mean)
                    , ema.average(batch_variance)))

            bn = tf.nn.batch_normalization(input, mean
                , var, beta, gamma, eps)
            return bn


    def fc(self, name, input, num_in, num_out
        , is_activiation=True, trainable=True
        , pretrainable=True):
        with tf.variable_scope(name) as scope:
            reshape = tf.reshape(input, [tf.shape(input)[0], -1])
            weight = self.make_weights("weights", [num_in, num_out]
                , trainable=trainable, pretrainable=pretrainable)
            bias = self.make_biases("biases", [num_out]
                , trainable=trainable
                , pretrainable=pretrainable)

            fc = tf.matmul(reshape, weight) + bias
            if is_activiation:
                fc = self.relu(input, "relu")
            return fc

    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    def avg_pool(self, input, name, k_h, k_w, s_h, s_w, padding="SAME"):
        return tf.nn.avg_pool(input, ksize=[1, k_h, k_w, 1]
            , strides=[1, s_h, s_w, 1], padding=padding
            , name=name)

    def max_pool(self, input, k_h, k_w, s_h, s_w
        , name, padding="SAME"):
        return tf.nn.max_pool(input, ksize=[1, k_h, k_w, 1]
            , strides=[1, s_h, s_w, 1], padding=padding
            , name=name)

    def lrn(self, input, radius, alpha
        , beta, name, bias=1.0):
        return tf.nn.local_response_normalization(
            input, depth_radius=radius, alpha=alpha
            , beta=beta, bias=bias, name=name)

    def smooth_l1(self, x):
        l2 = 0.5 * (x**2.0)
        l1 = tf.abs(x) - 0.5
        res = tf.where(tf.less(tf.abs(x), 1.0)
            , l2, l1)
        return res

    def inference(self, inputs):
        raise NotImplementedError

    def loss_layer(self, predicts, lables, objects_num):
        raise NotImplementedError
