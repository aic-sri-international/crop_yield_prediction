import numpy as np
import tensorflow as tf


class CNNModel:
    def __init__(self, config):

        self.x = tf.placeholder(tf.float32, [None, config.W, config.H, config.C], name="x")
        self.y = tf.placeholder(tf.float32, [None])
        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])

        self.conv1_1 = self.conv_relu_batch(self.x, 128, 3, 1, name="conv1_1")
        conv1_1_d = tf.nn.dropout(self.conv1_1, self.keep_prob)
        conv1_2 = self.conv_relu_batch(conv1_1_d, 256, 3, 2, name="conv1_2")
        conv1_2_d = tf.nn.dropout(conv1_2, self.keep_prob)

        conv2_1 = self.conv_relu_batch(conv1_2_d, 256, 3, 1, name="conv2_1")
        conv2_1_d = tf.nn.dropout(conv2_1, self.keep_prob)
        conv2_2 = self.conv_relu_batch(conv2_1_d, 512, 3, 2, name="conv2_2")
        conv2_2_d = tf.nn.dropout(conv2_2, self.keep_prob)

        conv3_1 = self.conv_relu_batch(conv2_2_d, 512, 3, 1, name="conv3_1")
        conv3_1_d = tf.nn.dropout(conv3_1, self.keep_prob)
        conv3_2 = self.conv_relu_batch(conv3_1_d, 1024, 3, 2, name="conv3_2")
        conv3_2_d = tf.nn.dropout(conv3_2, self.keep_prob)

        dim = np.prod(conv3_2_d.get_shape().as_list()[1:])
        flattened = tf.reshape(conv3_2_d, [-1, dim])
        # flattened_d = tf.nn.dropout(flattened, 0.25)

        self.fc6 = self.dense(flattened, 1024, name="fc6")
        # self.fc6 = tf.concat(1, [self.fc6_img,self.year])

        self.logits = tf.squeeze(self.dense(self.fc6, 1, name="dense"))
        self.loss_err = tf.nn.l2_loss(self.logits - self.y)

        with tf.variable_scope('dense') as scope:
            scope.reuse_variables()
            self.dense_W = tf.get_variable('W')
            self.dense_B = tf.get_variable('b')
        with tf.variable_scope('conv1_1/conv2d') as scope:
            scope.reuse_variables()
            self.conv_W = tf.get_variable('W')
            self.conv_B = tf.get_variable('b')

        # L1 term
        self.loss_reg = tf.abs(tf.reduce_sum(self.logits - self.y))
        # soybean
        # alpha = 1.5
        # corn
        alpha = 5
        self.loss = self.loss_err + self.loss_reg * alpha

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def conv2d(self, input_data, out_channels, filter_size, stride, in_channels=None, name="conv2d"):
        if not in_channels:
            in_channels = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            W = tf.get_variable("W", [filter_size, filter_size, in_channels, out_channels],
                                initializer=tf.contrib.layers.variance_scaling_initializer())
            b = tf.get_variable("b", [1, 1, 1, out_channels])
            return tf.nn.conv2d(input_data, W, [1, stride, stride, 1], "SAME") + b

    def conv_relu_batch(self, input_data, out_channels, filter_size, stride, in_channels=None, name="crb"):
        with tf.variable_scope(name):
            a = self.conv2d(input_data, out_channels, filter_size, stride, in_channels)
            b = self.batch_normalization(a, axes=[0, 1, 2])
            r = tf.nn.relu(b)
            return r

    def dense(self, input_data, H, N=None, name="dense"):
        if not N:
            N = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            W = tf.get_variable("W", [N, H], initializer=tf.contrib.layers.variance_scaling_initializer())
            b = tf.get_variable("b", [1, H])
            return tf.matmul(input_data, W, name="matmul") + b

    def batch_normalization(self, input_data, axes=[0], name="batch"):
        with tf.variable_scope(name):
            mean, variance = tf.nn.moments(input_data, axes, keep_dims=True, name="moments")
            return tf.nn.batch_normalization(input_data, mean, variance, None, None, 1e-6, name="batch")

    def train(self):
        pass

    def test(self):
        pass
