from abc import ABC
import tensorflow as tf
from base.base_model import BaseModel

from sklearn.datasets import fetch_openml


class CNN(BaseModel, ABC):
    def __init__(self, config):
        super(CNN, self).__init__(config)
        self.config = config
        self.is_training = None
        self.cross_entropy = None
        self.x = None
        self.x_reshaped = None
        self.y = None
        self.cross_entropy = None
        self.loss = None
        self.train_step = None
        self.accuracy = None
        self.correct = None
        self.saver = None
        self.build()
        self.init_saver()

    def build(self):
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.x = tf.placeholder(tf.float32, shape=[None, 784], name="input_x")
        self.x_reshaped = tf.reshape(self.x, shape=[-1, 28, 28, 1], name="x_reshaped")
        self.y = tf.placeholder(tf.int32, shape=[None], name="y")

        # network architecture
        with tf.name_scope("input"):
            d1 = tf.layers.conv2d(self.x_reshaped, filters=20, kernel_size=1, strides=1, padding="SAME", name="dense2")
            d2 = tf.layers.conv2d(d1, filters=10, kernel_size=3, strides=2, padding="VALID", activation=tf.nn.relu,\
                                  name="dense3")

        with tf.name_scope("pool"):
            pool1 = tf.nn.max_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool1")
            pool1_flat = tf.reshape(pool1, shape=[-1, 6 * 6 * 10])
            pool1_flat_drop = tf.layers.dropout(pool1_flat, rate=0.5, name="dropout")

        with tf.name_scope("fc1"):
            fc1 = tf.layers.dense(pool1_flat_drop, 100, activation=tf.nn.relu, name="fc1")
            fc1_drop = tf.layers.dropout(fc1, 0.6, training=True)

        with tf.name_scope("output"):
            logits = tf.layers.dense(fc1_drop, 10, name="output")
            Y_proba = tf.nn.softmax(logits, name="Y_proba")

        with tf.name_scope("loss"):
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.y)
            self.loss = tf.reduce_mean(self.cross_entropy)
            self.train_step = tf.train.AdamOptimizer(
                self.config.learning_rate)\
                .minimize(self.loss)

        with tf.name_scope("Evaluation"):
            self.correct = tf.nn.in_top_k(logits, self.y, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)


class CNN_LSTM(BaseModel, ABC):
    def __init__(self, config):
        super(BaseModel, self).__init__(config)
        self.config = config
