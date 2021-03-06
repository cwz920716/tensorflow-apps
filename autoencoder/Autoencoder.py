import tensorflow as tf
import numpy as np
import autoencoder.Utils

class Autoencoder(object):

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer = tf.train.AdamOptimizer()):
        # Usage: test_*.py [dev_type_string] [dev_id_string]
        # dev_string= /gpu:2
        # generate from basic block
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        all_weights = dict()
        with tf.device('/gpu:2'):
            all_weights['w1'] = tf.Variable(autoencoder.Utils.xavier_init(self.n_input, self.n_hidden))
            all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
            all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
            all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        
        self.weights = all_weights
        with tf.device('/gpu:2'):
            self.x = tf.placeholder(tf.float32, [None, self.n_input])
            self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
            self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
            self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.sub(self.reconstruction, self.x), 2.0))
            self.optimizer = optimizer.minimize(self.cost)
            init = tf.initialize_all_variables()
        
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.sess.run(init)
        

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X})

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])


