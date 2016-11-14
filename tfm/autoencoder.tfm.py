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

