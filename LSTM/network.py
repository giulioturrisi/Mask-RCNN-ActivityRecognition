import tensorflow as tf

def extract_state(data, ind):

    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)

    return res

class network:

    def __init__(self, max_steps, num_objects, num_classes, num_units, keep_prob):

        self.input = tf.placeholder(dtype=tf.float32, shape=[None, max_steps, num_objects])

        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])

        self.lenghts = tf.placeholder(dtype=tf.int32, shape=[None])

        #self.keep_prob = tf.placeholder(dtype=tf.float32)

        self.keep_prob = keep_prob

        cell = tf.contrib.rnn.LSTMCell(num_units=num_units)

        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = self.keep_prob)

        #self.input = tf.cast(self.input,tf.float32)

        hidden_state, _ = tf.nn.dynamic_rnn(cell, self.input, sequence_length=self.lenghts,
                                                    swap_memory=True, dtype=tf.float32)


        last_state = extract_state(hidden_state, self.lenghts-1)

        self.logits = tf.layers.dense(last_state,num_classes)

        self.softmax = tf.nn.softmax(self.logits)

        self.prediction = tf.argmax(self.softmax, axis=-1)

        label_index = tf.argmax(self.labels, axis=-1)

        self.correct_classified = tf.reduce_mean(tf.cast(tf.equal(self.prediction, label_index), dtype=tf.float32))

        loss_per_sample = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)

        self.loss = tf.reduce_mean(loss_per_sample)

        tf.summary.scalar("loss", self.loss)

        tf.summary.scalar("accuracy", self.correct_classified)

        opt = tf.train.AdamOptimizer()

        self.train_step = opt.minimize(self.loss)
