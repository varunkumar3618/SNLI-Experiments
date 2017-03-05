import tensorflow as tf

class SNLIModel(object):
    """Abstracts a Tensorflow graph for a learning task.

    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs.
    """
    def __init__(self, max_seq_len, use_lens, use_dropout, learning_rate, dropout_rate=-1):
        self._max_seq_len = max_seq_len
        self._use_lens = use_lens
        self._use_dropout = use_dropout
        self._dropout_rate = dropout_rate

    def add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph.

        Tensorflow uses placeholder variables to represent locations in a
        computational graph where data is inserted.  These placeholders are used as
        inputs by the rest of the model building and will be fed data during
        training.

        See for more information:
        https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
        """
        self.labels_placeholder = tf.placeholder(tf.int64, shape=[None], name="labels")
        self.sentence1_placeholder = tf.placeholder(tf.int64, shape=[None, self._max_seq_len], name="sentence1")
        self.sentence2_placeholder = tf.placeholder(tf.int64, shape=[None, self._max_seq_len], name="sentence2")

        if self._use_lens:
            self.sentence1_lens_placeholder = tf.placeholder(tf.int64, shape=[None], name="sentence1_lengths")
            self.sentence2_lens_placeholder = tf.placeholder(tf.int64, shape=[None], name="sentence2_lengths")

        if self._use_dropout:
            self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self,
                         sentence1_batch, sentence1_lens_batch,
                         sentence2_batch, sentence2_lens_batch,
                         labels_batch=None, train_mode=True):
        """Creates the feed_dict for one step of training.

        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        If labels_batch is None, then no labels are added to feed_dict.

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Args:
            sentence1_batch: np.ndarray of shape (n_samples, max_len)
            sentence1_lens_batch: np.ndarray of shape (n_samples)
            sentence2_batch: np.ndarray of shape (n_samples, max_len)
            sentence2_lens_batch: np.ndarray of shape (n_samples)
            labels_batch: np.ndarray of shape (n_samples)
            train_mode: boolean
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {}
        feed_dict[self.sentence1_placeholder] = sentence1_batch
        feed_dict[self.sentence2_placeholder] = sentence2_batch

        if self._use_lens:
            feed_dict[self.sentence1_lens_placeholder] = sentence1_lens_batch
            feed_dict[self.sentence2_lens_placeholder] = sentence2_lens_batch

        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch

        if self._use_dropout:
            if train_mode:
                feed_dict[self.dropout_placeholder] = self._dropout_rate
            else:
                feed_dict[self.dropout_placeholder] = 1.

        return feed_dict

    def add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input data into predictions.

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
            logits: A tensor of shape (batch_size, n_classes)
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, pred, logits):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes)
            logits: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=self.labels_placeholder))

    def add_acc_op(self, pred):
        """Adds Ops for the accuracy to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            acc: A 0-d tensor (scalar) output
        """
        return tf.contrib.metrics.accuracy(pred, self.labels_placeholder)

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        sess.run() to train the model. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Args:
            loss: Loss tensor (a scalar).
        Returns:
            train_op: The Op for training.
        """
        return tf.train.AdamOptimizer().minimize(loss)

    def train_on_batch(self, sess,
                       sentence1_batch, sentence1_lens_batch,
                       sentence2_batch, sentence2_lens_batch,
                       labels_batch):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: tf.Session()
            sentence1_batch: np.ndarray of shape (n_samples, max_len)
            sentence1_lens_batch: np.ndarray of shape (n_samples)
            sentence2_batch: np.ndarray of shape (n_samples, max_len)
            sentence2_lens_batch: np.ndarray of shape (n_samples)
            labels_batch: np.ndarray of shape (n_samples)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(sentence1_batch, sentence1_lens_batch,
                                     sentence2_batch, sentence2_lens_batch,
                                     labels_batch, train_mode=True)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def evaluate_on_batch(self, sess,
                          sentence1_batch, sentence1_lens_batch,
                          sentence2_batch, sentence2_lens_batch,
                          labels_batch):
        """Obtain the loss and accuracy on a batch of data.

        Args:
            sess: tf.Session()
            sentence1_batch: np.ndarray of shape (n_samples, max_len)
            sentence1_lens_batch: np.ndarray of shape (n_samples)
            sentence2_batch: np.ndarray of shape (n_samples, max_len)
            sentence2_lens_batch: np.ndarray of shape (n_samples)
            labels_batch: np.ndarray of shape (n_samples)
        Returns:
            acc: accuracy over the batch (a scalar)
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(sentence1_batch, sentence1_lens_batch,
                                     sentence2_batch, sentence2_lens_batch,
                                     labels_batch, train_mode=False)
        return sess.run([self.acc_op, self.loss], feed_dict=feed)

    def predict_on_batch(self, sess,
                         sentence1_batch, sentence1_lens_batch,
                         sentence2_batch, sentence2_lens_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            sentence1_batch: np.ndarray of shape (n_samples, max_len)
            sentence1_lens_batch: np.ndarray of shape (n_samples)
            sentence2_batch: np.ndarray of shape (n_samples, max_len)
            sentence2_lens_batch: np.ndarray of shape (n_samples)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(sentence1_batch, sentence1_lens_batch,
                                     sentence2_batch, sentence2_lens_batch,
                                     train_mode=False)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    def build(self):
        self.add_placeholders()
        self.pred, self.logits = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred, self.logits)
        self.train_op = self.add_training_op(self.loss)
        self.acc_op = self.add_acc_op(self.pred)
