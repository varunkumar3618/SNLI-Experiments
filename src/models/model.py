import tensorflow as tf

def get_activation(activation):
    if activation == "tanh":
        return tf.tanh
    elif activation == "relu":
        return tf.nn.relu
    else:
        raise ValueError("Unsupported activation: %s." % activation)

def get_initializer(init):
    if init == "xavier":
        return tf.contrib.layers.xavier_initializer()
    elif init == "orth":
        return tf.orthogonal_initializer()
    else:
        raise ValueError("Unsupported initializer: %s." % init)

class SNLIModel(object):
    """Abstracts a Tensorflow graph for a learning task.

    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs.
    """
    def __init__(self, learning_rate, max_seq_len,
                 activation, dense_init, rec_init,
                 use_lens=False,
                 dropout_rate=-1, use_dropout=False,
                 clip_gradients=False, max_grad_norm=-1):
        self._max_seq_len = max_seq_len
        self._use_lens = use_lens
        self._use_dropout = use_dropout
        self._dropout_rate = dropout_rate
        self._clip_gradients = clip_gradients
        self._max_grad_norm = max_grad_norm

        self.activation = get_activation(activation)
        self.dense_init = get_initializer(dense_init)
        self.rec_init = get_initializer(rec_init)

    def apply_dropout(self, tensor):
        """Applies dropout to a tensor"""
        if not self._use_dropout:
            raise ValueError("The model has not been configured to use dropout.")
        return tf.layers.dropout(tensor, rate=self._dropout_rate, training=self.training_placeholder)

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
            self.training_placeholder = tf.placeholder(tf.bool)

    def create_feed_dict(self,
                         sentence1_batch, sentence1_lens_batch,
                         sentence2_batch, sentence2_lens_batch,
                         labels_batch=None, is_training=True):
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
            is_training: boolean
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
            feed_dict[self.training_placeholder] = is_training

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
        loss = (
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=self.labels_placeholder)
            )
            + tf.reduce_sum(sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
        )
        tf.summary.scalar("loss", loss)
        return loss

    def add_acc_op(self, pred):
        """Adds Ops for the accuracy to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            acc: A 0-d tensor (scalar) output
        """
        return tf.contrib.metrics.accuracy(pred, self.labels_placeholder)

    def add_summary_op(self):
        """Adds the Op to generate summary data.

        Returns:
            summary_op: A serialized Tensor<string> containing the Summary protobuf
        """
        return tf.summary.merge_all()

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
        optimizer = tf.train.AdamOptimizer()
        gradients = optimizer.compute_gradients(loss)
        if self._clip_gradients:
            gradient_values = tf.clip_by_global_norm([g[0] for g in gradients], self._max_grad_norm)[0]
            gradients = [(gv, var) for gv, (_, var) in zip(gradient_values, gradients)]
        train_op = optimizer.apply_gradients(gradients)
        return train_op

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
                                     labels_batch, is_training=True)
        _, loss, summary = sess.run([self.train_op, self.loss, self.summary_op],
                                    feed_dict=feed)
        return loss, summary

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
            predictions: np.ndarray of shape (n_samples,)
        """
        feed = self.create_feed_dict(sentence1_batch, sentence1_lens_batch,
                                     sentence2_batch, sentence2_lens_batch,
                                     labels_batch, is_training=False)
        return sess.run([self.acc_op, self.loss, self.pred], feed_dict=feed)

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
            predictions: np.ndarray of shape (n_samples,)
        """
        feed = self.create_feed_dict(sentence1_batch, sentence1_lens_batch,
                                     sentence2_batch, sentence2_lens_batch,
                                     is_training=False)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    def build(self):
        self.add_placeholders()
        self.pred, self.logits = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred, self.logits)
        self.train_op = self.add_training_op(self.loss)
        self.acc_op = self.add_acc_op(self.pred)
        self.summary_op = self.add_summary_op()
