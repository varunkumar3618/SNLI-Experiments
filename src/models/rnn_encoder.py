import tensorflow as tf

from src.models.model import SNLIModel
from src.utils.ops import get_embedding


class RNNEncoder(SNLIModel):
    def __init__(self, embedding_matrix, update_embeddings,
                 hidden_size, use_peepholes, clip_gradients, max_grad_norm,
                 l2_reg,
                 *args, **kwargs):
        super(RNNEncoder, self).__init__(use_lens=True, use_dropout=True, *args, **kwargs)
        self._embedding_matrix = embedding_matrix
        self._update_embeddings = update_embeddings
        self._l2_reg = l2_reg
        self._hidden_size = hidden_size
        self._use_peepholes = use_peepholes
        self._clip_gradients = clip_gradients
        self._max_grad_norm = max_grad_norm

    def add_prediction_op(self):
        with tf.variable_scope("prediction"):
            reg = tf.contrib.layers.l2_regularizer(self._l2_reg)

            prem_embed = get_embedding(self.sentence1_placeholder, self._embedding_matrix,
                                       self._update_embeddings)
            hyp_embed = get_embedding(self.sentence2_placeholder, self._embedding_matrix,
                                      self._update_embeddings, reuse=True)

            prem_embed = tf.layers.dropout(prem_embed, self.dropout_placeholder)
            hyp_embed = tf.layers.dropout(hyp_embed, self.dropout_placeholder)

            prem_proj = tf.layers.dense(prem_embed, self._hidden_size / 2,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        activation=tf.tanh, name="prem_proj")
            hyp_proj = tf.layers.dense(hyp_embed, self._hidden_size / 2,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       activation=tf.tanh, name="hyp_proj")

            cell = tf.contrib.rnn.LSTMCell(self._hidden_size / 2, use_peepholes=self._use_peepholes)
            with tf.variable_scope("prem_encoder"):
                prem_states, _ = tf.nn.dynamic_rnn(cell, prem_proj, dtype=tf.float32,
                                                   sequence_length=self.sentence1_lens_placeholder)
                prem_encoded = tf.unstack(prem_states, axis=1)[-1]
            with tf.variable_scope("hyp_encoder"):
                hyp_states, _ = tf.nn.dynamic_rnn(cell, hyp_proj, dtype=tf.float32,
                                                  sequence_length=self.sentence2_lens_placeholder)
                hyp_encoded = tf.unstack(hyp_states, axis=1)[-1]

            both_encoded = tf.concat([prem_encoded, hyp_encoded], axis=1)
            both_encoded = tf.layers.dropout(both_encoded, self.dropout_placeholder)

            h1 = tf.layers.dense(both_encoded, self._hidden_size,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 kernel_regularizer=reg,
                                 activation=tf.tanh, name="h1")
            h2 = tf.layers.dense(h1, self._hidden_size,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 kernel_regularizer=reg,
                                 activation=tf.tanh, name="h2")
            h3 = tf.layers.dense(h2, self._hidden_size,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 kernel_regularizer=reg,
                                 activation=tf.tanh, name="h3")
            logits = tf.layers.dense(h3, self._hidden_size,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=reg,
                                     name="logits")
            preds = tf.argmax(logits, axis=1)
            return preds, logits

    def add_loss_op(self, pred, logits):
        loss = super(RNNEncoder, self).add_loss_op(pred, logits)\
            + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer()
        gradients = optimizer.compute_gradients(loss)
        if self._clip_gradients:
            gradient_values = tf.clip_by_global_norm([g[0] for g in gradients], self._max_grad_norm)[0]
            gradients = [(gv, var) for gv, (_, var) in zip(gradient_values, gradients)]
        train_op = optimizer.apply_gradients(gradients)
        return train_op
