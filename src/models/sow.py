import tensorflow as tf

from src.models.model import SNLIModel
from src.utils.ops import get_embedding


class SumOfWords(SNLIModel):
    def __init__(self, embedding_matrix, update_embeddings,
                 hidden_size,
                 l2_reg,
                 *args, **kwargs):
        super(SumOfWords, self).__init__(use_lens=False, use_dropout=True, *args, **kwargs)
        self._embedding_matrix = embedding_matrix
        self._update_embeddings = update_embeddings
        self._l2_reg = l2_reg
        self._hidden_size = hidden_size

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

            prem_sow = tf.reduce_sum(prem_proj, axis=1)
            hyp_sow = tf.reduce_sum(hyp_proj, axis=1)
            both_sow = tf.concat([prem_sow, hyp_sow], axis=1)

            both_sow = tf.layers.dropout(both_sow, self.dropout_placeholder)

            h1 = tf.layers.dense(both_sow, self._hidden_size,
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
        loss = super(SumOfWords, self).add_loss_op(pred, logits)\
            + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss
