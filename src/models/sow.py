import tensorflow as tf

from src.models.model import SNLIModel
from src.utils.ops import get_embedding


class SumOfWords(SNLIModel):
    def __init__(self, embedding_matrix, update_embeddings,
                 hidden_size,
                 *args, **kwargs):
        super(SumOfWords, self).__init__(use_lens=False, *args, **kwargs)
        self._embedding_matrix = embedding_matrix
        self._update_embeddings = update_embeddings
        self._hidden_size = hidden_size

    def add_prediction_op(self):
        with tf.variable_scope("prediction"):
            prem_embed = get_embedding(self.sentence1_placeholder, self._embedding_matrix,
                                       self._update_embeddings)
            hyp_embed = get_embedding(self.sentence2_placeholder, self._embedding_matrix,
                                      self._update_embeddings, reuse=True)

            prem_sow = tf.reduce_sum(prem_embed, axis=1)
            hyp_sow = tf.reduce_sum(hyp_embed, axis=1)
            both_sow = tf.concat([prem_sow, hyp_sow], axis=1)

            h1 = tf.layers.dense(both_sow, self._hidden_size,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.tanh, name="h1")
            h2 = tf.layers.dense(h1, self._hidden_size,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.tanh, name="h2")
            h3 = tf.layers.dense(h2, self._hidden_size,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.tanh, name="h3")
            logits = tf.layers.dense(h3, self._hidden_size,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name="logits")
            preds = tf.argmax(logits, axis=1)
            return preds, logits
