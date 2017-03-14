import tensorflow as tf

from src.models.model import SNLIModel
from src.utils.ops import get_embedding


class SumOfWords(SNLIModel):

    def __init__(self, embedding_matrix, update_embeddings,
                 hidden_size, l2_reg, 
                 use_dropout=True,
                 *args, **kwargs):
        super(SumOfWords, self).__init__(
            use_dropout=use_dropout, *args, **kwargs)
        self._embedding_matrix = embedding_matrix
        self._update_embeddings = update_embeddings
        self._l2_reg = l2_reg
        self._hidden_size = hidden_size
        

    def embedding(self):
        reg = tf.contrib.layers.l2_regularizer(self._l2_reg)
        with tf.variable_scope("embedding"):
            prem_embed = get_embedding(self.sentence1_placeholder, self._embedding_matrix,
                                       self._update_embeddings, self._train_unseen_vocab,
                                       self._missing_indices)
            hyp_embed = get_embedding(self.sentence2_placeholder, self._embedding_matrix,
                                      self._update_embeddings, self._train_unseen_vocab,
                                      self._missing_indices, reuse=True)

            prem_embed = self.apply_dropout(prem_embed)
            hyp_embed = self.apply_dropout(hyp_embed)

            prem_proj = tf.layers.dense(prem_embed, self._hidden_size / 2,
                                        kernel_initializer=self.dense_init,
                                        kernel_regularizer=reg,
                                        activation=self.activation, name="prem_proj")
            hyp_proj = tf.layers.dense(hyp_embed, self._hidden_size / 2,
                                       kernel_initializer=self.dense_init,
                                       kernel_regularizer=reg,
                                       activation=self.activation, name="hyp_proj")

        return prem_proj, hyp_proj

    def encoding(self, prem_proj, hyp_proj):
        with tf.variable_scope("encoding"):
            prem_sow = tf.reduce_sum(prem_proj, axis=1)
            hyp_sow = tf.reduce_sum(hyp_proj, axis=1)
            both_sow = tf.concat([prem_sow, hyp_sow], axis=1)
            both_sow = self.apply_dropout(both_sow)
        return both_sow

    def classification(self, encoded):
        reg = tf.contrib.layers.l2_regularizer(self._l2_reg)
        with tf.variable_scope("classification"):
            h1 = tf.layers.dense(encoded, self._hidden_size,
                                 kernel_initializer=self.dense_init,
                                 kernel_regularizer=reg,
                                 activation=self.activation, name="h1")
            h2 = tf.layers.dense(h1, self._hidden_size,
                                 kernel_initializer=self.dense_init,
                                 kernel_regularizer=reg,
                                 activation=self.activation, name="h2")
            h3 = tf.layers.dense(h2, self._hidden_size,
                                 kernel_initializer=self.dense_init,
                                 kernel_regularizer=reg,
                                 activation=self.activation, name="h3")
            logits = tf.layers.dense(h3, 3,
                                     kernel_initializer=self.dense_init,
                                     kernel_regularizer=reg,
                                     name="logits")
            preds = tf.argmax(logits, axis=1)
            return preds, logits

    def add_prediction_op(self):
        with tf.variable_scope("prediction"):
            prem_proj, hyp_proj = self.embedding()
            encoded = self.encoding(prem_proj, hyp_proj)
            preds, logits = self.classification(encoded)
        return preds, logits
