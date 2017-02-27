import tensorflow as tf

from src.models.model import SNLIModel


class SumOfWords(SNLIModel):
    def __init__(self, embedding_matrix, update_embeddings,
                 hidden_size,
                 *args, **kwargs):
        super(SumOfWords, self).__init__(use_lens=False, *args, **kwargs)
        self._embedding_matrix = embedding_matrix
        self._update_embeddings = update_embeddings
        self._hidden_size = hidden_size

    def add_prediction_op(self):
        prem_embed = self._get_embedding(self.sentence1_placeholder)
        hyp_embed = self._get_embedding(self.sentence2_placeholder, reuse=True)

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
    
    def _get_embedding(self, indices, reuse=False):
        with tf.variable_scope("embeddings", reuse=reuse):
            if self._update_embeddings:
                embeddings = tf.get_variable(
                    "E",
                    initializer=tf.constant(self._embedding_matrix, dtype=tf.float32)
                )
            else:
                embeddings = self._embedding_matrix
            embedded_vectors = tf.nn.embedding_lookup(self._embedding_matrix, indices)
            output_shape = [tf.shape(indices)[0], tf.shape(indices)[1], self._embedding_matrix.shape[1]]
            embeddings = tf.reshape(embedded_vectors, shape=output_shape)
        return embeddings