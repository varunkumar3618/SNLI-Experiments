import tensorflow as tf

from src.models.model import SNLIModel
from src.utils.ops import get_embedding, matmul_3d_1d, matmul_3d_2d

class AttentionModel(SNLIModel):
    def __init__(self, embedding_matrix, update_embeddings,
                 hidden_size, use_peepholes,
                 *args, **kwargs):
        super(AttentionModel, self).__init__(use_lens=True, *args, **kwargs)
        self._embedding_matrix = embedding_matrix
        self._update_embeddings = update_embeddings
        self._hidden_size = hidden_size
        self._use_peepholes = use_peepholes

    def add_prediction_op(self):
        with tf.variable_scope("prediction"):
            prem_embed = get_embedding(self.sentence1_placeholder, self._embedding_matrix,
                                       self._update_embeddings)
            hyp_embed = get_embedding(self.sentence2_placeholder, self._embedding_matrix,
                                      self._update_embeddings, reuse=True)

            cell = tf.contrib.rnn.LSTMCell(self._hidden_size, use_peepholes=self._use_peepholes)
            with tf.variable_scope("prem_encoder"):
                prem_states, prem_final_state\
                  = tf.nn.dynamic_rnn(cell, prem_embed, dtype=tf.float32,
                                      sequence_length=self.sentence1_lens_placeholder)
            with tf.variable_scope("hyp_encoder"):
                hyp_states, _ = tf.nn.dynamic_rnn(cell, hyp_embed, initial_state=prem_final_state,
                                                  sequence_length=self.sentence2_lens_placeholder)
                final_hyp_state = tf.squeeze(
                    tf.slice(hyp_states, [0, tf.shape(hyp_states)[1] - 1, 0], [-1, 1, -1]),
                    axis=1
                )

            with tf.variable_scope("attention"):
                W_y = tf.get_variable("W_y", shape=[self._hidden_size, self._hidden_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
                W_h = tf.get_variable("W_h", shape=[self._hidden_size, self._hidden_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
                w = tf.get_variable("w", shape=[self._hidden_size],
                                    initializer=tf.contrib.layers.xavier_initializer())
                W_p = tf.get_variable("W_p", shape=[self._hidden_size, self._hidden_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
                W_x = tf.get_variable("W_x", shape=[self._hidden_size, self._hidden_size],
                                      initializer=tf.contrib.layers.xavier_initializer())

                M = matmul_3d_2d(prem_states, W_y, lastdim=True)\
                    + tf.expand_dims(tf.matmul(final_hyp_state, W_h), axis=1)
                alpha = tf.nn.softmax(matmul_3d_1d(M, w))
                r = tf.reduce_sum(prem_states + tf.expand_dims(alpha, axis=2), axis=2)

                logits = tf.tanh(tf.matmul(r, W_p) + tf.matmul(final_hyp_state, W_x))
                preds = tf.argmax(logits, axis=1)
        return preds, logits
