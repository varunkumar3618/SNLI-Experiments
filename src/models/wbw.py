import tensorflow as tf

from src.models.attention import AttentionModel
from src.utils.ops import get_embedding, matmul_3d_1d, matmul_3d_2d


class WBWCell(tf.contrib.rnn.RNNCell):
    def __init__(self, hidden_size, subject, initializer, regularizer):
        self._hidden_size = hidden_size
        self._subject = subject
        self._initializer = initializer
        self._regularizer = regularizer

    @property
    def state_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        return self._hidden_size

    def __call__(self, inputs, state, scope=None):
        scope = scope or type(self).__name__

        with tf.variable_scope(scope):
            M_prem = tf.layers.dense(self._subject, self._hidden_size,
                                     kernel_initializer=self._initializer,
                                     kernel_regularizer=self._regularizer,
                                     name="M_prem")
            M_in = tf.layers.dense(inputs, self._hidden_size,
                                   kernel_initializer=self._initializer,
                                   kernel_regularizer=self._regularizer,
                                   name="M_in")
            M_state = tf.layers.dense(state, self._hidden_size,
                                      kernel_initializer=self._initializer,
                                      kernel_regularizer=self._regularizer,
                                      name="M_state")
            M = tf.tanh(M_prem + tf.expand_dims(M_in, axis=1) + tf.expand_dims(M_state, axis=1))

            A = tf.layers.dense(M, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=self._regularizer, name="A")
            A = tf.squeeze(A, axis=2)
            alpha = tf.nn.softmax(A)

            r_subject = tf.reduce_sum(self._subject * tf.expand_dims(alpha, axis=2), axis=1)
            r_state = tf.layers.dense(state, self._hidden_size,
                                      kernel_initializer=self._initializer,
                                      kernel_regularizer=self._regularizer,
                                      activation=tf.tanh,
                                      name="r_state")
            r = r_subject + r_state

            return r

class WBWModel(AttentionModel):
    def _attention(self, prem_hiddens, prem_final_state, hyp_hiddens, hyp_final_state):
        reg = tf.contrib.layers.l2_regularizer(self._l2_reg)
        hyp_final_hidden = hyp_final_state[1]
        with tf.variable_scope("attention"):
            att_cell = WBWCell(self._hidden_size, prem_hiddens, tf.contrib.layers.xavier_initializer(), reg)
            with tf.variable_scope("attention"):
                _, (_, r_final) = tf.nn.dynamic_rnn(att_cell, hyp_hiddens, dtype=tf.float32,
                                                    sequence_length=self.sentence2_lens_placeholder)
                h_star = tf.layers.dense(tf.concat([r_final, hyp_final_hidden], axis=1),
                                         self._hidden_size,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         kernel_regularizer=reg,
                                         activation=tf.tanh,
                                         name="h_star")
        return h_star
