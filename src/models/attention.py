import tensorflow as tf

from src.models.model import SNLIModel
from src.utils.ops import get_embedding, matmul_3d_1d, matmul_3d_2d

class AttentionModel(SNLIModel):
    def __init__(self, embedding_matrix, update_embeddings,
                 hidden_size, use_peepholes,
                 l2_reg,
                 *args, **kwargs):
        super(AttentionModel, self).__init__(use_lens=True, use_dropout=True, *args, **kwargs)
        self._embedding_matrix = embedding_matrix
        self._update_embeddings = update_embeddings
        self._hidden_size = hidden_size
        self._use_peepholes = use_peepholes
        self._l2_reg = l2_reg

    def add_prediction_op(self):
        with tf.variable_scope("prediction"):
            reg = tf.contrib.layers.l2_regularizer(self._l2_reg)

            # Premise and hypothesis embedding tensors, both with shape
            # [batch_size, max_len_seq, word_embed_dim]
            prem_embed = get_embedding(self.sentence1_placeholder, self._embedding_matrix,
                                       self._update_embeddings)
            hyp_embed = get_embedding(self.sentence2_placeholder, self._embedding_matrix,
                                      self._update_embeddings, reuse=True)

            prem_embed = tf.layers.dropout(prem_embed, self.dropout_placeholder)
            hyp_embed = tf.layers.dropout(hyp_embed, self.dropout_placeholder)

            prem_proj = tf.layers.dense(prem_embed, self._hidden_size / 2,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        kernel_regularizer=reg,
                                        activation=tf.tanh, name="prem_proj")
            hyp_proj = tf.layers.dense(hyp_embed, self._hidden_size / 2,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       kernel_regularizer=reg,
                                       activation=tf.tanh, name="hyp_proj")

            # Use LSTMs to run through the entire premise and hypothesis vectors.
            # Initialize the initial hidden state of the hypothesis LSTM using the final premise
            # hidden state.
            cell = tf.contrib.rnn.LSTMCell(
                self._hidden_size / 2,
                use_peepholes=self._use_peepholes,
                initializer=tf.contrib.layers.xavier_initializer()
            )
            with tf.variable_scope("prem_encoder"):
                # prem_states.shape => [batch_size, max_len_seq, _hidden_size]
                prem_states, prem_final_state \
                  = tf.nn.dynamic_rnn(cell, prem_embed, dtype=tf.float32,
                                      sequence_length=self.sentence1_lens_placeholder)

            # TODO(kennyleung): determine whether a fresh instance of the LSTMCell is needed.
            # Also, consider using a delimiter and running through one LSTM instance as in the paper.
            with tf.variable_scope("hyp_encoder"):
                # hyp_states.shape => [batch_size, max_len_seq, _hidden_size]
                # Hardcoded to use an LSTMCell
                hyp_states, (_, final_hyp_hidden)\
                    = tf.nn.dynamic_rnn(cell, hyp_embed, initial_state=prem_final_state,
                                        sequence_length=self.sentence2_lens_placeholder)

            both_encoded = tf.concat([prem_final_state[1], final_hyp_hidden], axis=1)
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

            # with tf.variable_scope("attention"):
            #     W_y = tf.get_variable("W_y", shape=[self._hidden_size, self._hidden_size],
            #                           initializer=tf.contrib.layers.xavier_initializer())
            #     W_h = tf.get_variable("W_h", shape=[self._hidden_size, self._hidden_size],
            #                           initializer=tf.contrib.layers.xavier_initializer())
            #     w = tf.get_variable("w", shape=[self._hidden_size],
            #                         initializer=tf.contrib.layers.xavier_initializer())
            #     W_p = tf.get_variable("W_p", shape=[self._hidden_size, self._hidden_size],
            #                           initializer=tf.contrib.layers.xavier_initializer())
            #     W_x = tf.get_variable("W_x", shape=[self._hidden_size, self._hidden_size],
            #                           initializer=tf.contrib.layers.xavier_initializer())

            #     # Matrix-multiply each horizontal slice of prem_states by weight matrix W_y,
            #     # and use broadcasting to add outer(W_h*h_n, e_L).
            #     # M.shape => [batch_size, max_len_seq, _hidden_size]
            #     M = tf.tanh(tf.tensordot(prem_states, W_y, axes=1) \
            #         + tf.expand_dims(tf.matmul(final_hyp_hidden, W_h), axis=1))
            #     M.set_shape([None, None, self._hidden_size])

            #     # alpha.shape => [batch_size, max_len_seq]
            #     alpha = tf.nn.softmax(tf.tensordot(M, w, axes=1))
            #     # r.shape => [batch_size, _hidden_size]
            #     r = tf.reduce_sum(prem_states * tf.expand_dims(alpha, axis=2), axis=1)

            #     logits = tf.tanh(tf.matmul(r, W_p) + tf.matmul(final_hyp_hidden, W_x))
            #     preds = tf.argmax(logits, axis=1)
        return preds, logits
