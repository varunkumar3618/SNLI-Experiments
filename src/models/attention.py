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

    def _embedding(self):
        reg = tf.contrib.layers.l2_regularizer(self._l2_reg)
        with tf.variable_scope("embedding"):
            # Premise and hypothesis embedding tensors, both with shape
            # [batch_size, max_len_seq, word_embed_dim]
            prem_embed = get_embedding(self.sentence1_placeholder, self._embedding_matrix,
                                       self._update_embeddings)
            hyp_embed = get_embedding(self.sentence2_placeholder, self._embedding_matrix,
                                      self._update_embeddings, reuse=True)

            prem_embed = tf.layers.dropout(prem_embed, self.dropout_placeholder)
            hyp_embed = tf.layers.dropout(hyp_embed, self.dropout_placeholder)

            prem_proj = tf.layers.dense(prem_embed, self._hidden_size,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        kernel_regularizer=reg,
                                        activation=tf.tanh, name="prem_proj")
            hyp_proj = tf.layers.dense(hyp_embed, self._hidden_size,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       kernel_regularizer=reg,
                                       activation=tf.tanh, name="hyp_proj")
        return prem_proj, hyp_proj

    def _encoding(self, prem_proj, hyp_proj):
        with tf.variable_scope("encoding"):
            cell = tf.contrib.rnn.LSTMCell(
                self._hidden_size,
                use_peepholes=self._use_peepholes,
                initializer=tf.contrib.layers.xavier_initializer()
            )
            with tf.variable_scope("prem_encoder"):
                # prem_states.shape => [batch_size, max_len_seq, _hidden_size]
                prem_hiddens, prem_final_state \
                  = tf.nn.dynamic_rnn(cell, prem_proj, dtype=tf.float32,
                                      sequence_length=self.sentence1_lens_placeholder)

            with tf.variable_scope("hyp_encoder"):
                # hyp_states.shape => [batch_size, max_len_seq, _hidden_size]
                # Hardcoded to use an LSTMCell
                hyp_hiddens, hyp_final_state\
                    = tf.nn.dynamic_rnn(cell, hyp_proj, initial_state=prem_final_state,
                                        sequence_length=self.sentence2_lens_placeholder)
        return prem_hiddens, prem_final_state, hyp_hiddens, hyp_final_state

    def _attention(self, prem_hiddens, prem_final_state, hyp_hiddens, hyp_final_state):
        reg = tf.contrib.layers.l2_regularizer(self._l2_reg)
        hyp_final_hidden = hyp_final_state[1]
        with tf.variable_scope("attention"):
            M_prem = tf.layers.dense(prem_hiddens, self._hidden_size,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=reg,
                                     name="M_prem")
            M_hyp_final = tf.layers.dense(hyp_final_hidden, self._hidden_size,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          kernel_regularizer=reg,
                                          name="M_hyp_final")
            M = tf.tanh(M_prem + tf.expand_dims(M_hyp_final, axis=1))

            A = tf.layers.dense(M, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=reg, name="A")
            A = tf.squeeze(A, axis=2)
            alpha = tf.nn.softmax(A)

            r = tf.reduce_sum(prem_hiddens * tf.expand_dims(alpha, axis=2), axis=1)

            h_star = tf.layers.dense(tf.concat([r, hyp_final_hidden], 1), self._hidden_size,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=reg,
                                     activation=tf.tanh,
                                     name="h_star")
        return h_star

    def _classification(self, h_star):
        reg = tf.contrib.layers.l2_regularizer(self._l2_reg)
        with tf.variable_scope("classification"):
            logits = tf.layers.dense(h_star, 3,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=reg,
                                     name="logits")
            preds = tf.argmax(logits, axis=1)
            return preds, logits

    def add_prediction_op(self):
        with tf.variable_scope("prediction"):
            embedded = self._embedding()
            encoded = self._encoding(*embedded)
            h_star = self._attention(*encoded)
            preds, logits = self._classification(h_star)
        return preds, logits
