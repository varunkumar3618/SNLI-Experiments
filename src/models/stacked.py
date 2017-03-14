import tensorflow as tf

from src.models.model import SNLIModel
from src.utils.ops import get_embedding

class StackedAttentionModel(SNLIModel):
    def __init__(self, embedding_matrix, update_embeddings,
                 hidden_size, use_peepholes,
                 l2_reg, num_att_layers=2,
                 use_lens=True, use_dropout=True,
                 *args, **kwargs):
        super(StackedAttentionModel, self).__init__(use_lens=use_lens, use_dropout=use_dropout,
                                                    *args, **kwargs)
        self._embedding_matrix = embedding_matrix
        self._update_embeddings = update_embeddings
        self._hidden_size = hidden_size
        self._use_peepholes = use_peepholes
        self._l2_reg = l2_reg
        self._num_att_layers = num_att_layers

    def embedding(self):
        with tf.variable_scope("embedding"):
            # Premise and hypothesis embedding tensors, both with shape
            # [batch_size, max_len_seq, word_embed_dim]
            prem_embed = get_embedding(self.sentence1_placeholder, self._embedding_matrix,
                                       self._update_embeddings)
            hyp_embed = get_embedding(self.sentence2_placeholder, self._embedding_matrix,
                                      self._update_embeddings, reuse=True)
        return prem_embed, hyp_embed

    def projection(self, prem, hyp, scope):
        reg = tf.contrib.layers.l2_regularizer(self._l2_reg)
        with tf.variable_scope(scope):

            prem_proj = tf.layers.dense(prem, self._hidden_size,
                                        kernel_initializer=self.dense_init,
                                        kernel_regularizer=reg,
                                        activation=self.activation, name="prem_proj")
            hyp_proj = tf.layers.dense(hyp, self._hidden_size,
                                       kernel_initializer=self.dense_init,
                                       kernel_regularizer=reg,
                                       activation=self.activation, name="hyp_proj")
        return self.apply_dropout(prem_proj), self.apply_dropout(hyp_proj)

    def encoding(self, prem, hyp, scope):
        reg = tf.contrib.layers.l2_regularizer(self._l2_reg)
        with tf.variable_scope(scope):
            cell = tf.contrib.rnn.LSTMCell(
                self._hidden_size,
                use_peepholes=self._use_peepholes,
                initializer=self.rec_init
            )
            with tf.variable_scope("prem_encoder"):
                # prem_states.shape => [batch_size, max_len_seq, _hidden_size]
                prem_hiddens, prem_final_state \
                  = tf.nn.dynamic_rnn(cell, prem, dtype=tf.float32,
                                      sequence_length=self.sentence1_lens_placeholder)

            with tf.variable_scope("hyp_encoder"):
                # hyp_states.shape => [batch_size, max_len_seq, _hidden_size]
                # Hardcoded to use an LSTMCell
                hyp_hiddens, hyp_final_state\
                    = tf.nn.dynamic_rnn(cell, hyp, initial_state=prem_final_state,
                                        sequence_length=self.sentence2_lens_placeholder)
        return prem_hiddens, prem_final_state, hyp_hiddens, hyp_final_state

    def attention(self, x, y, scope):
        with tf.variable_scope(scope):
            x_att, y_att = self.projection(x, y, "projection_att")
            x_subj, y_subj = self.projection(x, y, "projection_subj")

            E = tf.exp(tf.einsum("aij,ajk->aik", x_tt, tf.transpose(y_att, perm=[0, 2, 1])))

            Num_beta = tf.einsum("aij,ajk->aik", E, y_subj)
            Den_beta = tf.reduce_sum(E, axis=2)
            beta = Num_beta / tf.expand_dims(Den_beta, axis=2)

            Num_alpha = tf.einsum("aij,aik->ajk", E, x_subj)
            Den_alpha = tf.reduce_sum(E, axis=1)
            alpha = Num_alpha / tf.expand_dims(Den_alpha, axis=2)
        return beta, alpha

    def all_attention(self, prem, hyp, scope):
        reg = tf.contrib.layers.l2_regularizer(self._l2_reg)
        with tf.variable_scope(scope):
            beta, alpha = self.attention(prem, hyp, "inter")
        return tf.concat([prem, beta], axis=2), tf.concat([hyp, alpha], axis=2)

    def full_layer(self, prem, hyp, scope):
        with tf.variable_scope(scope):
            prem_hiddens, _, hyp_hiddens, _ = self.encoding(prem, hyp, "encoding")
            prem_att, hyp_att = self.all_attention(prem_hiddens, hyp_hiddens, "attention")
            prem_proj, hyp_proj = self.projection(prem_att, hyp_att, "projection")
        return prem_proj, hyp_proj

    def classification(self, h_star):
        reg = tf.contrib.layers.l2_regularizer(self._l2_reg)
        with tf.variable_scope("classification"):
            logits = tf.layers.dense(h_star, 3,
                                     kernel_initializer=self.dense_init,
                                     kernel_regularizer=reg,
                                     name="logits")
            preds = tf.argmax(logits, axis=1)
            return preds, logits

    def add_prediction_op(self):
        with tf.variable_scope("prediction"):
            prem_embed, hyp_embed = self.embedding()
            prem_proj, hyp_proj = self.projection(prem_embed, hyp_embed, "input_projection")

            for i in range(self._num_att_layers):
                prem_proj, hyp_proj = self.full_layer(prem_proj, hyp_proj, "full%s" % (i + 1))

            _, prem_final_state, _, hyp_final_state\
                = self.encoding(prem_proj, hyp_proj, "output_encoding")
            prem_final_hidden, hyp_final_hidden = prem_final_state[1], hyp_final_state[1]

            h_star = tf.concat([prem_final_hidden, hyp_final_hidden], axis=1)
            preds, logits = self.classification(h_star)
        return preds, logits
