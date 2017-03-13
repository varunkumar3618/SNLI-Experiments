import tensorflow as tf

from src.models.model import SNLIModel

class StackedAttentionModel(SNLIModel):
    def __init__(self, embedding_matrix, update_embeddings,
                 hidden_size, use_peepholes,
                 l2_reg,
                 use_lens=True, use_dropout=True,
                 *args, **kwargs):
        super(StackedAttentionModel, self).__init__(use_lens=use_lens, use_dropout=use_dropout,
                                                    *args, **kwargs)
        self._embedding_matrix = embedding_matrix
        self._update_embeddings = update_embeddings
        self._hidden_size = hidden_size
        self._use_peepholes = use_peepholes
        self._l2_reg = l2_reg

    def embedding(self):
        with tf.variable_scope("embedding"):
            # Premise and hypothesis embedding tensors, both with shape
            # [batch_size, max_len_seq, word_embed_dim]
            prem_embed = get_embedding(self.sentence1_placeholder, self._embedding_matrix,
                                       self._update_embeddings)
            hyp_embed = get_embedding(self.sentence2_placeholder, self._embedding_matrix,
                                      self._update_embeddings, reuse=True)
        return prem_embed, hyp_embed

    def dropout(self, prem, hyp):
        return self.apply_dropout(prem), self.apply_dropout(hyp)

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
        return prem_proj, hyp_proj

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
                  = tf.nn.dynamic_rnn(cell, prem_proj, dtype=tf.float32,
                                      sequence_length=self.sentence1_lens_placeholder)

            with tf.variable_scope("hyp_encoder"):
                # hyp_states.shape => [batch_size, max_len_seq, _hidden_size]
                # Hardcoded to use an LSTMCell
                hyp_hiddens, hyp_final_state\
                    = tf.nn.dynamic_rnn(cell, hyp_proj, initial_state=prem_final_state,
                                        sequence_length=self.sentence2_lens_placeholder)
        return prem_hiddens, prem_final_state, hyp_hiddens, hyp_final_state

    def attention(self, x, y, scope):
        with tf.variable_scope(scope):
            E = tf.exp(tf.einsum("aij,ajk->aik", x, tf.transpose(y, perm=[0, 2, 1])))

            Num_beta = tf.einsum("aij,ajk->aik", E, y)
            Den_beta = tf.reduce_sum(E, axis=2)
            beta = Num_beta / tf.expand_dims(Den_beta, axis=2)

            Num_alpha = tf.einsum("aij,aik->ajk", E, x)
            Den_alpha = tf.reduce_sum(E, axis=1)
            alpha = Num_alpha / tf.expand_dims(Den_alpha, axis=1)
        return beta, alpha

    def all_attention(self, prem, hyp, scope):
        reg = tf.contrib.layers.l2_regularizer(self._l2_reg)
        with tf.variable_scope(scope):
            beta, alpha = self.attention(prem, hyp, "inter")
        return tf.concat([prem, beta], axis=2), tf.concat([hyp, alpha], axis=2)

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
            prem_proj, hyp_proj = self.projection(prem_embed, hyp_embed, "projection1")
            prem_proj, hyp_proj = self.dropout(prem_proj, hyp_proj)
            prem_hiddens, _, hyp_hiddens, _ = self.encoding(prem_proj, hyp_proj, "encoding1")
            prem_att, hyp_att = self.all_attention(prem_hiddens, hyp_hiddens, "attention1")

            prem_proj, hyp_proj = self.projection(prem_att, hyp_att, "projection1")
            prem_proj, hyp_proj = self.dropout(prem_proj, hyp_proj)
            _, prem_final_state, _, hyp_final_state = self.encoding(prem_att, hyp_att, "encoding2")
            prem_final_hidden, hyp_final_hidden = prem_final_state[1], hyp_final_state[1]

            h_star = tf.concat([prem_final_hidden, hyp_final_hidden], axis=1)
            preds, logits = self.classification(h_star)
        return preds, logits
