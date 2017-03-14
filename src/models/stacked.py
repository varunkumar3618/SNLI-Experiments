import tensorflow as tf

from src.models.model import SNLIModel
from src.utils.ops import get_embedding

class StackedAttentionModel(SNLIModel):
    def __init__(self, embedding_matrix, update_embeddings,
                 hidden_size, use_peepholes,
                 l2_reg, num_att_layers=2, use_skip=True,
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
        self._use_skip = True

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
            #     (prem_fw_hiddens, prem_bw_hiddens), prem_final_state\
            #         = tf.nn.bidirectional_dynamic_rnn(cell, cell, prem, dtype=tf.float32,
            #                                           sequence_length=self.sentence1_lens_placeholder)
            #     prem_hiddens = tf.concat([prem_fw_hiddens, prem_bw_hiddens], axis=2)
                prem_hiddens, prem_final_state \
                  = tf.nn.dynamic_rnn(cell, prem, dtype=tf.float32,
                                      sequence_length=self.sentence1_lens_placeholder)

            with tf.variable_scope("hyp_encoder"):
            #     (hyp_fw_hiddens, hyp_bw_hiddens), hyp_final_state \
            #       = tf.nn.bidirectional_dynamic_rnn(cell, cell, prem, dtype=tf.float32,
            #                           sequence_length=self.sentence1_lens_placeholder)
            #     hyp_hiddens = tf.concat([hyp_fw_hiddens, hyp_bw_hiddens], axis=2)
                hyp_hiddens, hyp_final_state\
                    = tf.nn.dynamic_rnn(cell, hyp, initial_state=prem_final_state,
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
            alpha = Num_alpha / tf.expand_dims(Den_alpha, axis=2)
        return beta, alpha

    def all_attention(self, prem, hyp, scope):
        reg = tf.contrib.layers.l2_regularizer(self._l2_reg)
        with tf.variable_scope(scope):
            beta, alpha = self.attention(prem, hyp, "inter")
        return tf.concat([prem, beta], axis=2), tf.concat([hyp, alpha], axis=2)

    def add_skip(self, prem_orig, hyp_orig, prem_repr, hyp_repr, scope):
        reg = tf.contrib.layers.l2_regularizer(self._l2_reg)
        with tf.variable_scope(scope):
            G_prem_in = tf.concat([prem_orig, prem_repr], axis=2)
            G_prem = tf.layers.dense(G_prem_in, self._hidden_size,
                                     kernel_initializer=self.dense_init,
                                     kernel_regularizer=reg,
                                     activation=tf.nn.sigmoid, name="prem_gate")
            prem = G_prem * prem_orig + (1 - G_prem) * prem_repr

            G_hyp_in = tf.concat([hyp_orig, hyp_repr], axis=2)
            G_hyp = tf.layers.dense(G_prem_in, self._hidden_size,
                                    kernel_initializer=self.dense_init,
                                    kernel_regularizer=reg,
                                    activation=tf.nn.sigmoid, name="hyp_gate")
            hyp = G_hyp * hyp_orig + (1 - G_hyp) * hyp_repr
        return prem, hyp

    def collection(self, prem_hiddens, prem_tilda, hyp_hiddens, hyp_tilda):
        def make_collection(x, x_tilda):
            tensors = [
                x,
                x_tilda,
                x - x_tilda,
                x * x_tilda
            ]
            return tf.concat(tensors, axis=2)
        return make_collection(prem_hiddens, prem_tilda), make_collection(hyp_hiddens, hyp_tilda)

    def pooling(self, prem_composed, hyp_composed):
        prem_avg_pool = tf.reduce_mean(prem_composed, axis=1)
        prem_max_pool = tf.reduce_max(prem_composed, axis=1)
        hyp_avg_pool = tf.reduce_mean(hyp_composed, axis=1)
        hyp_max_pool = tf.reduce_mean(hyp_composed, axis=1)

        return prem_avg_pool, prem_max_pool, hyp_avg_pool, hyp_max_pool

    def full_layer(self, prem, hyp, scope):
        with tf.variable_scope(scope):
            prem_hiddens, _, hyp_hiddens, _ = self.encoding(prem, hyp, "encoding")
            prem_att, hyp_att = self.all_attention(prem_hiddens, hyp_hiddens, "attention")
            prem_proj, hyp_proj = self.projection(prem_att, hyp_att, "projection")
        return prem_proj, hyp_proj

    def classification(self, h_star):
        reg = tf.contrib.layers.l2_regularizer(self._l2_reg)
        with tf.variable_scope("classification"):
            hidden = tf.layers.dense(h_star, self._hidden_size,
                                     kernel_initializer=self.dense_init,
                                     kernel_regularizer=reg,
                                     activation=self.activation,
                                     name="hidden")
            logits = tf.layers.dense(hidden, 3,
                                     kernel_initializer=self.dense_init,
                                     kernel_regularizer=reg,
                                     name="logits")
            preds = tf.argmax(logits, axis=1)
            return preds, logits

    def add_prediction_op(self):
        with tf.variable_scope("prediction"):
            prem_embed, hyp_embed = self.embedding()
            prem_hiddens, _, hyp_hiddens, _ = self.encoding(prem_embed, hyp_embed, "encoding")
            prem_tilda, hyp_tilda = self.attention(prem_hiddens, hyp_hiddens, "attention")
            prem_m, hyp_m = self.collection(prem_hiddens, prem_tilda, hyp_hiddens, hyp_tilda)
            prem_composed, _, hyp_composed, _ = self.encoding(prem_m, hyp_m, "composition")
            prem_avg_pool, prem_max_pool, hyp_avg_pool, hyp_max_pool\
                = self.pooling(prem_composed, hyp_composed)
            h_star = tf.concat([prem_avg_pool, prem_max_pool, hyp_avg_pool, hyp_max_pool], axis=1)
            preds, logits = self.classification(h_star)
        return preds, logits
