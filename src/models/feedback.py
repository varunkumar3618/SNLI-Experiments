import tensorflow as tf

from src.models.model import SNLIModel

class FeedbackModel(SNLIModel):
    def __init__(self,
                 hidden_size, use_peepholes, feedback_iters,
                 l2_reg, use_lens=True, use_dropout=True,
                 *args, **kwargs):
        super(FeedbackModel, self).__init__(use_lens=use_lens, use_dropout=use_dropout,
                                            *args, **kwargs)
        self._hidden_size = hidden_size
        self._use_peepholes = use_peepholes
        self._l2_reg = l2_reg
        self._feedback_iters = feedback_iters

    def embedding(self):
        with tf.variable_scope("embedding"):
            # Premise and hypothesis embedding tensors, both with shape
            # [batch_size, max_len_seq, word_embed_dim]
            prem_embed = self.embed_indices(self.sentence1_placeholder)
            hyp_embed = self.embed_indices(self.sentence2_placeholder)
        return prem_embed, hyp_embed

    def make_lstm_cell(self):
        cell = tf.contrib.rnn.LSTMCell(
            self._hidden_size,
            use_peepholes=self._use_peepholes,
            initializer=self.rec_init
        )
        cell = self.apply_dropout_wrapper(cell)
        return cell

    def make_gru_cell(self, size):
        cell = tf.contrib.rnn.GRUCell(
            self._hidden_size
        )
        cell = self.apply_dropout_wrapper(cell)
        return cell

    def run_birnns(self, fw_cell, bw_cell, prem, hyp, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope("prem_encoder"):
                (prem_fw_hiddens, prem_bw_hiddens), prem_final_state\
                    = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, prem, dtype=tf.float32,
                                                      sequence_length=self.sentence1_lens_placeholder)
                prem_hiddens = tf.concat([prem_fw_hiddens, prem_bw_hiddens], axis=2)
                prem_final = tf.concat(prem_final_state, axis=1)

            with tf.variable_scope("hyp_encoder"):
                (hyp_fw_hiddens, hyp_bw_hiddens), hyp_final_state \
                  = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, hyp, dtype=tf.float32,
                                      sequence_length=self.sentence2_lens_placeholder)
                hyp_hiddens = tf.concat([hyp_fw_hiddens, hyp_bw_hiddens], axis=2)
                hyp_final = tf.concat(hyp_final_state, axis=1)
        return prem_hiddens, prem_final, hyp_hiddens, hyp_final

    def encoding(self, prem, hyp, scope):
        reg = tf.contrib.layers.l2_regularizer(self._l2_reg)
        fw_cell = self.make_gru_cell(self._hidden_size)
        bw_cell = self.make_gru_cell(self._hidden_size)
        return self.run_birnns(fw_cell, bw_cell, prem, hyp, scope)

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

    def collection(self, prem_hiddens, prem_tilda, hyp_hiddens, hyp_tilda, memory):
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

    def episode(self, prem_hiddens, hyp_hiddens, memory, scope):
        with tf.variable_scope(scope):
            prem_tilda, hyp_tilda = self.attention(prem_hiddens, hyp_hiddens, scope)
            prem_m, hyp_m = self.collection(prem_hiddens, prem_tilda, hyp_hiddens, hyp_tilda, memory)
            prem_composed, _, hyp_composed, _ = self.encoding(prem_m, hyp_m, "composition")
            prem_avg_pool, prem_max_pool, hyp_avg_pool, hyp_max_pool\
                = self.pooling(prem_composed, hyp_composed)
            h_star = tf.concat([prem_avg_pool, prem_max_pool, hyp_avg_pool, hyp_max_pool], axis=1)
        return h_star

    def add_prediction_op(self):
        with tf.variable_scope("prediction"):
            prem_embed, hyp_embed = self.embedding()
            prem_hiddens, _, hyp_hiddens, _ = self.encoding(prem_embed, hyp_embed, "encoding")

            memory = tf.zeros(shape=[tf.shape(prem_embed)[0], self._hidden_size])
            memory_cell = self.make_gru_cell(self._hidden_size)

            with tf.variable_scope("feedback"):
                for i in range(self._feedback_iters):
                    episode = self.episode(prem_hiddens, hyp_hiddens, memory, "attention")
                    memory, _ = memory_cell(episode, memory)

                    if i == 0:
                        tf.get_variable_scope().reuse_variables()

            preds, logits = self.classification(memory)
            return preds, logits
