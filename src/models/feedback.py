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

    def encoding(self, prem, hyp, scope):
        reg = tf.contrib.layers.l2_regularizer(self._l2_reg)
        with tf.variable_scope(scope):
            fw_cell = tf.contrib.rnn.LSTMCell(
                self._hidden_size / 2,
                use_peepholes=self._use_peepholes,
                initializer=self.rec_init
            )
            bw_cell = tf.contrib.rnn.LSTMCell(
                self._hidden_size / 2,
                use_peepholes=self._use_peepholes,
                initializer=self.rec_init
            )
            fw_cell = self.apply_dropout_wrapper(fw_cell)
            bw_cell = self.apply_dropout_wrapper(bw_cell)
            with tf.variable_scope("prem_encoder"):
                (prem_fw_hiddens, prem_bw_hiddens), prem_final_state\
                    = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, prem, dtype=tf.float32,
                                                      sequence_length=self.sentence1_lens_placeholder)
                prem_hiddens = tf.concat([prem_fw_hiddens, prem_bw_hiddens], axis=2)

            with tf.variable_scope("hyp_encoder"):
                (hyp_fw_hiddens, hyp_bw_hiddens), hyp_final_state \
                  = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, hyp, dtype=tf.float32,
                                      sequence_length=self.sentence2_lens_placeholder)
                hyp_hiddens = tf.concat([hyp_fw_hiddens, hyp_bw_hiddens], axis=2)

        return prem_hiddens, prem_final_state, hyp_hiddens, hyp_final_state

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

    def collection(self, prem_hiddens, prem_tilda, hyp_hiddens, hyp_tilda,
                   global_memory, local_prem_memory, local_hyp_memory):
        def make_collection(x, x_tilda, global_memory, x_memory):
            tensors = [
                x,
                x_tilda,
                x - x_tilda,
                x * x_tilda,
                x - tf.expand_dims(global_memory, axis=1),
                x * tf.expand_dims(global_memory, axis=1),
                x - x_memory,
                x * x_memory
            ]
            return tf.concat(tensors, axis=2)
        return make_collection(prem_hiddens, prem_tilda, global_memory, local_prem_memory),\
            make_collection(hyp_hiddens, hyp_tilda, global_memory, local_hyp_memory)

    def pooling(self, prem_composed, hyp_composed):
        prem_avg_pool = tf.reduce_mean(prem_composed, axis=1)
        prem_max_pool = tf.reduce_max(prem_composed, axis=1)
        hyp_avg_pool = tf.reduce_mean(hyp_composed, axis=1)
        hyp_max_pool = tf.reduce_mean(hyp_composed, axis=1)

        return prem_avg_pool, prem_max_pool, hyp_avg_pool, hyp_max_pool

    def episode(self, prem_hiddens, hyp_hiddens, global_memory, local_prem_memory, local_hyp_memory, scope):
        reg = tf.contrib.layers.l2_regularizer(self._l2_reg)
        with tf.variable_scope(scope):
            prem_tilda, hyp_tilda = self.attention(prem_hiddens, hyp_hiddens, scope)
            prem_m, hyp_m = self.collection(prem_hiddens, prem_tilda, hyp_hiddens, hyp_tilda,
                                            global_memory, local_prem_memory, local_hyp_memory)
            prem_composed, hyp_composed = self.projection(prem_m, hyp_m, "composition")

            prem_avg_pool, prem_max_pool, hyp_avg_pool, hyp_max_pool\
                = self.pooling(prem_composed, hyp_composed)
            global_features = tf.concat([prem_avg_pool, prem_max_pool, hyp_avg_pool, hyp_max_pool], axis=1)
            global_episode = tf.layers.dense(global_features, self._hidden_size,
                                             kernel_initializer=self.dense_init,
                                             kernel_regularizer=reg,
                                             activation=self.activation,
                                             name="global")
        return global_episode, prem_composed, hyp_composed

    def gating(self, old, new, scope):
        with tf.variable_scope(scope):
            old_proj = tf.layers.dense(old, self._hidden_size,
                                       kernel_initializer=self.dense_init,
                                       kernel_regularizer=reg,
                                       use_bias=False,
                                       name="old_proj")
            hidden_axis = old_proj.get_shape().ndims - 1
            e = tf.expand_dims(tf.reduce_sum(old_proj * new, axis=hidden_axis), axis=hidden_axis)
            updated = e * old + (1 - e) * new
        return updated

    def add_prediction_op(self):
        with tf.variable_scope("prediction"):
            prem_embed, hyp_embed = self.embedding()
            prem_hiddens, _, hyp_hiddens, _ = self.encoding(prem_embed, hyp_embed, "encoding")

            global_memory = tf.zeros([tf.shape(prem_hiddens)[0], self._hidden_size])
            local_prem_memory = tf.zeros([tf.shape(prem_hiddens)[0], tf.shape(prem_hiddens)[1], self._hidden_size])
            local_hyp_memory = tf.zeros([tf.shape(hyp_hiddens)[0], tf.shape(hyp_hiddens)[1], self._hidden_size])

            with tf.variable_scope("feedback"):
                for i in range(self._feedback_iters):
                    global_episode, local_prem_episode, local_hyp_episode\
                        = self.episode(
                            prem_hiddens, hyp_hiddens,
                            global_memory,
                            local_prem_memory, local_hyp_memory,
                            "episode"
                        )
                    global_memory = self.gating(global_memory, global_episode, "global_gating")
                    local_prem_memory = self.gating(local_prem_memory, local_prem_episode, "prem_gating")
                    local_hyp_memory = self.gating(local_hyp_memory, local_hyp_episode, "hyp_gating")

                    if i == 0:
                        tf.get_variable_scope().reuse_variables()

            preds, logits = self.classification(memory)
            return preds, logits
