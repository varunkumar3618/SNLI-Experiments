import tensorflow as tf

from src.models.model import SNLIModel

def cosine_matching(x, y, d, l, initializer, regularizer, scope):
    with tf.variable_scope(scope):
        W = tf.get_variable("W", shape=[d, l],
                            initializer=initializer,
                            regularizer=regularizer)
        W = tf.expand_dims(tf.expand_dims(W, axis=0), axis=0)

        x_pers = tf.expand_dims(x, axis=2) * W
        x_pers = tf.expand_dims(y, axis=2) * W

        x_mags = tf.sqrt(tf.reduce_sum(x_pers * x_pers, axis=3))
        y_mags = tf.sqrt(tf.reduce_sum(y_pers * y_pers, axis=3))

        xy = tf.reduce_sum(x_mags * y_mags, axis=3)

        match = xy / (x_mags * y_mags)
    return match

def get_bilstm_finals(state):
    fw_state, bw_state = state
    return fw_state[1], bw_state[1]

class MPMatchingModel(SNLIModel):
    def __init__(self,
                 hidden_size, use_peepholes, perspectives,
                 l2_reg, use_lens=True, use_dropout=True,
                 *args, **kwargs):
        super(MPMatchingModel, self).__init__(use_lens=use_lens, use_dropout=use_dropout,
                                              *args, **kwargs)
        self._hidden_size = hidden_size
        self._use_peepholes = use_peepholes
        self._l2_reg = l2_reg
        self._perspectives = perspectives

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
                self._hidden_size,
                use_peepholes=self._use_peepholes,
                initializer=self.rec_init
            )
            fw_cell = self.apply_dropout_wrapper(fw_cell)
            bw_cell = tf.contrib.rnn.LSTMCell(
                self._hidden_size,
                use_peepholes=self._use_peepholes,
                initializer=self.rec_init
            )
            bw_cell = self.apply_dropout_wrapper(bw_cell)
            with tf.variable_scope("prem_encoder"):
                prem_hiddens, prem_final_state\
                    = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, prem, dtype=tf.float32,
                                                      sequence_length=self.sentence1_lens_placeholder)

            with tf.variable_scope("hyp_encoder"):
                hyp_hiddens, hyp_final_state \
                  = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, hyp, dtype=tf.float32,
                                      sequence_length=self.sentence2_lens_placeholder)

        return prem_hiddens, prem_final_state, hyp_hiddens, hyp_final_state

    def matching(self, prem_hiddens, prem_final_state, hyp_hiddens, hyp_final_state, scope):
        reg = tf.contrib.layers.l2_regularizer(self._l2_reg)

        prem_fw_hiddens, prem_bw_hiddens = prem_hiddens
        hyp_fw_hiddens, hyp_bw_hiddens = hyp_hiddens
        prem_fw_final, prem_bw_final = get_bilstm_finals(prem_final_state)
        hyp_fw_final, hyp_bw_final = get_bilstm_finals(hyp_final_state)

        prem_fw_final = tf.expand_dims(prem_fw_final, axis=1)
        prem_bw_final = tf.expand_dims(prem_bw_final, axis=1)
        hyp_fw_final = tf.expand_dims(hyp_fw_final, axis=1)
        hyp_bw_final = tf.expand_dims(hyp_bw_final, axis=1)

        def match_vecs(x, y, scope):
            return cosine_matching(x, y, self._hidden_size, self._perspectives,
                                   self.dense_init, reg, scope)

        with tf.variable_scope(scope):
            # Full matching
            prem_fw_full_match = match_vecs(prem_fw_hiddens, hyp_fw_final, "prem_fw_full")
            prem_bw_full_match = match_vecs(prem_bw_hiddens, hyp_bw_final, "prem_bw_full")
            hyp_fw_full_match = match_vecs(hyp_fw_hiddens, prem_fw_final, "prem_fw_full")
            hyp_bw_full_match = match_vecs(hyp_bw_hiddens, prem_bw_final, "prem_bw_full")

            prem_matched = tf.concat([prem_fw_full_match, prem_bw_full_match], axis=2)
            hyp_matched = tf.concat([hyp_fw_full_match, hyp_bw_full_match], axis=2)
        return prem_matched, hyp_matched

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
            prem_hiddens, prem_final_state, hyp_hiddens, hyp_final_state = self.encoding(prem_embed, hyp_embed, "encoding")
            prem_matched, hyp_matched = self.matching(prem_hiddens, prem_final_state, hyp_hiddens, hyp_final_state, "matching")
            _, prem_final_state, _,hyp_final_state = self.encoding(prem_matched, hyp_matched, "composition")
            h_star = tf.concat(
                list(get_bilstm_finals(prem_final_state)) + list(get_bilstm_finals(hyp_final_state)),
                axis=1
            )
            preds, logits = self.classification(h_star)
        return preds, logits