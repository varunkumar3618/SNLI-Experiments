import tensorflow as tf

from src.models.model import SNLIModel

def get_bilstm_finals(state):
    fw_state, bw_state = state
    return fw_state[1], bw_state[1]

class MPMStackedModel(SNLIModel):
    def __init__(self,
                 hidden_size, use_peepholes, perspectives,
                 l2_reg, use_lens=True, use_dropout=True,
                 *args, **kwargs):
        super(MPMStackedodel, self).__init__(use_lens=use_lens, use_dropout=use_dropout,
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

        def cosine_single(x, y, axis):
            x = tf.nn.l2_normalize(x, axis)
            y = tf.nn.l2_normalize(y, axis)
            return tf.reduce_sum(x * y, axis=axis)

        def cosine_complex(x, y, axis, formula):
            x = tf.nn.l2_normalize(x, axis)
            y = tf.nn.l2_normalize(y, axis)
            return tf.einsum(formula, x, y)

        def cosine_matching_single(x, y, scope):
            with tf.variable_scope(scope):
                W = tf.get_variable("W", shape=[self._perspectives, self._hidden_size],
                                    initializer=self.dense_init,
                                    regularizer=reg)
                W = tf.expand_dims(tf.expand_dims(W, axis=0), axis=1)

                x_pers = tf.expand_dims(x, axis=2) * W
                y_pers =tf.expand_dims(y, axis=2) * W

                match = cosine_single(x_pers, y_pers, axis=3)

            return match

        def cosine_matching_all(x, y, scope):
            with tf.variable_scope(scope):
                W = tf.get_variable("W", shape=[self._perspectives, self._hidden_size],
                                    initializer=self.dense_init,
                                    regularizer=reg)

                x_pers = tf.expand_dims(x, axis=2) * W
                y_pers = tf.expand_dims(y, axis=2) * W

                match = cosine_complex(x_pers, y_pers, 3, "aikl,ajkl->aijk")
            return match

        def all_matchings(prem_hiddens, prem_final, hyp_hiddens, hyp_final, scope):
            with tf.variable_scope(scope):
                # Full matching
                prem_full_match = cosine_matching_single(prem_hiddens, hyp_final, "prem_full")
                hyp_full_match = cosine_matching_single(hyp_hiddens, prem_final, "hyp_full")

                # Max-pool matching
                all_max_match = cosine_matching_all(prem_hiddens, hyp_hiddens, "all_max")
                prem_max_match = tf.reduce_max(all_max_match, axis=2)
                hyp_max_match = tf.reduce_max(all_max_match, axis=1)

                # Attentive matching
                alpha = cosine_complex(prem_hiddens, hyp_hiddens, 2, "aik,ajk->aij")
                prem_weights = tf.nn.softmax(alpha, dim=2)
                hyp_weights = tf.nn.softmax(alpha, dim=1)

                prem_means = tf.reduce_sum(tf.expand_dims(hyp_hiddens, axis=1) * tf.expand_dims(prem_weights, 3), axis=2)
                prem_att_match = cosine_matching_single(prem_hiddens, prem_means, "prem_att")

                hyp_means = tf.reduce_sum(tf.expand_dims(prem_hiddens, axis=2) * tf.expand_dims(hyp_weights, 3), axis=1)
                hyp_att_match = cosine_matching_single(hyp_hiddens, hyp_means, "hyp_att")

            prem_matches = [
                prem_full_match,
                prem_max_match
            ]
            hyp_matches = [
                hyp_full_match,
                hyp_max_match
            ]
            return prem_matches, hyp_matches


        with tf.variable_scope(scope):
            prem_fw_matches, hyp_fw_matches = all_matchings(prem_fw_hiddens, prem_fw_final,
                                                            hyp_fw_hiddens, hyp_fw_final,
                                                            "forward")
            prem_bw_matches, hyp_bw_matches = all_matchings(prem_bw_hiddens, prem_bw_final,
                                                            hyp_bw_hiddens, hyp_bw_final,
                                                            "backward")
            prem_matches = prem_fw_matches + prem_bw_matches
            hyp_matches = hyp_fw_matches + hyp_bw_matches
            prem_matched = tf.concat(prem_matches, axis=2)
            hyp_matched = tf.concat(hyp_matches, axis=2)

            prem_matched = self.apply_dropout(prem_matched)
            hyp_matched = self.apply_dropout(hyp_matched)

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

            prem_hiddens, prem_final_state, hyp_hiddens, hyp_final_state = self.encoding(prem_embed, hyp_embed, "encoding1")
            prem_matched, hyp_matched = self.matching(prem_hiddens, prem_final_state, hyp_hiddens, hyp_final_state, "matching1")

            prem_hiddens, prem_final_state, hyp_hiddens, hyp_final_state = self.encoding(prem_matched, hyp_matched, "encoding2")
            prem_matched, hyp_matched = self.matching(prem_hiddens, prem_final_state, hyp_hiddens, hyp_final_state, "matching2")

            _, prem_final_state, _,hyp_final_state = self.encoding(prem_matched, hyp_matched, "composition")
            h_star = tf.concat(
                list(get_bilstm_finals(prem_final_state)) + list(get_bilstm_finals(hyp_final_state)),
                axis=1
            )
            preds, logits = self.classification(h_star)
        return preds, logits
