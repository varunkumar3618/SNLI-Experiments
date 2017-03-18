import tensorflow as tf

from src.models.model import SNLIModel

class AttentionModel(SNLIModel):
    def __init__(self,
                 hidden_size, use_peepholes,
                 l2_reg,
                 use_lens=True, use_dropout=True,
                 *args, **kwargs):
        super(AttentionModel, self).__init__(use_lens=use_lens, use_dropout=use_dropout,
                                             *args, **kwargs)
        self._hidden_size = hidden_size
        self._use_peepholes = use_peepholes
        self._l2_reg = l2_reg
        
    def embedding(self):
        reg = tf.contrib.layers.l2_regularizer(self._l2_reg)
        with tf.variable_scope("embedding"):
            prem_embed = self.embed_indices(self.sentence1_placeholder)
            hyp_embed = self.embed_indices(self.sentence2_placeholder)

            prem_proj = tf.layers.dense(prem_embed, self._hidden_size,
                                        kernel_initializer=self.dense_init,
                                        kernel_regularizer=reg,
                                        activation=self.activation, name="prem_proj")
            hyp_proj = tf.layers.dense(hyp_embed, self._hidden_size,
                                       kernel_initializer=self.dense_init,
                                       kernel_regularizer=reg,
                                       activation=self.activation, name="hyp_proj")
            prem_proj = self.apply_dropout(prem_proj)
            hyp_proj = self.apply_dropout(hyp_proj)
        return prem_proj, hyp_proj

    def encoding(self, prem_proj, hyp_proj):
        with tf.variable_scope("encoding"):
            cell = tf.contrib.rnn.LSTMCell(
                self._hidden_size,
                use_peepholes=self._use_peepholes,
                initializer=self.rec_init
            )
            cell = self.apply_dropout_wrapper(cell)
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

    def attention(self, prem_hiddens, prem_final_state, hyp_hiddens, hyp_final_state):
        reg = tf.contrib.layers.l2_regularizer(self._l2_reg)
        hyp_final_hidden = hyp_final_state[1]
        with tf.variable_scope("attention"):
            zeros = tf.zeros([tf.shape(prem_hiddens)[0], 1, self._hidden_size])
            subject = tf.concat([zeros, prem_hiddens], axis=1)

            M_prem = tf.layers.dense(subject, self._hidden_size,
                                     kernel_initializer=self.dense_init,
                                     kernel_regularizer=reg,
                                     name="M_prem")
            M_hyp_final = tf.layers.dense(hyp_final_hidden, self._hidden_size,
                                          kernel_initializer=self.dense_init,
                                          kernel_regularizer=reg,
                                          name="M_hyp_final")
            M = self.activation(M_prem + tf.expand_dims(M_hyp_final, axis=1))

            A = tf.layers.dense(M, 1, kernel_initializer=self.dense_init,
                                kernel_regularizer=reg, name="A")
            A = tf.squeeze(A, axis=2)
            alpha = tf.nn.softmax(A)

            np.save(alpha, sess.)
            r = tf.reduce_sum(subject * tf.expand_dims(alpha, axis=2), axis=1)

            h_star = tf.layers.dense(tf.concat([r, hyp_final_hidden], 1), self._hidden_size,
                                     kernel_initializer=self.dense_init,
                                     kernel_regularizer=reg,
                                     activation=self.activation,
                                     name="h_star")
        return h_star

    def classification(self, h_star):
        reg = tf.contrib.layers.l2_regularizer(self._l2_reg)
        with tf.variable_scope("classification"):
            h_star = self.apply_dropout(h_star)
            logits = tf.layers.dense(h_star, 3,
                                     kernel_initializer=self.dense_init,
                                     kernel_regularizer=reg,
                                     name="logits")
            preds = tf.argmax(logits, axis=1)
            return preds, logits

    def add_prediction_op(self):
        with tf.variable_scope("prediction"):
            prem_proj, hyp_proj = self.embedding()
            prem_hiddens, prem_final_state, hyp_hiddens, hyp_final_state\
                = self.encoding(prem_proj, hyp_proj)
            h_star = self.attention(prem_hiddens, prem_final_state, hyp_hiddens, hyp_final_state)
            preds, logits = self.classification(h_star)
        return preds, logits
