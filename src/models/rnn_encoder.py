import tensorflow as tf

from src.models.sow import SumOfWords


class RNNEncoder(SumOfWords):

    def __init__(self, use_peepholes, use_lens=True,
                 *args, **kwargs):
        super(RNNEncoder, self).__init__(use_lens=use_lens, *args, **kwargs)
        self._use_peepholes = use_peepholes
        
    def encoding(self, prem_proj, hyp_proj):
        with tf.variable_scope("encoding"):
            cell = tf.contrib.rnn.LSTMCell(
                self._hidden_size / 2,
                use_peepholes=self._use_peepholes,
                initializer=self.rec_init
            )
            cell = self.apply_dropout_wrapper(cell)
            with tf.variable_scope("prem"):
                # Harcoded to use an LSTM
                _, (_, prem_encoded) = tf.nn.dynamic_rnn(cell, prem_proj, dtype=tf.float32,
                                                         sequence_length=self.sentence1_lens_placeholder)
            with tf.variable_scope("hyp"):
                # Harcoded to use an LSTM
                _, (_, hyp_encoded) = tf.nn.dynamic_rnn(cell, hyp_proj, dtype=tf.float32,
                                                        sequence_length=self.sentence2_lens_placeholder)
            both_encoded = tf.concat([prem_encoded, hyp_encoded], axis=1)
            both_encoded = self.apply_dropout(both_encoded)

        return both_encoded
