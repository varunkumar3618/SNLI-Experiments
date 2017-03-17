import tensorflow as tf

from src.models.attention import AttentionModel

""" The mLSTM model is based on the paper by Cheng and Jiang. Since the basic RNNCell
format does not allow access to the previous h (just the old state, which corresponds to
cell in an LSTM), the state passed at each timestep is actually a concatentation of the 
cell and the hidden output. """
class mLSTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, hidden_size, subject, initializer, regularizer):
        self._state_size = 2*hidden_size
        self._hidden_size = hidden_size
        self._subject = subject
        self._initializer = initializer
        self._regularizer = regularizer

        # A hack to make regularization work
        with tf.variable_scope(type(self).__name__) as scope:
            self._scope = scope
            _ = self(
                tf.zeros_like(subject),
                tf.zeros(shape=[tf.shape(subject)[0], self._hidden_size], dtype=subject.dtype)
            )
            self._scope.reuse_variables()
            
    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):
        scope = scope or type(self).__name__
        with tf.variable_scope(scope):
            # State is [batch_size X 2 * hidden_size], the concatenated cell and last hidden output
            cell, hidden = tf.split(state, num_or_size_splits=2, axis=1)
            # W_y * Y
            M_prem = tf.layers.dense(self._subject, self._hidden_size,
                                     kernel_initializer=self._initializer,
                                     kernel_regularizer=self._regularizer,
                                     name="M_prem")

            M_in = tf.layers.dense(inputs, self._hidden_size,
                                     kernel_initializer=self._initializer,
                                     kernel_regularizer=self._regularizer,
                                     name="M_input")

            M_state = tf.layers.dense(hidden, self._hidden_size,
                                     kernel_initializer=self._initializer,
                                     kernel_regularizer=self._regularizer,
                                     name="M_state")
            M = tf.tanh(M_prem + tf.expand_dims(M_in, axis=1) + tf.expand_dims(M_state, axis=1))

            A = tf.layers.dense(M, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=self._regularizer, name="A")
            A = tf.squeeze(A, axis=2)
            alpha = tf.nn.softmax(A)

            r_subject = tf.reduce_sum(self._subject * tf.expand_dims(alpha, axis=2), axis=1)

            # r_subject is a_k, the attention weighted premise vectors, in the Chen and Jiang paper
            # Now, implement an LSTM, where the input to the LSTM is m_k = [a_k, inputs]
            m_k = tf.concat([r_subject, inputs], axis=1, name="m_k")
            full_inputs = tf.concat([m_k, hidden], axis=1, name="full_inputs")
            input_gate = tf.layers.dense(full_inputs, self._hidden_size, 
                                       kernel_initializer=self._initializer, 
                                       kernel_regularizer=self._regularizer, 
                                       activation=tf.sigmoid, name="input_gate")

            forget_gate = tf.layers.dense(full_inputs, self._hidden_size, 
                                       kernel_initializer=self._initializer, 
                                       kernel_regularizer=self._regularizer, 
                                       activation=tf.sigmoid, name="forget_gate")

            output_gate = tf.layers.dense(full_inputs, self._hidden_size, 
                                       kernel_initializer=self._initializer, 
                                       kernel_regularizer=self._regularizer, 
                                       activation=tf.sigmoid, name="output_gate")

            intermediate = tf.layers.dense(full_inputs, self._hidden_size, 
                                       kernel_initializer=self._initializer, 
                                       kernel_regularizer=self._regularizer, 
                                       activation=tf.tanh, name="intermediate")
           
            new_cell = tf.multiply(forget_gate, cell) + tf.multiply(input_gate, intermediate)
            new_hidden = tf.multiply(output_gate, tf.tanh(new_cell))
            new_state = tf.concat([new_cell, new_hidden], axis=1, name="new_state")

        return new_state, new_state

class mLSTMModel(AttentionModel):
    def _attention(self, prem_hiddens, prem_final_state, hyp_hiddens, hyp_final_state):
        reg = tf.contrib.layers.l2_regularizer(self._l2_reg)
        hyp_final_hidden = hyp_final_state[1]
        with tf.variable_scope("attention"):
            att_cell = mLSTMCell(
                self._hidden_size,
                prem_hiddens,
                tf.contrib.layers.xavier_initializer(),
                reg
            )
            with tf.variable_scope("attention"):
                _, final_output = tf.nn.dynamic_rnn(att_cell, hyp_hiddens, dtype=tf.float32,
                                               sequence_length=self.sentence2_lens_placeholder)
                _, final_hidden = tf.split(final_output, num_or_size_splits=2, axis=1)
                h_star = tf.layers.dense(final_hidden,
                                         self._hidden_size,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         kernel_regularizer=reg,
                                         activation=tf.tanh,
                                         name="h_star")
        return h_star
