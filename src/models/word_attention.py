import tensorflow as tf

from src.models.model import SNLIModel
from src.utils.ops import get_embedding, matmul_3d_1d, matmul_3d_2d

class WordAttentionRNNCell(tf.nn.rnn_cell.RNNCell):
    # Custom RNN cell for the word-for-word RNN, based on tutorial
    # at http://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
    def __init__(self, hidden_size, prem_states):
        self._hidden_size = hidden_size
        self._prem_states = prem_states

    @property
    def state_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        return self._hidden_size

    def __call__(self, inputs, state, scope=None):
        """ Run one step of the custom RNN
        Used tensorflow/tensorflow/contrib/rnn/python/ops/rnn_cell.py for guidance
        (can find it on github)

        Args:
          inputs: input Tensor, 2D, batch x num_units.
          state: if `state_is_tuple` is False, this must be a state Tensor,
            `2-D, batch x state_size`.  If `state_is_tuple` is True, this must be a
            tuple of state Tensors, both `2-D`, with column sizes `c_state` and
            `m_state`.
          scope: VariableScope for the created subgraph; defaults to "LSTMCell".
        Returns:
          A tuple containing:
          - A `2-D, [batch x output_dim]`, Tensor representing the output of the
            LSTM after reading `inputs` when previous state was `state`.
            Here output_dim is:
               num_proj if num_proj was set,
               num_units otherwise.
          - Tensor(s) representing the new state of LSTM after reading `inputs` when
            the previous state was `state`.  Same type and shape(s) as `state`.

        """

        with tf.variable_scope(scope or type(self).__name__):  # "WordAttentionRNNCell"
            # Mostly borrowed from attention.py
            W_y = tf.get_variable("W_y", shape=[self._hidden_size, self._hidden_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
            W_h = tf.get_variable("W_h", shape=[self._hidden_size, self._hidden_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
            W_r = tf.get_variable("W_r", shape=[self._hidden_size, self._hidden_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
            w = tf.get_variable("w", shape=[self._hidden_size],
                                initializer=tf.contrib.layers.xavier_initializer())            
            W_t = tf.get_variable("W_t", shape=[self._hidden_size, self._hidden_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
            
            # Matrix-multiply each horizontal slice of prem_states by weight matrix W_y,
            # and use broadcasting to add outer(W_h*h_t + W_r*r_{t-1}, e_L).
            # M.shape => [batch_size, max_len_seq, _hidden_size]
            M = tf.tanh(tf.tensordot(self._prem_states, W_y, axes=1) \
                + tf.expand_dims(tf.matmul(inputs, W_h) + tf.matmul(state, W_r), axis=1))
            M.set_shape([None, None, self._hidden_size])

            # alpha.shape => [batch_size, max_len_seq]
            alpha = tf.nn.softmax(tf.tensordot(M, w, axes=1))

            # r.shape => [batch_size, _hidden_size]
            # We want each row of alpha to be the linear transformation to apply to its
            # corresponding horizontal "slice" of prem_states.
            Y_alpha = tf.reduce_sum(prem_states * tf.expand_dims(alpha, axis=2), axis=1)
            new_h = Y_alpha + tf.tanh(tf.matmul(state, W_r))
        return new_h, new_h


class WordAttentionModel(SNLIModel):
    def __init__(self, embedding_matrix, update_embeddings,
                 hidden_size, use_peepholes,
                 *args, **kwargs):
        super(AttentionModel, self).__init__(use_lens=True, use_dropout=False, *args, **kwargs)
        self._embedding_matrix = embedding_matrix
        self._update_embeddings = update_embeddings
        self._hidden_size = hidden_size
        self._use_peepholes = use_peepholes

    def add_prediction_op(self):
        with tf.variable_scope("prediction"):
            # Premise and hypothesis embedding tensors, both with shape
            # [batch_size, max_len_seq, word_embed_dim]
            prem_embed = get_embedding(self.sentence1_placeholder, self._embedding_matrix,
                                       self._update_embeddings)
            hyp_embed = get_embedding(self.sentence2_placeholder, self._embedding_matrix,
                                      self._update_embeddings, reuse=True)

            # Use LSTMs to run through the entire premise and hypothesis vectors.
            # Initialize the initial hidden state of the hypothesis LSTM using the final premise
            # hidden state.
            cell = tf.contrib.rnn.LSTMCell(self._hidden_size, use_peepholes=self._use_peepholes)
            with tf.variable_scope("prem_encoder"):
                # project embedding matrix to have _hidden_size dimension
                prem_proj = tf.layers.dense(prem_embed, self._hidden_size,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                activation=tf.tanh, name="prem_proj")

                # prem_states.shape => [batch_size, max_len_seq, _hidden_size]
                prem_states, prem_final_state \
                  = tf.nn.dynamic_rnn(cell, prem_proj, dtype=tf.float32,
                                      sequence_length=self.sentence1_lens_placeholder)

            # Consider using a delimiter and running through one LSTM instance as in the paper.
            with tf.variable_scope("hyp_encoder"):
                # hyp_states.shape => [batch_size, max_len_seq, _hidden_size]
                hyp_proj = tf.layers.dense(hyp_embed, self._hidden_size,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                activation=tf.tanh, name="hyp_proj")

                hyp_states, (_, final_hyp_state)\
                      = tf.nn.dynamic_rnn(cell, hyp_proj, initial_state=prem_final_state,
                                          sequence_length=self.sentence2_lens_placeholder)

            with tf.variable_scope("word_attention"):
                attention_cell = WordAttentionRNNCell(self._hidden_size, prem_states)
                attn_states, (_, final_attn_state) \
                      = tf.nn.dynamic_rnn(cell, hyp_states, 
                                          sequence_length=self.sentence2_placeholder)
                
                W_p = tf.get_variable("W_p", shape=[self._hidden_size, self._hidden_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
                W_x = tf.get_variable("W_x", shape=[self._hidden_size, self._hidden_size],
                                      initializer=tf.contrib.layers.xavier_initializer())

                # logits.shape => [batch_size, _hidden_size]
                logits = tf.tanh(tf.matmul(final_attn_state, W_p) + tf.matmul(final_hyp_state, W_x))

                # preds.shape => [batch_size, ]
                preds = tf.argmax(logits, axis=1)
                return preds, logits