import tensorflow as tf

from src.models.model import SNLIModel
from src.utils.config import ModelConfig


class SumOfWords(SNLIModel):
    def get_embedding(self, indices, reuse=False):
        with tf.variable_scope("embeddings", reuse=reuse):
            embeddings = tf.get_variable(
                "E",
                initializer=tf.constant(self.embedding_matrix, dtype=tf.float32)
            )
            embedded_vectors = tf.nn.embedding_lookup(embeddings, indices)
            output_shape = [tf.shape(indices)[0], tf.shape(indices)[1], self.embedding_matrix.shape[1]]
            embeddings = tf.reshape(embedded_vectors, shape=output_shape)
        return embeddings

    def build_model(self, config, sentence1, sentence2):
        prem_embed = self.get_embedding(sentence1)
        hyp_embed = self.get_embedding(sentence2, reuse=True)

        prem_sow = tf.reduce_sum(prem_embed, axis=1)
        hyp_sow = tf.reduce_sum(hyp_embed, axis=1)
        both_sow = tf.concat([prem_sow, hyp_sow], axis=1)

        h1 = tf.layers.dense(both_sow, config.model.hidden_size, activation=tf.tanh, name="h1")
        h2 = tf.layers.dense(h1, config.model.hidden_size, activation=tf.tanh, name="h2")
        h3 = tf.layers.dense(h2, config.model.hidden_size, activation=tf.tanh, name="h3")
        logits = tf.layers.dense(h3, config.model.hidden_size, name="logits")

        return logits

class SumOfWordsConfig(ModelConfig):
    def __init__(self, *args, **kwargs):
        super(SumOfWordsConfig, self).__init__(*args, **kwargs)
        config_dict = self.config_dict
        self.hidden_size = config_dict["hidden_size"]