import tensorflow as tf

def get_embedding(indices, embedding_matrix, update_embeddings, reuse=False):
    with tf.variable_scope("embeddings", reuse=reuse):
        if update_embeddings:
            embeddings = tf.get_variable(
                "E",
                initializer=tf.constant(embedding_matrix, dtype=tf.float32)
            )
        else:
            embeddings = embedding_matrix
        embedded_vectors = tf.nn.embedding_lookup(embeddings, indices)
        output_shape = [tf.shape(indices)[0], tf.shape(indices)[1], embedding_matrix.shape[1]]
        return tf.reshape(embedded_vectors, shape=output_shape)
