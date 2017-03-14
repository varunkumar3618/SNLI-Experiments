import tensorflow as tf
import numpy as np

def get_embeddings(embedding_matrix_np, embedding_mode, missing_indices):
    with tf.variable_scope("embeddings"):
        if embedding_mode == "all":
            original_embeddings = tf.get_variable(
                "E1",
                initializer=tf.constant(embedding_matrix, dtype=tf.float32)
            )
        else:
            original_embeddings = embedding_matrix
        if embedding_mode == "all" or embedding_mode == "unseen":
            new_embeddings = tf.get_variable(
                "E2",
                initializer=tf.random_normal_initializer(stddev=0.1),
                shape=embedding_matrix.shape
            )
        elif embedding_mode == "none":
            new_embeddings = tf.zeros_like(embedding_matrix)
        else:
            raise ValueError("Unsupported embedding mode: %s." % embedding_mode)
        missing_one_hot = np.zeros(len(embedding_matrix))
        missing_one_hot[missing_indices] = 1
        missing = tf.expand_dims(tf.constant(missing_one_hot, dtype=tf.float32), axis=1)
        print tf.constant(missing_one_hot).shape, missing.shape
        raise ValueError

        return original_embeddings * (1 - missing) + new_embeddings * missing

def embed_indices(indices, embeddings):
    embedded_vectors = tf.nn.embedding_lookup(embeddings, indices)
    output_shape = [tf.shape(indices)[0], tf.shape(indices)[1], embedding_matrix.shape[1]]
    return tf.reshape(embedded_vectors, shape=output_shape)
