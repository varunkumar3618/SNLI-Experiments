import tensorflow as tf
import numpy as np

def get_embeddings(embedding_matrix, embedding_mode, missing_indices):
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

        return original_embeddings * (1 - missing) + new_embeddings * missing

def cosine(x, y):
    x_mag = tf.sqrt(tf.tensordot(x, x, axes=1))
    y_mag = tf.sqrt(tf.tensordot(y, y, axes=1))
    return tf.tensordot(x, y, axes=1) / (X_mag * y_mag)

def masked_sequence_softmax(logits, sequence_lens):
    logits = logits - tf.reduce_max(logits, axis=1, keep_dims=True)

    mask = tf.sequence_mask(sequence_lens, tf.shape(logits)[1], dtype=logits.dtype)
    mask = tf.expand_dims(mask, axis=2)

    unnorm = tf.exp(logits) * mask
    return unnorm / tf.reduce_sum(unnorm, axis=1, keep_dims=True)

def add_null_vector(seq):
    zeros = tf.zeros([tf.shape(seq)[0], 1, tf.shape(seq)[2]])
    return tf.concat([zeros, seq], axis=1)
