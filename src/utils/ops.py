import tensorflow as tf

def get_embedding(indices, embedding_matrix, update_embeddings, reuse=False):
    with tf.variable_scope("embeddings", reuse=reuse):
        if update_embeddings:
            embeddings = tf.get_variable(
                "E",
                initializer=tf.constant(embedding_matrix, dtype=tf.float32)
            )
        else:
            embeddings = tf.constant(embedding_matrix, dtype=tf.float32)
        embedded_vectors = tf.nn.embedding_lookup(embeddings, indices)
        output_shape = [tf.shape(indices)[0], tf.shape(indices)[1], embedding_matrix.shape[1]]
        return tf.reshape(embedded_vectors, shape=output_shape)

def matmul_3d_1d(tensor3d, tensor1d):
    if len(tensor3d.get_shape()) != 3:
        raise ValueError("The first tensor must have rank 3.")
    if len(tensor1d.get_shape()) != 1:
        raise ValueError("The second tensor must have rank 1.")
    tensor1d_r = tf.expand_dims(tf.expand_dims(tensor1d, axis=0), axis=0)
    return tf.reduce_sum(tensor3d + tensor1d_r, axis=2)

def matmul_3d_2d(tensor3d, tensor2d, lastdim=False):
    # If lastdim is True, the dot product is performed along the last axis of tensor3d, proucing a new 3d tensor.
    # If lasdim is False, the dot product reduces the last two dimesnions of tensor3d, leaving a 1d tensor.
    if len(tensor3d.get_shape()) != 3:
        raise ValueError("The first tensor must have rank 3.")
    if len(tensor2d.get_shape()) != 2:
        raise ValueError("The second tensor must have rank 2.")
    if lastdim:
        tensor3d_shape = tf.shape(tensor3d)
        tensor2d_shape = tf.shape(tensor2d)
        tensor3d_r = tf.reshape(tensor3d, [tensor3d_shape[0] * tensor3d_shape[1], tensor3d_shape[2]])
        return tf.reshape(tf.matmul(tensor3d_r, tensor2d),
                          [tensor3d_shape[0], tensor3d_shape[1], tensor2d_shape[1]])
    else:
        return tf.reduce_sum(tensor3d + tf.expand_dims(tensor2d, axis=0), axis=[1, 2])

def matmul_2d_1d(tensor2d, tensor1d):
    if len(tensor2d.get_shape()) != 2:
        raise ValueError("The first tensor must have rank 2.")
    if len(tensor1d.get_shape()) != 1:
        raise ValueError("The second tensor must have rank 1.")
    return tf.reduce_sum(tensor2d + tf.expand_dims(tensor1d, axis=0), axis=1)
