import tensorflow as tf
import numpy as np
from test_utils import test_all_close

def get_embedding(indices, embedding_matrix, update_embeddings, 
                  train_unseen_vocab=False, missing_indices=None, reuse=False):
    with tf.variable_scope("embeddings", reuse=reuse):
        if train_unseen_vocab:
            fixed_embedding = tf.constant(embedding_matrix, dtype=tf.float32)
            trainable_embedding = tf.get_variable("new_embed", 
                                        initializer=tf.contrib.layers.xavier_initializer(), 
                                        shape=embedding_matrix.shape)
            
            print "Embedding shape and len", embedding_matrix.shape, len(embedding_matrix)
            missing_one_hot = np.zeros(len(embedding_matrix))
            missing_one_hot[missing_indices] = 1
            missing = tf.expand_dims(tf.constant(missing_one_hot, dtype=tf.float32), axis=1)
            embeddings = fixed_embedding + tf.multiply(missing, trainable_embedding)
        elif update_embeddings:
            embeddings = tf.get_variable(
                "E",
                initializer=tf.constant(embedding_matrix, dtype=tf.float32)
            )
        else:
            embeddings = tf.constant(embedding_matrix, dtype=tf.float32)
        embedded_vectors = tf.nn.embedding_lookup(embeddings, indices)
        output_shape = [tf.shape(indices)[0], tf.shape(indices)[1], embedding_matrix.shape[1]]
        return tf.reshape(embedded_vectors, shape=output_shape)

"""
NOTE: These functions are deprecated in favor of the tf.tensordot approach below.

def matmul_3d_1d(tensor3d, tensor1d):
    if len(tensor3d.get_shape()) != 3:
        raise ValueError("The first tensor must have rank 3.")
    if len(tensor1d.get_shape()) != 1:
        raise ValueError("The second tensor must have rank 1.")
    tensor1d_r = tf.expand_dims(tf.expand_dims(tensor1d, axis=0), axis=0)
    return tf.reduce_sum(tensor3d + tensor1d_r, axis=2)

def matmul_3d_2d(tensor3d, tensor2d, lastdim=False):
    # If lastdim is True, the dot product is performed along the last axis of tensor3d, producing a new 3d tensor.
    # If lastdim is False, the dot product reduces the last two dimesnions of tensor3d, leaving a 1d tensor.
    if len(tensor3d.get_shape()) != 3:
        raise ValueError("The first tensor must have rank 3.")
    if len(tensor1d.get_shape()) != 2:
        raise ValueError("The second tensor must have rank 2.")
    if lastdim:
        tensor3d_shape = tf.shape(tensor3d)
        tensor2d_shape = tf.shape(tensor2d)
        tensor3d_r = tf.reshape(tensor3d, [tensor3d_shape[0] * tensor3d_shape[1], tensor3d_shape[2]])
        return tf.reshape(tf.matmul(tensor3d_r, tensor2d),
                          [tensor3d_shape[0], tensor3d_shape[1], tensor2d_shape[1]])
    else:
        return tf.reduce_sum(tensor3d + tf.expand_dims(tensor2d, axis=0), axis=[1, 2])
"""

def matmul_3d_1d(tensor3d, tensor1d):
    """Returns the resulting 2d tensor obtained when performing a batch matrix multiplication
    between a 3d and 1d tensor, where we assume the left hand operand is the 1d tensor, and
    we perform matrix multiplication along the last dimension of the 3d tensor.

    See tensordot doc: https://www.tensorflow.org/versions/master/api_docs/python/math_ops/tensor_math_function
    """
    if len(tensor3d.get_shape()) != 3:
        raise ValueError("The first tensor must have rank 3.")
    if len(tensor1d.get_shape()) != 1:
        raise ValueError("The second tensor must have rank 1.")
    if tensor3d.shape[2] != tensor1d.shape[0]:
        raise ValueError("The last dimension of the 3d tensor must equal the dimension of the 1d tensor.")
    return tf.tensordot(tensor3d, tensor1d, axes=1)


def matmul_3d_2d(tensor3d, tensor2d):
    """Returns the resulting 3d tensor obtained when performing a batch matrix multiplication
    between a 3d and 2d tensor, where we assume the left hand operand is the 2d tensor, and
    we perform matrix multiplication along the last dimension of the 3d tensor.

    See tensordot doc: https://www.tensorflow.org/versions/master/api_docs/python/math_ops/tensor_math_function
    """
    if len(tensor3d.get_shape()) != 3:
        raise ValueError("The first tensor must have rank 3.")
    if len(tensor2d.get_shape()) != 2:
        raise ValueError("The second tensor must have rank 2.")
    if tensor3d.shape[2] != tensor2d.shape[1]:
        raise ValueError("The last dimension of the 3d tensor must equal the last dimension of the 2d tensor.")
    return tf.tensordot(tensor3d, tensor2d, axes=[[2], [1]])


def matmul_2d_1d(tensor2d, tensor1d):
    if len(tensor2d.get_shape()) != 2:
        raise ValueError("The first tensor must have rank 2.")
    if len(tensor1d.get_shape()) != 1:
        raise ValueError("The second tensor must have rank 1.")
    return tf.reduce_sum(tensor2d + tf.expand_dims(tensor1d, axis=0), axis=0)


# Sanity checks

def test_matmul_3d_1d():
    # Conceptually: (1,2) and (3,4) are the hidden state of the words in the first hypothesis training instance.
    # We want to the matmul of this slice by the 2d transformation matrix.
    # 
    # For one hypothesis sentence:
    # [1001 x 1002]     [ 1  3        [ 3005 7011 ]
    #                x    2  4 ]   = 
    matrix_1d =tf.constant(np.array([1001, 1002]), dtype=tf.float32)
    matrix_3d = tf.constant(np.array([[[1, 2], [3, 4]],
                                               [[5, 6], [7, 8]],
                                               [[9, 10], [11, 12]]]), dtype=tf.float32)
    test1 = matmul_3d_1d(matrix_3d, matrix_1d)
    with tf.Session() as sess:
        test1 = sess.run(test1)
    test_all_close("Matmul_3d_1d test 1", test1, np.array([[3005, 7011],
                                                           [11017, 15023],
                                                           [19029, 23035]]))
    print "Basic (non-exhaustive) batch matmul 1d x 3d tests pass\n"


def test_matmul_3d_2d():
    # Conceptually: h1=(1,2) and h2=(3,4) are the hidden state of the words in the first hypothesis training instance.
    # We want to matmul this slice by the 2d transformation matrix.
    # 
    # For one hypothesis sentence: TODO: ? is this right?
    # [ 1001  1002      [ 1  3      [ 3005 7011
    #       3    4 ] x    2  4 ]  =   11   25 ]

    matrix_2d = tf.constant(np.array([[1001, 1002], [3, 4]]), dtype=tf.float32)
    matrix_3d = tf.constant(np.array([[[1, 2], [3, 4]],
                                       [[5, 6], [7, 8]],
                                       [[9, 10], [11, 12]]]), dtype=tf.float32)
    test1 = matmul_3d_2d(matrix_3d, matrix_2d)
    with tf.Session() as sess:
        test1 = sess.run(test1)
    test_all_close("Matmul_3d_2d test 1", test1, np.array([[[3005, 11], [7011, 25]],
                                                           [[11017, 39], [15023, 53]],
                                                           [[19029, 67], [23035, 81]]]))
    print "Basic (non-exhaustive) batch matmul 2d x 3d tests pass\n"


if __name__ == "__main__":
    test_matmul_3d_2d()
    test_matmul_3d_1d()
