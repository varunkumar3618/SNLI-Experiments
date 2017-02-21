import tensorflow as tf


def apply_activation(x, activation):
    if activation == "tanh":
        return tf.tanh(x)
    else:
        raise ValueError("Unsupported activation: %s" % activation)


def linear(x, output_size, activation=None, scope=None, reuse=None, use_bias=True, w_init=None, b_init=None):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        W = tf.get_variable("W", shape=[tf.shape(
            x)[-1], output_size], initializer=w_init)
        output = tf.matmul(x, W)
        if use_bias:
            b = tf.get_variable("b", shape=[output_size], initializer=b_init)
            output = output + b
        elif b_init is not None:
            raise ValueError("The bias is being initialized but not used.")

        if activation:
            output = apply_activation(output, activation)
    return output


def train_op(config, learning_rate, loss):
    if config["name"] == "AdaDelta":
        opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate,
            epsilon=config["epsilon"], rho=config["rho"])