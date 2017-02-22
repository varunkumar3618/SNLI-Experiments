import tensorflow as tf


def apply_activation(x, activation):
    if activation == "tanh":
        return tf.tanh(x)
    else:
        raise ValueError("Unsupported activation: %s" % activation)

def mean_ce_with_logits(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

def make_train_op(config, loss):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    if config.train.train_op == "AdaDelta":
        opt = tf.train.AdadeltaOptimizer(learning_rate=config.train.learning_rate,
            epsilon=config.train.epsilon, rho=config.train.rho)
    return opt.minimize(loss, global_step=global_step)
