import os

import tensorflow as tf
import numpy as np

from src.models.sow import SumOfWords
from src.models.rnn_encoder import RNNEncoder
from src.models.attention import AttentionModel
from src.utils.dataset import Dataset
from src.utils.vocab import Vocab
from src.utils.wvecs import get_glove_vectors
from src.utils.progbar import Progbar

flags = tf.app.flags
flags.DEFINE_string("model", "SOW", "The type of model to train.")

# File paths
flags.DEFINE_string("data_dir", "data/", "The location of the data files.")
flags.DEFINE_string("checkpoint_subdir", "model/", "The checkpoint subdirectory inside data_dir")
flags.DEFINE_string("glove_type", "common", "The source of the Glove word vectors used: one of 'wiki' and 'common'")

# Data
flags.DEFINE_integer("max_vocab_size", 10000, "The maximum size of the vocabulary.")
flags.DEFINE_integer("max_seq_len", 100, "The maximum length of a sentence. Sentences longer than this will be truncated.")

# Model
flags.DEFINE_integer("word_embed_dim", 300, "The dimension of the embedding matrix.")
flags.DEFINE_integer("hidden_size", 200, "The size of the hidden layer, applicable to some models.")
flags.DEFINE_boolean("update_embeddings", False, "Whether the word vectors should be updated")
flags.DEFINE_boolean("use_peepholes", True, "Whether to use peephole connections, applicable to LSTM models.")
flags.DEFINE_float("dropout_rate", 0.15, "How many units to eliminate during training, applicable to models using dropout.")

# Training
flags.DEFINE_integer("batch_size", 100, "The batch size.")
flags.DEFINE_integer("num_epochs", 50, "The numer of epochs to train for.")
flags.DEFINE_float("l2_reg", 1e-4, "The level of l2 regularization to use.")

flags.DEFINE_boolean("debug", False, "Whether to run in debug mode, i.e. use a smaller dataset and increase verbosity.")
flags.DEFINE_boolean("train", True, "Whether to train or test the model.")
flags.DEFINE_boolean("save", True, "Whether to save the model periodically")

FLAGS = flags.FLAGS

snli_dir = os.path.join(FLAGS.data_dir, "snli_1.0")
vocab_file = os.path.join(FLAGS.data_dir, "vocab.txt")
data_file = os.path.join(FLAGS.data_dir, "data.pkl")
checkpoint_dir = os.path.join(FLAGS.data_dir, FLAGS.checkpoint_subdir)

if FLAGS.glove_type == "wiki":
    glove_file = os.path.join(os.path.join(FLAGS.data_dir, "glove.6B"), "glove.6B.%sd.txt" % FLAGS.word_embed_dim)
elif FLAGS.glove_type == "common":
    if FLAGS.word_embed_dim != 300:
        raise ValueError("Common Crawl word vectors are only available with dimension 300.")
    glove_file = os.path.join(os.path.join(FLAGS.data_dir, "glove.840B.300d"), "glove.840B.300d.txt")
else:
    raise ValueError("Unrecognized word vector type: %s." % FLAGS.glove_type)


def run_train_epoch(sess, model, dataset, epoch_num):
    print "="*79
    print "Epoch: %s" % (epoch_num + 1)
    prog = Progbar(target=dataset.split_num_batches("train", FLAGS.batch_size))
    for i, batch in enumerate(dataset.get_shuffled_iterator("train", FLAGS.batch_size)):
        loss = model.train_on_batch(sess, *batch)
        prog.update(i + 1, [("train loss", loss)])
    print "="*79

def run_eval_epoch(sess, model, dataset, split):
    batch_sizes = []
    accuracies = []

    print "-"*79
    print "Evaluating on %s." % split
    prog = Progbar(target=dataset.split_num_batches(split, FLAGS.batch_size))
    for i, batch in enumerate(dataset.get_iterator(split, FLAGS.batch_size)):
        acc, loss = model.evaluate_on_batch(sess, *batch)
        prog.update(i + 1, [("%s loss" % split, loss)])

        batch_sizes.append(batch[0].shape[0])
        accuracies.append(acc)

    accuracy = np.average(accuracies, weights=batch_sizes)
    print "Accuracy: %s" % accuracy
    print "-"*79
    return accuracy

def main(_):
    if not os.path.exists('./data/model/'):
        os.makedirs('./data/model/')

    with tf.Graph().as_default():
        vocab = Vocab(snli_dir, vocab_file, FLAGS.max_vocab_size)
        dataset = Dataset(snli_dir, data_file, vocab, FLAGS.max_seq_len, debug=FLAGS.debug)
        embedding_matrix = get_glove_vectors(glove_file, FLAGS.word_embed_dim, vocab)

        if FLAGS.model == "SOW":
            model = SumOfWords(
                embedding_matrix=embedding_matrix,
                update_embeddings=FLAGS.update_embeddings,
                hidden_size=FLAGS.hidden_size,
                l2_reg=FLAGS.l2_reg,
                max_seq_len=FLAGS.max_seq_len,
                dropout_rate=FLAGS.dropout_rate
            )
        elif FLAGS.model == "RNN_Encoder":
            model = RNNEncoder(
                embedding_matrix=embedding_matrix,
                update_embeddings=FLAGS.update_embeddings,
                hidden_size=FLAGS.hidden_size,
                l2_reg=FLAGS.l2_reg,
                max_seq_len=FLAGS.max_seq_len,
                dropout_rate=FLAGS.dropout_rate,
                use_peepholes=FLAGS.use_peepholes
            )
        elif FLAGS.model == "Attention":
            model = AttentionModel(
                embedding_matrix=embedding_matrix,
                update_embeddings=FLAGS.update_embeddings,
                hidden_size=FLAGS.hidden_size,
                use_peepholes=FLAGS.use_peepholes,
                max_seq_len=FLAGS.max_seq_len
            )
        else:
            raise ValueError("Unrecognized model: %s." % FLAGS.model)
        model.build()

        saver = None
        if FLAGS.save:
            saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            if FLAGS.train:
                best_accuracy = 0
                for epoch in range(FLAGS.num_epochs):
                    run_train_epoch(sess, model, dataset, epoch)
                    accuracy = run_eval_epoch(sess, model, dataset, "train" if FLAGS.debug else "dev")

                    if accuracy > best_accuracy and FLAGS.save:
                        saver.save(sess, checkpoint_dir + 'best_model_' + FLAGS.model,
                           global_step=epoch+1)
            else:
                raise ValueError("Cannot test the model just yet.")

if __name__ == "__main__":
    tf.app.run()