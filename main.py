import tensorflow as tf

from src.models.sow import SumOfWords
from src.utils.dataset import Dataset
from src.utils.vocab import Vocab
from src.utils.wvecs import get_glove_vectors
from src.utils.progbar import Progbar

flags = tf.app.flags
flags.DEFINE_string("model", "SOW", "The type of model to train.")

# File paths
flags.DEFINE_string("snli_dir", "data/snli_1.0", "The location of the SNLI dataset.")
flags.DEFINE_string("glove_dir", "data/glove.6B", "The location of the Glove word embeddings.")
flags.DEFINE_string("vocab_file", "data/vocab.txt", "The location of the vocabulary file.")
flags.DEFINE_string("checkpoint_dir", "data/model", "Where to save the model.")

# Data
flags.DEFINE_integer("max_vocab_size", 10000, "The maximum size of the vocabulary.")
flags.DEFINE_integer("max_seq_len", 100, "The maximum length of a sentence. Sentences longer than this will be truncated.")

# Model
flags.DEFINE_integer("word_embed_dim", 50, "The dimension of the embedding matrix.")
flags.DEFINE_integer("hidden_size", 100, "The size of the hidden layer, applicable to some models.")

# Training
flags.DEFINE_integer("batch_size", 100, "The batch size.")
flags.DEFINE_integer("num_epochs", 50, "The numer of epochs to train for.")

flags.DEFINE_boolean("debug", False, "Whether to run in debug mode, i.e. use a smaller dataset and increase verbosity.")
flags.DEFINE_boolean("train", True, "Whether to train or test the model.")

FLAGS = flags.FLAGS

def run_train_epoch(sess, model, dataset, epoch_num):
    prog = Progbar(target=dataset.split_num_batches("train", FLAGS.batch_size))
    for i, batch in enumerate(dataset.get_shuffled_iterator("train", FLAGS.batch_size)):
        loss = model.train_on_batch(sess, *batch)
        prog.update(i + 1, [("train loss", loss)])

def run_eval_epoch(sess, model, dataset, split):
    prog = Progbar(target=dataset.split_num_batches(split, FLAGS.batch_size))
    for i, batch in enumerate(dataset.get_iterator(split, FLAGS.batch_size)):
        _, loss = model.evaluate_on_batch(sess, *batch)
        prog.update(i + 1, [("%s loss" % split, loss)])

def main(_):
    with tf.Graph().as_default():
        vocab = Vocab(FLAGS.snli_dir, FLAGS.vocab_file, FLAGS.max_vocab_size)
        dataset = Dataset(FLAGS.snli_dir, vocab, FLAGS.max_seq_len, debug=FLAGS.debug)
        embedding_matrix = get_glove_vectors(FLAGS.glove_dir, FLAGS.word_embed_dim, vocab)

        if FLAGS.model == "SOW":
            model = SumOfWords(
                embedding_matrix=embedding_matrix,
                hidden_size=FLAGS.hidden_size,
                max_seq_len=FLAGS.max_seq_len
            )
        else:
            raise ValueError("Unrecognized model: %s." % FLAGS.model)
        model.build()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            if FLAGS.train:
                for epoch in range(FLAGS.num_epochs):
                    run_train_epoch(sess, model, dataset, epoch)
                    run_eval_epoch(sess, model, dataset, "dev")
            else:
                raise ValueError("Cannot test the model just yet.")

if __name__ == "__main__":
    tf.app.run()