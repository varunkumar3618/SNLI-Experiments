import os

import tensorflow as tf
import numpy as np

from src.models.sow import SumOfWords
from src.models.rnn_encoder import RNNEncoder
from src.models.attention import AttentionModel
from src.models.wbw import WBWModel
from src.models.mLSTM import mLSTMModel
from src.models.chen import Chen
from src.models.mpm import MPMatchingModel
from src.models.stacked import MPMStackedModel
from src.utils.dataset import Dataset
from src.utils.vocab import Vocab
from src.utils.wvecs import get_glove_vectors
from src.utils.progbar import Progbar

flags = tf.app.flags
flags.DEFINE_string("model", "SOW", "The type of model to train.")

# File paths
flags.DEFINE_string("data_dir", "data/", "The location of the data files.")
flags.DEFINE_string("name", "model", "The name of the model, used to save logs and checkpoints.")
flags.DEFINE_string("glove_type", "common", "The source of the Glove word vectors used: one of 'wiki' and 'common'")

# Data
flags.DEFINE_integer("max_seq_len", 100, "The maximum length of a sentence. Sentences longer than this will be truncated.")
flags.DEFINE_string("embedding_train_mode", "unseen", "Which glove vectors to train, one of 'all', 'unseen' and 'none'")

# Model
flags.DEFINE_integer("word_embed_dim", 300, "The dimension of the embedding matrix.")
flags.DEFINE_integer("hidden_size", 200, "The size of the hidden layer, applicable to some models.")
flags.DEFINE_boolean("use_peepholes", True, "Whether to use peephole connections, applicable to LSTM models.")
flags.DEFINE_float("dropout_rate", 0.15, "How many units to eliminate during training, applicable to models using dropout.")
flags.DEFINE_boolean("clip_gradients", True, "Whether to clip gradients, applicable to LSTM models.")
flags.DEFINE_float("max_grad_norm", 5., "The maxmium norm that gradients should be allowed to take.")
flags.DEFINE_string("activation", "tanh", "The activation to use in dense layers.")
flags.DEFINE_string("dense_init", "xavier", "The initializer to use in dense layers.")
flags.DEFINE_string("rec_init", "xavier", "The initializer to use in recurrent layers.")
flags.DEFINE_integer("perspectives", 20, "The number of pespectives in multi-perspective matching layers")

# Training
flags.DEFINE_integer("batch_size", 100, "The batch size.")
flags.DEFINE_integer("num_epochs", 50, "The numer of epochs to train for.")
flags.DEFINE_float("l2_reg", 1e-4, "The level of l2 regularization to use.")
flags.DEFINE_float("learning_rate", 1e-3, "The learning rate.")

flags.DEFINE_boolean("debug", False, "Whether to run in debug mode, i.e. use a smaller dataset and increase verbosity.")
flags.DEFINE_string("mode", "train", "Whether to run the model in 'train,' 'dev,', or 'test' mode.")
flags.DEFINE_boolean("save", True, "Whether to save the model.")

FLAGS = flags.FLAGS

snli_dir = os.path.join(FLAGS.data_dir, "snli_1.0")
vocab_file = os.path.join(FLAGS.data_dir, "vocab.txt")
regular_data_file = os.path.join(FLAGS.data_dir, "data.pkl")
debug_data_file = os.path.join(FLAGS.data_dir, "debug_data.pkl")

base_models_dir = os.path.join(FLAGS.data_dir, "models")
model_dir = os.path.join(base_models_dir, FLAGS.name)
checkpoint_dir = os.path.join(model_dir, "checkpoint")
checkpoint_path = os.path.join(checkpoint_dir, "model.ckpt")
results_dir = os.path.join(model_dir, "results")
train_log_dir = os.path.join(model_dir, "train")

if not os.path.isdir(base_models_dir):
    os.mkdir(base_models_dir)
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)
if not os.path.isdir(train_log_dir):
    os.mkdir(train_log_dir)

if FLAGS.glove_type == "wiki":
    glove_folder = os.path.join(FLAGS.data_dir, "glove.6B")
    glove_file = os.path.join(glove_folder, "glove.6B.%sd.txt" % FLAGS.word_embed_dim)
    glove_saved_file = os.path.join(glove_folder, "glove.6B.%sd.npy" % FLAGS.word_embed_dim)
elif FLAGS.glove_type == "common":
    if FLAGS.word_embed_dim != 300:
        raise ValueError("Common Crawl word vectors are only available with dimension 300.")
    glove_folder = os.path.join(FLAGS.data_dir, "glove.840B.300d")
    glove_file = os.path.join(glove_folder, "glove.840B.300d.txt")
    glove_saved_file = os.path.join(glove_folder, "glove.840B.300d.npy")
else:
    raise ValueError("Unrecognized word vector type: %s." % FLAGS.glove_type)

def get_model(vocab):
    print "Embedding matrix"
    snli_dir = os.path.join(FLAGS.data_dir, "snli_1.0")
    embedding_matrix, missing_indices\
            = get_glove_vectors(glove_file, glove_saved_file, FLAGS.word_embed_dim, vocab)

    kwargs = {
        "embedding_matrix": embedding_matrix,
        "embedding_train_mode": FLAGS.embedding_train_mode,
        "hidden_size": FLAGS.hidden_size,
        "l2_reg": FLAGS.l2_reg,
        "max_seq_len": FLAGS.max_seq_len,
        "dropout_rate": FLAGS.dropout_rate,
        "learning_rate": FLAGS.learning_rate,
        "clip_gradients": FLAGS.clip_gradients,
        "max_grad_norm": FLAGS.max_grad_norm,
        "activation": FLAGS.activation,
        "dense_init": FLAGS.dense_init,
        "rec_init": FLAGS.rec_init,
        "missing_indices": missing_indices
    }
    if FLAGS.model != "SOW":
        kwargs["use_peepholes"] = FLAGS.use_peepholes

    if FLAGS.model == "SOW":
        return SumOfWords(**kwargs)
    elif FLAGS.model == "RNNE":
        return RNNEncoder(**kwargs)
    elif FLAGS.model == "ATT":
        return AttentionModel(**kwargs)
    elif FLAGS.model == "WBW":
        return WBWModel(**kwargs)
    elif FLAGS.model == "mLSTM":
        return mLSTMModel(**kwargs)
    elif FLAGS.model == "CHEN":
        return Chen(**kwargs)
    elif FLAGS.model == "MPM":
        kwargs["perspectives"] = FLAGS.perspectives
        return MPMatchingModel(**kwargs)
    elif FLAGS.model == "STK":
        kwargs["perspectives"] = FLAGS.perspectives
        return MPMStackedodel(**kwargs)
    else:
        raise ValueError("Unrecognized model: %s." % FLAGS.model)

def run_train_epoch(sess, model, dataset, train_writer, epoch_num):
    print "="*79
    print "Epoch: %s" % (epoch_num + 1)
    prog = Progbar(target=dataset.split_num_batches("train", FLAGS.batch_size))
    for i, batch in enumerate(dataset.get_shuffled_iterator("train", FLAGS.batch_size)):
        loss, summary = model.train_on_batch(sess, *batch)
        prog.update(i + 1, [("train loss", loss)])
        train_writer.add_summary(summary, global_step=epoch_num * dataset.split_size("train") + i)
    print "="*79

def run_eval_epoch(sess, model, dataset, split):
    batch_sizes = []
    accuracies = []
    preds = []

    print "-"*79
    print "Evaluating on %s." % split
    prog = Progbar(target=dataset.split_num_batches(split, FLAGS.batch_size))
    for i, batch in enumerate(dataset.get_iterator(split, FLAGS.batch_size)):
        acc, loss, pred = model.evaluate_on_batch(sess, *batch)
        prog.update(i + 1, [("%s loss" % split, loss)])

        batch_sizes.append(batch[0].shape[0])
        accuracies.append(acc)
        preds.append(pred)

    accuracy = np.average(accuracies, weights=batch_sizes)
    print "Accuracy: %s" % accuracy
    print "-"*79
    return accuracy, np.concatenate(preds)

def train(model, dataset):
    train_writer = tf.summary.FileWriter(train_log_dir)
    if FLAGS.save:
        saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        best_accuracy = 0
        for epoch in range(FLAGS.num_epochs):
            run_train_epoch(sess, model, dataset, train_writer, epoch)
            dev_accuracy, _ = run_eval_epoch(sess, model, dataset, "train" if FLAGS.debug else "dev")
            if dev_accuracy > best_accuracy and FLAGS.save:
                saver.save(sess, checkpoint_path)
                best_accuracy = dev_accuracy
    train_writer.close()

def test(model, dataset, split):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, checkpoint_path)
        _, preds = run_eval_epoch(sess, model, dataset, split)

        save_path = os.path.join(results_dir, "predictions_%s.txt" % split)
        np.savetxt(save_path, preds)
        np_save_path = os.path.join(results_dir, "predictions_%s.npy" % split)
        np.save(np_save_path, preds)

def main(_):
    with tf.Graph().as_default():
        print "Vocab"
        vocab = Vocab(snli_dir, vocab_file)
        print "Dataset"
        dataset = Dataset(snli_dir, regular_data_file, debug_data_file, vocab,
                          FLAGS.max_seq_len, debug=FLAGS.debug)

        print "Model"
        model = get_model(vocab)
        model.build()

        if FLAGS.mode == "train":
            train(model, dataset)
        elif FLAGS.mode == "dev":
            test(model, dataset, "dev")
        elif FLAGS.mode == "test":
            test(model, dataset, "test")
        else:
            raise ValueError("Unrecognized mode: %s." % FLAGS.mode)

if __name__ == "__main__":
    tf.app.run()
