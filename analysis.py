import os
import itertools

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.metrics

from src.utils.dataset import Dataset
from src.utils.vocab import Vocab

flags = tf.app.flags

# File paths
flags.DEFINE_string("data_dir", "data/", "The location of the data files.")
flags.DEFINE_string("name", "model", "The name of the model, used to save logs and checkpoints.")

# Data
flags.DEFINE_integer("max_seq_len", 100, "The maximum length of a sentence. Sentences longer than this will be truncated.")

#Analysis
flags.DEFINE_string("analysis_type", "", "The analysis to run.")
flags.DEFINE_string("split", "dev", "The split to analyze.")
flags.DEFINE_string("analysis_path", "", "Where to save the analysis.")

FLAGS = flags.FLAGS
if len(FLAGS.analysis_path) == 0 or len(FLAGS.analysis_type) == 0: 
    raise ValueError("The analysis type and the analysis path must both be supplied.")

snli_dir = os.path.join(FLAGS.data_dir, "snli_1.0")
vocab_file = os.path.join(FLAGS.data_dir, "vocab.txt")
regular_data_file = os.path.join(FLAGS.data_dir, "data.pkl")
debug_data_file = os.path.join(FLAGS.data_dir, "debug_data.pkl")
base_models_dir = os.path.join(FLAGS.data_dir, "models")
model_dir = os.path.join(base_models_dir, FLAGS.name)
checkpoint_dir = os.path.join(model_dir, "checkpoint")
checkpoint_path = os.path.join(checkpoint_dir, "model.ckpt")
results_dir = os.path.join(model_dir, "results")

def confusion(vocab, dataset):
    true_labels = dataset.get_true_labels(FLAGS.split)
    predicted_labels = np.load(os.path.join(results_dir, "predictions_%s.npy" % FLAGS.split))
    cm = sklearn.metrics.confusion_matrix(true_labels, predicted_labels)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.colorbar()
    classes = ["entailment", "neutral", "contradicton"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(FLAGS.analysis_path)

def error_report(vocab, dataset):
    true_labels = dataset.get_true_labels(FLAGS.split)
    sentence1s = dataset.get_sentence1(FLAGS.split)
    sentence2s = dataset.get_sentence2(FLAGS.split)
    predicted_labels = np.load(os.path.join(results_dir, "predictions_%s.npy" % FLAGS.split))

    with open(FLAGS.analysis_path, "wb") as outf:
        for i, (true_label, sentence1, sentence2, predicted_label)\
                in enumerate(zip(true_labels, sentence1s, sentence2s, predicted_labels)):
            outf.write("%s)\n" % i)
            outf.write("Sentence1: %s\n" % sentence1)
            outf.write("Sentence2: %s\n" % sentence2)
            outf.write("True label: %s\n" % true_label)
            outf.write("Predicted label: %s\n" % predicted_label)
            outf.write("\n")

def main(_):
    vocab = Vocab(snli_dir, vocab_file)
    dataset = Dataset(snli_dir, regular_data_file, debug_data_file, vocab,
                      FLAGS.max_seq_len, debug=True)

    if FLAGS.analysis_type == "confusion":
        confusion(vocab, dataset)
    elif FLAGS.analysis_type == "error_report":
        error_report(vocab, dataset)
    else:
        raise ValueError("Unrecognized analysis: %s" % FLAGS.analysis_type)

if __name__ == "__main__":
    tf.app.run()