import os
from itertools import count

import tensorflow as tf

from src.utils.dataset import load_data
from src.utils.ops import make_train_op, mean_ce_with_logits
from src.utils.vocab import Vocab
from src.utils.wvecs import get_glove_vectors

class SNLIModel(object):
    def __init__(self, config):
        self.vocab = Vocab(config)
        self.embedding_matrix = get_glove_vectors(config, self.vocab)

        data = load_data(config)

        self.train_labels, self.train_sentence1s, self.train_sentence2s = data["train"]
        self.dev_labels, self.dev_sentence1s, self.dev_sentence2s = data["dev"]
        self.test_labels, self.test_sentence1s, self.test_sentence2s = data["test"]

        with tf.variable_scope("model"):
            with tf.variable_scope("logits"):
                self.train_logits = self.build_model(config, self.train_sentence1s, self.train_sentence2s)
            with tf.variable_scope("logits", reuse=True):
                self.dev_logits = self.build_model(config, self.dev_sentence1s, self.dev_sentence2s)
            with tf.variable_scope("logits", reuse=True):
                self.test_logits = self.build_model(config, self.test_sentence1s, self.test_sentence2s)

            self.train_loss = mean_ce_with_logits(self.train_logits, self.train_labels)
            self.train_op = make_train_op(config, self.train_loss)

            self.dev_loss = mean_ce_with_logits(self.dev_logits, self.dev_labels)
            self.test_loss = mean_ce_with_logits(self.test_logits, self.test_labels)

            self.dev_preds = tf.argmax(self.dev_logits, axis=1)
            self.test_preds = tf.argmax(self.test_logits, axis=1)

            self.dev_acc, self.dev_acc_update_op\
                = tf.metrics.accuracy(self.dev_labels, self.dev_preds, weights=tf.ones_like(self.dev_labels))
            self.test_acc, self.test_acc_update_op\
                = tf.metrics.accuracy(self.test_labels, self.test_preds, weights=tf.ones_like(self.test_labels))

        self.logdir = os.path.join(os.path.join(config.data_dir, config.log_dir), config.model.name)

    def build_model(self, config, sentence1, sentence2, reuse=False):
        raise NotImplementedError("This function should return predictions and logits for the two sentences.")

    def train(self):
        sv = tf.train.Supervisor(logdir=self.logdir)
        with sv.managed_session() as sess:
            print "="*79
            print "="*79
            for step in count():
                _, loss = sess.run([self.train_op, self.train_loss])
                print "Step: %s. Loss: %s" % (step, loss)

                if step % 10000 == 0:
                    print "="*79
                    print "="*79
                    acc, loss, _ = sess.run([self.dev_acc, self.dev_loss, self.dev_acc_update_op])
                    print "Step: %s. Dev loss: %s. Dev accuracy: %s" % (step, loss, acc)
                    print "="*79
                    print "="*79

    def test(self):
        acc, loss = sess.run([self.test_acc, self.test_loss])
        print "Accuracy: %s, loss: %s" % (acc, loss)
