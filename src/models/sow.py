import tensorflow as tf

from src.models.model import Model
from src.utils.ops import linear, train_op
from src.utils.vocab import Vocab
from src.utils.wvecs import GloveWordVectors


class SumOfWords(Model):

    def __init__(self, config):
        super(SumOfWords, self).__init__(config)
        self.batch_size = config["batch_size"]
        self.seq_len = config["seq_len"]
        self.num_classes = config["num_classes"]
        self.embed_size = config["embed_size"]
        self.hidden_size = config["hidden_size"]
        self.opt_config = config["opt"]
        self.learning_rate = config["learning_rate"]
        self.vocab = Vocab(vocab_file=config["vocab_file"], dataset_path=config["dataset_path"],
            max_vocab_size=config["vocab_size"])
        self.pretrained_embeddings = GloveWordVectors(tokens=self.vocab.token_id, \
            glove_file=config["glove_file"]).word_vectors

    def add_placeholders(self):
        self.prem = tf.placeholder(tf.int32, [self.batch_size, self.seq_len])
        self.hyp = tf.placeholder(tf.int32, [self.batch_size, self.seq_len])
        self.labels = tf.placeholder(tf.int32, [self.num_classes])

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        prem_input, hyp_input = inputs_batch
        feed_dict = {}
        feed_dict[self.prem] = prem_input
        feed_dict[self.hyp] = hyp_input
        if labels_batch is not None:
            feed_dict[self.labels] = labels_batch
        return feed_dict

    def _get_embedding(self, indices, output_shape):
        with tf.variable_scope("embedding"):
            embeddings = tf.get_variable(
                "E",
                initializer=tf.constant(self.pretrained_embeddings)
            )
            embedded_vectors = tf.nn.embedding_lookup(embeddings, indices)
            embeddings = tf.reshape(embedded_vectors, shape=output_shape)
        return embeddings

    def add_prediction_op(self):
        with tf.variable_scope("prediction"):
            prem_embed = self._get_embedding(
                self.prem, [self.batch_size, self.seq_len, self.embed_size])
            hyp_embed = self._get_embedding(
                self.prem, [self.batch_size, self.seq_len, self.embed_size])

            prem_sow = tf.reduce_sum(prem_embed, axis=1)
            hyp_sow = tf.reduce_sum(hyp_sow, axis=1)
            both_sow = tf.concat(1, [prem_sow, hyp_sow])

            h1 = linear(both_sow, self.hidden_size, activation="tanh")
            h2 = linear(h1, self.hidden_size, activation="tanh")
            h3 = linear(h2, self.hidden_size, activation="tanh")

            logits = linear(h3, self.num_classes)
            preds = tf.argmax(logits, axis=1)
            return preds, logits

    def add_loss_op(self, pred, logits):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.labels))

    def add_training_op(self, loss):
        return train_op(self.opt_config, self.learning_rate, loss)

if __name__ == "__main__":
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="
    debug = True
    parser, embeddings, train_examples, dev_set, test_set = load_and_preprocess_data(debug)
    if not os.path.exists('./data/weights/'):
        os.makedirs('./data/weights/')
    config = {
        "batch_size": 10,
        "seq_len": 10,
        "num_classes": 3,
        "embed_size": 50,
        "hidden_size": 10,
        "num_classes": 3,
        "learning_rate": 0.0001,
        "opt": {
            "name": "AdaDelta",
            "rho": 0.95,
            "epsilon": 1e-8
        }
    }

    with tf.Graph().as_default():
        print "Building model...",
        start = time.time()
        model = SumOfWords(config)
        parser.model = model
        print "took {:.2f} seconds\n".format(time.time() - start)

        init = tf.global_variables_initializer()
        # If you are using an old version of TensorFlow, you may have to use
        # this initializer instead.
        # init = tf.initialize_all_variables()
        saver = None if debug else tf.train.Saver()

        with tf.Session() as session:
            parser.session = session
            session.run(init)

            print 80 * "="
            print "TRAINING"
            print 80 * "="
            model.fit(session, saver, parser, train_examples, dev_set)

            if not debug:
                print 80 * "="
                print "TESTING"
                print 80 * "="
                print "Restoring the best model weights found on the dev set"
                saver.restore(session, './data/weights/parser.weights')
                print "Final evaluation on test set",
                UAS, dependencies = parser.parse(test_set)
                print "- test UAS: {:.2f}".format(UAS * 100.0)
                print "Writing predictions"
                with open('q2_test.predicted.pkl', 'w') as f:
                    cPickle.dump(dependencies, f, -1)
                print "Done!"