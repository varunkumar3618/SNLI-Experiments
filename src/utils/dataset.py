import os
import math
import pickle

import tensorflow as tf
import pandas as pd
import numpy as np


class Dataset(object):
    def __init__(self, snli_dir, regular_data_file, debug_data_file, vocab,
                 max_seq_length, debug=False):
        data_file = debug_data_file if debug else regular_data_file
        if os.path.isfile(data_file)\
                and self._file_is_valid(data_file, vocab, max_seq_length, debug):
            with open(data_file, "r") as f:
                self._dataframes = pickle.load(f)[0]
            print "Loading data from the pickled file..."
        else:
            self._dataframes = self._create_dataframes(snli_dir, data_file, vocab,
                                                       max_seq_length, debug)
            print "Pickling the data object..."
            with open(data_file, "wb") as f:
                pickle.dump((self._dataframes, max_seq_length, vocab.max_vocab_size,
                            vocab.vocab_file, debug), f)

    def _file_is_valid(self, data_file, vocab, max_seq_length, debug):
        with open(data_file, "r") as f:
            _, max_seq_length_, max_vocab_size, vocab_file, debug_ = pickle.load(f)
            return max_seq_length == max_seq_length_ and max_vocab_size == vocab.max_vocab_size\
                and vocab_file == vocab.vocab_file and debug == debug_

    def _create_dataframes(self, snli_dir, data_file, vocab, max_seq_length, debug=False):
        dataframes = {}
        for split in ["train", "dev", "test"]:
            filepath = os.path.join(snli_dir, "snli_1.0_%s.jsonl" % split)
            df = pd.read_json(filepath, lines=True)

            if debug:
                df = df[:1000]

            # Remove examples without a gold label due to annotator disagreement
            df = df.loc[df["gold_label"] != "-"]

            # Tokenize the sentences, convert the tokens to indices, and pad the sequences
            df["s1_indices"] = df["sentence1"].apply(lambda s1: vocab.ids_for_sentence(s1))
            df["s2_indices"] = df["sentence2"].apply(lambda s1: vocab.ids_for_sentence(s1))
            df["s1_len"] = df["s1_indices"].apply(lambda seq: min(len(seq), max_seq_length))
            df["s2_len"] = df["s2_indices"].apply(lambda seq: min(len(seq), max_seq_length))

            def pad_fn(seq):
                if len(seq) > max_seq_length:
                    print "WARNING: some sequences will be truncated."
                    return seq[:max_seq_length]
                else:
                    return np.pad(seq, (0, max_seq_length - len(seq)), "constant")
            df["s1_padded"] = df["s1_indices"].apply(pad_fn)
            df["s2_padded"] = df["s2_indices"].apply(pad_fn)

            # Covert the label to an integer
            labels_to_ints = {"entailment": 0, "neutral": 1, "contradiction": 2}
            df["l_int"] = df["gold_label"].apply(lambda l: np.array(labels_to_ints[l], dtype=np.int64))

            dataframes[split] = df
        return dataframes

    def get_true_labels(self, split):
        return self._dataframes[split]["l_int"].values

    def _make_batch(self, df):
        # The sequence lengths are required in order to use Tensorflow's dynamic rnn functions correctly
        return np.stack(df["s1_padded"]), np.stack(df["s1_len"]),\
            np.stack(df["s2_padded"]), np.stack(df["s2_len"]),\
            np.stack(df["l_int"])

    def _make_iterator(self, df, batch_size):
        total_examples = len(df)
        examples_read = 0
        while examples_read + batch_size <= total_examples:
            yield self._make_batch(df[examples_read:examples_read + batch_size])
            examples_read += batch_size
        yield self._make_batch(df[examples_read:])

    def get_iterator(self, split, batch_size):
        return self._make_iterator(self._dataframes[split], batch_size)

    def get_shuffled_iterator(self, split, batch_size):
        df = self._dataframes[split]
        return self._make_iterator(df.sample(len(df)), batch_size)

    def split_size(self, split):
        return len(self._dataframes[split])

    def split_num_batches(self, split, batch_size):
        return int(math.ceil(float(len(self._dataframes[split]))/batch_size))
