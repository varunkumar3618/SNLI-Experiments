# Source: https://github.com/DeNeutoy/Decomposable_Attn/blob/master/Vocab.py
import os
import json
from collections import Counter
from nltk.tokenize import word_tokenize


class Vocab(object):

    def __init__(self, snli_dir, vocab_file):
        self.vocab_file = vocab_file

        self.token_id = {}
        self.id_token = {}
        self.PAD_ID = 0
        self.UNK_ID = 1
        self.seq = 2

        if os.path.isfile(vocab_file):
            self.load_vocab_from_file(vocab_file)
        else:
            self.create_vocab(snli_dir, vocab_file)
            self.load_vocab_from_file(vocab_file)

    def load_vocab_from_file(self, vocab_file):
        for i, line in enumerate(open(vocab_file, "r")):
            token, idx = line.strip().split("\t")
            idx = int(idx)
            assert token not in self.token_id, "dup entry for token [%s]" % token
            assert idx not in self.id_token, "dup entry for idx [%s]" % idx
            self.token_id[token] = idx
            self.id_token[idx] = token

    def create_vocab(self, dataset_path, vocab_path):
        print("generating vocab from dataset at {}".format(dataset_path))
        all_words = []
        for dataset in ["snli_1.0_train.jsonl", "snli_1.0_dev.jsonl", "snli_1.0_test.jsonl"]:
            for line in open(os.path.join(dataset_path, dataset), "r").readlines():
                data = json.loads(line)
                all_words += word_tokenize(data["sentence1"].lower())
                all_words += word_tokenize(data["sentence2"].lower())

        counter = Counter(all_words)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(len(words))))

        vocab_size = len(word_to_id)
        with open(vocab_path, "w") as file:
            for word, id in word_to_id.items():
                file.write("{}\t{}\n".format(word, id))

        print("vocab of size {} written to {}, with _PAD_ token == 0, _UNK_ token == 1".format(
            vocab_size, vocab_path))

    def size(self):
        return len(self.token_id) + 2  # +1 for UNK & PAD

    def id_for_token(self, token):
        return self.token_id[token]

    def has_token(self, token):
        return token in self.token_id

    def ids_for_tokens(self, tokens):
        return [self.id_for_token(t) for t in tokens]

    def ids_for_sentence(self, sentence):
        return self.ids_for_tokens(word_tokenize(sentence.lower()))

    def sentence_for_ids(self, ids):
        return " ".join(self.tokens_for_ids(ids)).strip()

    def token_for_id(self, id):
        return self.id_token[id]

    def tokens_for_ids(self, ids):
        return [self.token_for_id(x) for x in ids]
