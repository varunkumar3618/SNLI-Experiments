# Source: https://github.com/DeNeutoy/Decomposable_Attn/blob/master/Vocab.py
import os
import json
from collections import Counter
from nltk.tokenize import word_tokenize


class Vocab(object):

    def __init__(self, snli_dir, vocab_file, max_vocab_size, use_all=True):
        self.vocab_file = vocab_file
        self.max_vocab_size = max_vocab_size
        self.use_all = use_all

        self.token_id = {}
        self.id_token = {}
        self.PAD_ID = 0
        self.UNK_ID = 1
        self.seq = 2

        if os.path.isfile(vocab_file) and self.has_correct_size(
                                            vocab_file, max_vocab_size, use_all):
            self.load_vocab_from_file(vocab_file)
        else:
            self.create_vocab(snli_dir, vocab_file, max_vocab_size, use_all)
            self.load_vocab_from_file(vocab_file)

    def has_correct_size(self, vocab_file, max_vocab_size, use_all):
        NUM_WORDS_ALL = 37285
        try:
            with open(vocab_file, "r") as f:
                size = int(f.readline())
        except:
            return False
        if not use_all:
            return size == max_vocab_size
        else:
            return size == NUM_WORDS_ALL

    def load_vocab_from_file(self, vocab_file):
        for i, line in enumerate(open(vocab_file, "r")):
            if i == 0:
                # Skip the max vocab size line
                continue
            token, idx = line.strip().split("\t")
            idx = int(idx)
            assert token not in self.token_id, "dup entry for token [%s]" % token
            assert idx not in self.id_token, "dup entry for idx [%s]" % idx
            # if idx == 0:
            #     assert token == "_PAD_", "expect id 0 to be [_PAD_] not [%s]" % token
            # if idx == 1:
            #     assert token == "_UNK_", "expect id 1 to be [_UNK_] not [%s]" % token
            self.token_id[token] = idx
            self.id_token[idx] = token

    def create_vocab(self, dataset_path, vocab_path, max_vocab_size, use_all):
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
        words = ["_PAD_"] + ["_UNK_"] + list(words)
        if not use_all:
            word_to_id = dict(zip(words[:max_vocab_size], range(max_vocab_size)))
        else:
            word_to_id = dict(zip(words, range(len(words))))

        vocab_size = len(word_to_id)
        print "Vocab size is %d" % vocab_size
        with open(vocab_path, "w") as file:
            file.write("{}\n".format(vocab_size))
            for word, id in word_to_id.items():
                file.write("{}\t{}\n".format(word, id))

        print("vocab of size {} written to {}, with _PAD_ token == 0, _UNK_ token == 1".format(
            vocab_size, vocab_path))

    def size(self):
        return len(self.token_id) + 2  # +1 for UNK & PAD

    def id_for_token(self, token, update=False):
        if token in self.token_id:
            return self.token_id[token]
        elif not update:
            return self.UNK_ID
        elif self.vocab_file is not None:
            raise Exception("cstrd with vocab_file=[%s] but missing entry [%s]" % (
                self.vocab_file, token))
        else:
            self.token_id[token] = self.seq
            self.id_token[self.seq] = token
            self.seq += 1
            return self.seq - 1

    def has_token(self, token):
        return token in self.token_id

    def ids_for_tokens(self, tokens, update=False):
        return [self.id_for_token(t, update) for t in tokens]

    def ids_for_sentence(self, sentence):
        return self.ids_for_tokens(word_tokenize(sentence.lower()))

    def sentence_for_ids(self, ids):
        return " ".join(self.tokens_for_ids(ids)).strip()

    def token_for_id(self, id):
        if id in self.id_token:
            return self.id_token[id]
        else:
            print("ID not in vocab, returning <UNK>")
            return self.id_token[self.UNK_ID]

    def tokens_for_ids(self, ids):
        return [self.token_for_id(x) for x in ids]
