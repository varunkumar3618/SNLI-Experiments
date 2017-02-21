from __future__ import absolute_import
import os

from src.utils.vocab import Vocab

DATA_DIR = "./data"
SNLI_DIR = os.path.join(DATA_DIR, "snli_1.0")
VOCAB_FILE = os.path.join(DATA_DIR, "vocab.txt")
MAX_VOCAB_SIZE = 10000


def create_vocab_file():
    vocab = Vocab(vocab_file=VOCAB_FILE, dataset_path=SNLI_DIR,
                  max_vocab_size=MAX_VOCAB_SIZE)


def main():
    create_vocab_file()

if __name__ == '__main__':
    main()
