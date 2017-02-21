import numpy as np
import os

"""
Loads a GloVe model with pretrained word embeddings, for a given set of tokens. 

For more information: http://nlp.stanford.edu/projects/glove/
GloVe data: http://nlp.stanford.edu/data/glove.6B.zip

"""

class GloveWordVectors(object):

    DEFAULT_FILE_PATH = "utils/datasets/glove.6B.50d.txt"

    def __init__(self, glove_file):
        self.word_vectors = np.zeros()  # initialized to an embedding matrix
        self.glove_file = glove_file

        if os.path.isfile(glove_file) and os.path.getsize(glove_file) > 0:
            self.load_vocab_from_file(glove_file)
        else:
            raise Exception("Must provide a pretrained GloVe file. See http://nlp.stanford.edu/data/glove.6B.zip.")


    """
    Loads pretrained GloVe vectors into memory.

    args:
        tokens - The mapping from token to idx for every token in the vocabulary. Must be incremental.
        filepath - The path of the file containing the dumped pretrained GloVe vectors.
        dimensions - The dimensionality of a word vector, as given by the GloVe file.
    """
    def load_vocab_from_file(tokens, filepath=DEFAULT_FILE_PATH, dimensions=50):
        self.word_vectors = np.zeros((len(tokens), dimensions))
        with open(filepath) as ifs:
            for line in ifs:
                line = line.strip()
                if not line:
                    continue
                row = line.split()
                token = row[0]
                if token not in tokens:
                    continue
                data = [float(x) for x in row[1:]]
                if len(data) != dimensions:
                    raise RuntimeError("wrong number of dimensions")
                self.word_vectors[tokens[token]] = np.asarray(data)
        print "Done.",len(self.word_vectors), " GloVe vectors loaded!"
