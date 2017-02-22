import numpy as np
import os

"""
Loads a GloVe model with pretrained word embeddings, for a given set of tokens. 

For more information: http://nlp.stanford.edu/projects/glove/
GloVe data: http://nlp.stanford.edu/data/glove.6B.zip

"""

def get_glove_vectors(config, vocab):
    glove_dir = os.path.join(config.data_dir, config.glove_dir)
    dim = config.model.wvec_dim
    glove_file = os.path.join(glove_dir, "glove.6B.%sd.txt" % dim)
    matrix = np.zeros([vocab.size(), dim])

    with open(glove_file) as ifs:
        for line in ifs:
            line = line.strip()
            if not line:
                continue
            row = line.split()
            token = row[0]
            if not vocab.has_token(token):
                continue
            data = [float(x) for x in row[1:]]
            if len(data) != dim:
                raise RuntimeError("wrong number of dimensions")
            matrix[vocab.id_for_token(token)] = np.asarray(data)
    return matrix
