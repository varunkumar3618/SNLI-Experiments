import numpy as np
import os
import json
from nltk.tokenize import word_tokenize


"""
Loads a GloVe model with pretrained word embeddings, for a given set of tokens. 

For more information: http://nlp.stanford.edu/projects/glove/
GloVe data: http://nlp.stanford.edu/data/glove.6B.zip

"""

def get_glove_vectors(glove_file, glove_saved_file, dim, vocab, dataset_dir, window_size):
    glove_saved_file = glove_file[:-4] + "_saved"
    glove_saved_file += ".npy"
    if os.path.isfile(glove_saved_file):
        with open(glove_saved_file, "r") as glove_saved:
            npzfile = np.load(glove_saved)
            matrix = npzfile['matrix']
            missing_indices = npzfile['missing_indices']
        return matrix, missing_indices

    matrix = np.zeros([vocab.size(), dim])
    missing_indices = set([x for x in xrange(vocab.size())])
    found_set = set()

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
            index = vocab.id_for_token(token)
            matrix[index] = np.asarray(data)
            missing_indices.remove(index)
            found_set.add(token)
    
    print "0, 1 in missing", (0 in missing_indices), (1 in missing_indices)
    matrix[0] = 0
    matrix[1] = 0
    print "WARNING: %s tokens were not found in the Glove file. Their embeddings will be set to zero."\
        % (len(vocab.token_id.keys()) - len(found_set))

    with open(glove_saved_file, "w") as out:
        np.savez(out, matrix=matrix, missing_indices=list(missing_indices))

    return matrix, list(missing_indices)
