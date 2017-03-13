import numpy as np
import os
import json
from nltk.tokenize import word_tokenize


"""
Loads a GloVe model with pretrained word embeddings, for a given set of tokens. 

For more information: http://nlp.stanford.edu/projects/glove/
GloVe data: http://nlp.stanford.edu/data/glove.6B.zip

"""

def get_glove_vectors(glove_file, dim, vocab, dataset_dir, 
                        avg_unseen_neighbors, window_size):
    glove_saved_file = glove_file[:-4] + "_saved"
    if avg_unseen_neighbors:
        glove_saved_file += "_avg_" + str(window_size) + ".npy"
    else:
        glove_saved_file += ".npy"
    # if os.path.isfile(glove_saved_file):
    #     with open(glove_saved_file, "r") as glove_saved:
    #         npzfile = np.load(glove_saved)
    #         matrix = npzfile['matrix']
    #         missing_indices = npzfile['missing_indices']
    #     return matrix, missing_indices

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

    if avg_unseen_neighbors:
        print "Getting the average of neighbors of the %d missing words" % len(missing_indices)
        matrix = average_neighbors(matrix, vocab, missing_indices, window_size, dataset_dir)    

    with open(glove_saved_file, "w") as out:
        np.savez(out, matrix=matrix, missing_indices=list(missing_indices))

    return matrix, list(missing_indices)

def average_neighbors(matrix, vocab, missing_indices, window_size, dataset_dir):
    missing_words = [vocab.token_for_id(i) for i in missing_indices]
    neighbors_count = {}
    for missing_word in missing_words:
        neighbors_count[missing_word] = 0

    for dataset in ["snli_1.0_train.jsonl", "snli_1.0_dev.jsonl", "snli_1.0_test.jsonl"]:
        print "Working on getting average of neighbors in ", dataset
        count = 0
        for line in open(os.path.join(dataset_dir, dataset), "r").readlines():
            data = json.loads(line)
            sentence = word_tokenize(data["sentence1"].lower())
            count += 1
            if count % 100 == 0:
                print "Working on example %d" % count
            for missing_word in missing_words:
                if missing_word in sentence:
                    missing_sentence_index = sentence.index(missing_word)

                    start = max(missing_sentence_index - window_size, 0)
                    end = min(len(sentence), missing_sentence_index + window_size)
                    for neighbor_index in xrange(start, end):
                        if not (neighbor_index == missing_sentence_index) and \
                                (sentence[neighbor_index] not in missing_words):
                            missing_index = vocab.id_for_token(missing_word)    
                            matrix[missing_index] += matrix[vocab.id_for_token(sentence[neighbor_index])]
                            neighbors_count[missing_word] += 1

    for missing_word in missing_words:
        if neighbors_count[missing_word] > 0:
            index = vocab.id_for_token(missing_word)
            matrix[index] /= neighbors_count[missing_word]    
    
    return matrix