import tensorflow as tf
import numpy as np
import os

from src.utils.dataset import Dataset
from src.utils.vocab import Vocab

models = ["SOW-default"]
results_dir = "data/models/"
batch_size = 100
split = "dev"

snli_dir = "data/snli_1.0"
regular_data_file = "data/data.pkl"
debug_data_file = "data/debug_data.pkl"
vocab_file = "data/vocab.txt"
max_seq_len = 100

sum_logits = np.zeros([126, ])
for model in models:
	save_path = os.path.join(results_dir, model)
	save_path = os.path.join(save_path, "results/logits_%s.npy" % split)
	logits = np.load(save_path)
	print logits.shape
	sum_logits += logits

print sum_logits
print sum_logits.shape
print "Loading vocab"
vocab = Vocab(snli_dir, vocab_file)
dataset = Dataset(snli_dir, regular_data_file, debug_data_file, vocab,
                          max_seq_len, debug=False)
print "Loaded dataset"

labels = []
for i, batch in enumerate(dataset.get_iterator(split, batch_size)):
    _, _, _, _, labels_batch = batch
    labels.append(labels_batch)

labels = np.concatenate(labels)
preds = np.argmax(sum_logits, axis=1)
accuracy = np.sum(pred == labels)/len(pred)
print "Ensemble accuracy is %f" % accuracy