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

sum_softmax = np.zeros([9842, 3])
for model in models:
	save_path = os.path.join(results_dir, model)
	save_path = os.path.join(save_path, "results/logits_%s.npy" % split)
	logits = np.load(save_path)

	softmax = np.exp(logits) 
	softmax2 = softmax/np.expand_dims(np.sum(softmax, axis=1), axis=1)
	sum_softmax += softmax2

for i in xrange(len(pred1)):
	if pred1[i] != pred2[i]:
		print pred1[i], pred2[i], i, logits[i, :], sum_softmax[i, :]

print "Loading vocab"
if not os.path.exists("./data/labels_%s.npy" % split):
	vocab = Vocab(snli_dir, vocab_file)
	dataset = Dataset(snli_dir, regular_data_file, debug_data_file, vocab,
	                          max_seq_len, debug=False)
	print "Loaded dataset"

	labels = dataset.get_true_labels(split)
	# for i, batch in enumerate(dataset.get_iterator(split, batch_size)):
	#     _, _, _, _, labels_batch = batch
	#     labels.append(labels_batch)
	for i in xrange(len(labels)):
		print labels[i]
	# labels2 = np.matrix(labels)
	np.save("./data/labels_%s.npy" % split, np.matrix(labels))
else:
	labels = np.load("./data/labels_%s.npy" % split)
	labels = np.squeeze(labels)

preds = np.argmax(sum_softmax, axis=1)
accuracy = float(np.sum(preds==labels))/len(preds)
print "Ensemble accuracy is %f" % accuracy