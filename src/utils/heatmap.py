import os
import itertools

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.metrics

from src.utils.dataset import Dataset
from src.utils.vocab import Vocab

def plot_heatmap(premise_sentence, hypothesis_sentence, attention, path):
    fig, ax = plt.subplots()

    # TODO: parse pandas style
    premise_sentence_len = len(premise_sentence.split(" "))
    hypothesis_sentence_len = len(hypothesis_sentence.split(" "))

    # Common formatting for both 1d, 2d attentions
    ax.set_xticks(np.arange(premise_sentence_len)+0.5, minor=False)
    ax.set_aspect('equal') # X scale matches Y scale
    ax.xaxis.tick_bottom()
    ax.set_xticklabels(premise_sentence.split(" "), rotation="vertical")

    heatmap = None
    if len(attention.shape) == 1:  # attention vector
        heatmap = ax.pcolor([attention[1 : premise_sentence_len + 1]], cmap=mpl.cm.Blues)

        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_title("Hypothesis: %s" % hypothesis_sentence)
    else:  # attention matrix

        # TODO: verify proper axis alignment of premise/hypothesis.
        # heatmap = ax.pcolor(attention[1 : premise_sentence_len + 1][1 : hypothesis_sentence_len], cmap=mpl.cm.Blues)
        attention_subset = attention[: premise_sentence_len + 1, : hypothesis_sentence_len + 1]
        print attention_subset
        heatmap = ax.pcolor(attention_subset, cmap=mpl.cm.Blues)
        ax.set_yticks(np.arange(hypothesis_sentence_len)+0.5, minor=False)
        ax.set_yticklabels(hypothesis_sentence.split(" "))

    plt.savefig(path)
