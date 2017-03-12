import itertools
import matplotlib.pyplot as plt
import numpy as np

def output_confusion_matrix(cm, filename):
    """ Prints a confusion matrix grid PNG to the given file.

    Args:
        cm: np.ndarray confusion matrix of type int32
        filename: the name of the JPG output file"""
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.colorbar()
    classes = ["entailment", "neutral", "contradicton"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)
