#!/bin/bash

DATASETS_DIR="data"

mkdir -p $DATASETS_DIR

cd $DATASETS_DIR

# Get the SNLI dataset
if hash wget 2>/dev/null; then
    wget http://nlp.stanford.edu/projects/snli/snli_1.0.zip
else:
    curl -O http://nlp.stanford.edu/projects/snli/snli_1.0.zip
fi
unzip snli_1.0.zip
rm snli_1.0.zip

# Get 50D GloVe vectors
if hash wget 2>/dev/null; then
  wget http://web.stanford.edu/~jamesh93/tmp/glove.6B.zip
else
  curl -O http://web.stanford.edu/~jamesh93/tmp/glove.6B.zip
fi
unzip glove.6B.zip
rm glove.6B.zip