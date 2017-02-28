#!/bin/bash

DATASETS_DIR="data"

mkdir -p $DATASETS_DIR

cd $DATASETS_DIR

# Get the SNLI dataset
if hash wget 2>/dev/null; then
  wget http://nlp.stanford.edu/projects/snli/snli_1.0.zip
else
  curl -O http://nlp.stanford.edu/projects/snli/snli_1.0.zip
fi
unzip snli_1.0.zip
rm snli_1.0.zip

# Get 50D GloVe vectors
if hash wget 2>/dev/null; then
  wget http://nlp.stanford.edu/data/glove.6B.zip
else
  curl -O http://nlp.stanford.edu/data/glove.6B.zip
fi
mkdir -p glove.6B
unzip glove.6B.zip -d glove.6B/
rm glove.6B.zip

# Get the common crawl word vectors
if hash wget 2>/dev/null; then
  wget http://nlp.stanford.edu/data/glove.840B.300d.zip
else
  curl -O http://nlp.stanford.edu/data/glove.840B.300d.zip
fi
mkdir -p glove.840B.300d
unzip glove.840B.300d.zip -d glove.840B.300d/
rm glove.840B.300d.zip
