import os
import json

import tensorflow as tf
import pandas as pd


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tensorflow(config, vocab):
    """Converts the SNLI json lines files to the TFRecords format."""
    splits = config.splits
    data_dir = config.data_dir
    snli_dir = os.path.join(data_dir, config.snli_dir)
    jsonl_split_files = config.jsonl_split_files
    tf_split_files = config.tf_split_files
    label_to_int = config.label_to_int

    for split in splits:
        jsonl_file = os.path.join(snli_dir, jsonl_split_files[split])
        tf_file = os.path.join(snli_dir, tf_split_files[split])

        if os.path.isfile(tf_file):
            print "The file %s already exists, skipping." % tf_file
            continue

        dataframe = pd.read_json(jsonl_file, lines=True)
        with tf.python_io.TFRecordWriter(tf_file) as tf_writer:
            for index, row in dataframe.iterrows():
                if row["gold_label"] not in label_to_int:
                    continue
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": _int64_feature(label_to_int[row["gold_label"]]),
                    "sentence1": _int64_feature(vocab.ids_for_tokens(row["sentence1"], update=False)),
                    "sentence2": _int64_feature(vocab.ids_for_tokens(row["sentence1"], update=False))
                }))
                tf_writer.write(example.SerializeToString())


def load_split(config, split):
    """Loads a single split from its TFRecords file"""
    snli_dir = os.path.join(config.data_dir, config.snli_dir)
    tf_split_files = config.tf_split_files
    filepath = os.path.join(snli_dir, tf_split_files[split])

    filename_queue = tf.train.string_input_producer([filepath])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "label": tf.FixedLenFeature([], tf.int64),
            "sentence1": tf.VarLenFeature(tf.int64),
            "sentence2": tf.VarLenFeature(tf.int64)
        }
    )
    return [features["label"], features["sentence1"].values, features["sentence2"].values]

def make_queue(config, data_split):
    return tf.train.batch(data_split, config.train.batch_size, num_threads=config.train.num_threads,
                          capacity=config.train.capacity, allow_smaller_final_batch=True,
                          dynamic_pad=True)

def make_shuffle_queue(config, data_split):
    return make_queue(config, data_split)
    # return tf.train.shuffle_batch(data_split, config.train.batch_size, config.train.capacity,
    #                               config.train.min_after_dequeue, num_threads=config.train.num_threads,
    #                               allow_smaller_final_batch=True)

def load_data(config):
    """Loads data from the TFRecords files"""
    return {
        "train": make_shuffle_queue(config, load_split(config, "train")),
        "dev": make_queue(config, load_split(config, "dev")),
        "test": make_queue(config, load_split(config, "test"))
    }
