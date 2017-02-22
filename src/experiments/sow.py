import tensorflow as tf

from src.utils.config import Config, TrainConfig
from src.models.sow import SumOfWords, SumOfWordsConfig

if __name__ == '__main__':
    config = Config(json_file="config.json")
    config.add_train_config(TrainConfig(config_dict={
        "batch_size": 10,
        "num_threads": 2,
        "capacity": 1000,
        "min_after_dequeue": 100,
        "train_op": "AdaDelta",
        "learning_rate": 0.01,
        "rho": 0.95,
        "epsilon": 1e-8
    }))
    config.add_model_config(SumOfWordsConfig(config_dict={
        "wvec_dim": 50,
        "name": "sow",
        "hidden_size": 20
    }))
    with tf.Graph().as_default():
        model = SumOfWords(config)
        model.train()
