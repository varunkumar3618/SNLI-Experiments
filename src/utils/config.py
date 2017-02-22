import json


class ConfigBase(object):

    def __init__(self, json_file=None, json_str=None, config_dict=None):
        if json_file is None and json_str is None and config_dict is None:
            raise ValueError(
                "Must provide at least one of json file, json str and config dict.")
        elif json_file is not None:
            self.config_dict = json.load(open(json_file))
        elif json_str is not None:
            self.config_dict = json.loads(json_str)
        else:
            self.config_dict = config_dict

    def full_id(self):
        raise NotImplementedError


class Config(ConfigBase):

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        config_dict = self.config_dict

        self.splits = config_dict["splits"]
        self.data_dir = config_dict["data_dir"]
        self.snli_dir = config_dict["snli_dir"]
        self.jsonl_split_files = config_dict["jsonl_split_files"]
        self.tf_split_files = config_dict["tf_split_files"]
        self.label_to_int = config_dict["label_to_int"]
        self.vocab_file = config_dict["vocab_file"]
        self.max_vocab_size = config_dict["max_vocab_size"]
        self.glove_dir = config_dict["glove_dir"]
        self.log_dir = config_dict["log_dir"]

        self.int_to_label = {int_: label for label,
                             int_ in self.label_to_int.items()}
        self.model = None

    def add_model_config(self, model):
        self.model = model

    def add_train_config(self, train):
        self.train = train


class ModelConfig(ConfigBase):

    def __init__(self, *args, **kwargs):
        super(ModelConfig, self).__init__(*args, **kwargs)
        config_dict = self.config_dict
        self.wvec_dim = config_dict["wvec_dim"]
        self.name = config_dict["name"]


class TrainConfig(ConfigBase):
    def __init__(self, *args, **kwargs):
        super(TrainConfig, self).__init__(*args, **kwargs)
        config_dict = self.config_dict
        self.batch_size = config_dict["batch_size"]
        self.num_threads = config_dict["num_threads"]
        self.capacity = config_dict["capacity"]
        self.min_after_dequeue = config_dict["min_after_dequeue"]

        self.train_op = config_dict["train_op"]
        self.learning_rate = config_dict["learning_rate"]
        self.epsilon = config_dict.get("epsilon")
        self.rho = config_dict.get("rho")
