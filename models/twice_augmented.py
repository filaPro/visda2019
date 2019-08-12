import tensorflow as tf
from utils import Preprocessor, make_domain_dataset


def make_dataset(
    source_paths, source_labels, source_config,
    target_paths, target_labels, target_config,
    batch_size
):
    source_preprocessor = Preprocessor(source_config)
    target_preprocessor = Preprocessor(target_config)
    datasets = []
    for paths, labels in zip(source_paths, source_labels):
        datasets.append(make_domain_dataset(paths, labels, source_preprocessor, batch_size, None))
    datasets.append(make_domain_dataset(target_paths, target_labels, target_preprocessor, batch_size, 42))
    datasets.append(make_domain_dataset(target_paths, target_labels, target_preprocessor, batch_size, 42))
    return tf.data.Dataset.zip(tuple(datasets)).repeat()


class TwiceAugmentedTrainStep:
    def __init__(self, build_model_lambda, domains, n_frozen_layers, learning_rate, loss_weight):
        pass

    @tf.function
    def train(self, batch):
        pass
