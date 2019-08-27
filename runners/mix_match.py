import os
import tensorflow as tf
from functools import partial

from trainer import Trainer
from tester import Tester
from models import MixMatchTrainStep, SelfEnsemblingPreprocessor, build_backbone
from utils import (
    DOMAINS, N_CLASSES, make_dataset, make_domain_dataset, get_time_string, copy_runner
)
from preprocessor import Preprocessor

DATA_PATH = '/content/data/tfrecords_links'
LOG_PATH = f'/content/data/logs/{get_time_string()}-mix-match'
BATCH_SIZE = 9
IMAGE_SIZE = 224
N_PROCESSES = 16
BACKBONE_NAME = 'efficient_net_b4'
CONFIG = [
    {'method': 'keras', 'mode': 'torch'},
    {'method': 'resize', 'height': IMAGE_SIZE, 'width': IMAGE_SIZE}
]
COMPLEX_CONFIG = [
    {'method': 'resize', 'height': 256, 'width': 256},
    {'method': 'random_flip_left_right'},
    {'method': 'keras', 'mode': 'torch'},
    {'method': 'random_crop', 'height': IMAGE_SIZE, 'width': IMAGE_SIZE, 'n_channels': 3}
]


def build_top(n_classes):
    return tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(input_shape=(7, 7, 1792)),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])


build_top_lambda = partial(build_top, n_classes=N_CLASSES)
build_backbone_lambda = partial(build_backbone, name=BACKBONE_NAME, size=IMAGE_SIZE)
source_preprocessor = Preprocessor(COMPLEX_CONFIG)
target_preprocessor = SelfEnsemblingPreprocessor(first_config=COMPLEX_CONFIG, second_config=CONFIG)

train_dataset = make_dataset(
    source_path=os.path.join(DATA_PATH, 'source', 'all'),
    source_preprocessor=source_preprocessor,
    source_batch_size=BATCH_SIZE // 3,
    target_path=os.path.join(DATA_PATH, 'target', 'all'),
    target_preprocessor=target_preprocessor,
    target_batch_size=BATCH_SIZE,
    domains=DOMAINS,
    n_processes=N_PROCESSES
)

copy_runner(__file__, LOG_PATH)
build_train_step_lambda = partial(
    MixMatchTrainStep,
    build_backbone_lambda=build_backbone_lambda,
    build_top_lambda=build_top_lambda,
    backbone_learning_rate=.0001,
    top_learning_rate=.0001,
    loss_weight=100.,
    temperature=.5,
    alpha=.75,
    batch_size=BATCH_SIZE
)
Trainer(
    build_train_step_lambda,
    n_epochs=1000,
    n_train_iterations=1000,
    log_path=LOG_PATH,
    restore_model_flag=False,
    restore_optimizer_flag=False,
    single_gpu_flag=True
)(train_dataset)
