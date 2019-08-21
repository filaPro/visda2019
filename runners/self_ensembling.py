import os
import tensorflow as tf
from functools import partial

from trainer import Trainer
from tester import Tester
from models import SelfEnsemblingTrainStep, SelfEnsemblingTestStep, SelfEnsemblingPreprocessor, build_backbone
from utils import (
    DOMAINS, N_CLASSES, make_dataset, make_domain_dataset, get_time_string, copy_runner
)
from preprocessor import Preprocessor

DATA_PATH = '/content/data/tfrecords_links'
LOG_PATH = f'/content/data/logs/{get_time_string()}-self-ensembling'
BATCH_SIZE = 4
IMAGE_SIZE = 224
BACKBONE_NAME = 'efficient_net_b5'
CONFIG = [
    {'method': 'keras', 'mode': 'torch'},
    {'method': 'resize', 'height': IMAGE_SIZE, 'width': IMAGE_SIZE}
]
TWICE_CONFIG = [
    {'method': 'resize', 'height': 256, 'width': 256},
    {'method': 'random_flip_left_right'},
    {'method': 'random_contrast', 'delta': .1},
    {'method': 'random_brightness', 'delta': .1},
    # {'method': 'random_rotate', 'angle': 20.},
    {'method': 'keras', 'mode': 'torch'},
    {'method': 'resize', 'height': IMAGE_SIZE, 'width': IMAGE_SIZE}
]


def build_top(n_classes):
    return tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(input_shape=(7, 7, 2048)),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(n_classes, input_shape=(4096,), activation='softmax')
    ])


build_top_lambda = partial(build_top, n_classes=N_CLASSES)
build_backbone_lambda = partial(build_backbone, name=BACKBONE_NAME, size=IMAGE_SIZE)
preprocessor = Preprocessor(CONFIG)
twice_preprocessor = SelfEnsemblingPreprocessor(first_config=TWICE_CONFIG, second_config=CONFIG)

train_dataset = make_dataset(
    source_path=os.path.join(DATA_PATH, 'source', 'all'),
    source_preprocessor=preprocessor,
    target_path=os.path.join(DATA_PATH, 'target', 'all'),
    target_preprocessor=twice_preprocessor,
    domains=DOMAINS,
    batch_size=BATCH_SIZE
)

copy_runner(__file__, LOG_PATH)
build_train_step_lambda = partial(
    SelfEnsemblingTrainStep,
    build_backbone_lambda=build_backbone_lambda,
    build_top_lambda=build_top_lambda,
    domains=DOMAINS,
    backbone_training_flag=False,
    top_learning_rate=.00001,
    backbone_learning_rate=.00001,
    loss_weight=1.,
    decay=.99,
    batch_size=BATCH_SIZE
)
Trainer(
    build_train_step_lambda,
    n_epochs=5,
    n_train_iterations=1000,
    n_validate_iterations=0,
    log_path=LOG_PATH,
    restore_model_flag=True,
    restore_optimizer_flag=False,
    single_gpu_flag=True
)(train_dataset, None)

test_dataset = make_domain_dataset(
    path=os.path.join(DATA_PATH, 'target', 'all'),
    preprocessor=preprocessor,
    batch_size=BATCH_SIZE
)
build_test_step_lambda = partial(
    SelfEnsemblingTestStep,
    build_backbone_lambda=build_backbone_lambda,
    build_top_lambda=build_top_lambda
)
Tester(build_test_step_lambda=build_test_step_lambda, log_path=LOG_PATH)(test_dataset)
