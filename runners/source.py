import os
import tensorflow as tf
from functools import partial

from trainer import Trainer
from tester import Tester
from models import SourceTrainStep, SourceTestStep, SelfEnsemblingPreprocessor, build_backbone
from utils import (
    DOMAINS, N_CLASSES, make_dataset, make_domain_dataset, get_time_string, copy_runner
)
from preprocessor import Preprocessor

DATA_PATH = '/content/data/tfrecords_links'
LOG_PATH = f'/content/data/logs/{get_time_string()}-source'
N_GPUS = 1
BATCH_SIZE = 33
IMAGE_SIZE = 224
N_PROCESSES = 16
BACKBONE_NAME = 'mobile_net_v2'
CONFIG = [
    {'method': 'keras', 'mode': 'tf'},
    {'method': 'resize', 'height': IMAGE_SIZE, 'width': IMAGE_SIZE, 'n_channels': 3}
]


def build_top(n_classes):
    return tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(input_shape=(7, 7, 1280)),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])


build_top_lambda = partial(build_top, n_classes=N_CLASSES)
build_backbone_lambda = partial(build_backbone, name=BACKBONE_NAME, size=IMAGE_SIZE)
preprocessor = Preprocessor(CONFIG)
test_preprocessor = SelfEnsemblingPreprocessor((CONFIG, CONFIG))

train_dataset = make_dataset(
    source_path=os.path.join(DATA_PATH, 'source', 'all'),
    source_preprocessor=preprocessor,
    source_batch_size=BATCH_SIZE,
    target_path=os.path.join(DATA_PATH, 'target', 'all'),
    target_preprocessor=preprocessor,
    target_batch_size=BATCH_SIZE,
    domains=DOMAINS,
    n_processes=N_PROCESSES
)

copy_runner(__file__, LOG_PATH)
build_train_step_lambda = partial(
    SourceTrainStep,
    build_backbone_lambda=build_backbone_lambda,
    build_top_lambda=build_top_lambda,
    domains=DOMAINS,
    learning_rate=.0001,
    batch_size=BATCH_SIZE
)
Trainer(
    build_train_step_lambda=build_train_step_lambda,
    n_epochs=10,
    n_train_iterations=1000,
    log_path=LOG_PATH,
    restore_model_flag=False,
    restore_optimizer_flag=False,
    single_gpu_flag=N_GPUS == 1
)(train_dataset)

test_dataset = make_domain_dataset(
    path=os.path.join(DATA_PATH, 'target', 'all'),
    preprocessor=test_preprocessor,
    n_processes=N_PROCESSES
).batch(BATCH_SIZE)
build_test_step_lambda = partial(
    SourceTestStep,
    build_backbone_lambda=build_backbone_lambda,
    build_top_lambda=build_top_lambda
)
Tester(build_test_step_lambda=build_test_step_lambda, log_path=LOG_PATH)(test_dataset)
