import os
import tensorflow as tf
from functools import partial

from trainer import Trainer
from tester import Tester
from models import SourceTrainStep, SourceTestStep, build_backbone
from utils import (
    DOMAINS, N_CLASSES, make_dataset, make_domain_dataset, get_time_string, copy_runner
)
from preprocessor import Preprocessor

DATA_PATH = '/content/data/tfrecords_links'
LOG_PATH = f'/content/data/logs/{get_time_string()}-source'
BATCH_SIZE = 32
IMAGE_SIZE = 224
BACKBONE_NAME = 'mobilenet_v2'
CONFIG = [
    {'method': 'keras', 'mode': 'tf'},
    {'method': 'resize', 'height': IMAGE_SIZE, 'width': IMAGE_SIZE}
]


def build_top(n_classes):
    return tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(input_shape=(7, 7, 1280)),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(n_classes, input_shape=(4096,), activation='softmax')
    ])


build_top_lambda = partial(build_top, n_classes=N_CLASSES)
build_backbone_lambda = partial(build_backbone, name=BACKBONE_NAME, size=IMAGE_SIZE)
preprocessor = Preprocessor(CONFIG)

train_dataset = make_dataset(
    source_path=os.path.join(DATA_PATH, 'source', 'train'),
    source_preprocessor=preprocessor,
    target_path=os.path.join(DATA_PATH, 'target', 'all'),
    target_preprocessor=preprocessor,
    domains=DOMAINS,
    batch_size=BATCH_SIZE
)
validate_dataset = make_dataset(
    source_path=os.path.join(DATA_PATH, 'source', 'test'),
    source_preprocessor=preprocessor,
    target_path=os.path.join(DATA_PATH, 'target', 'all'),
    target_preprocessor=preprocessor,
    domains=DOMAINS,
    batch_size=BATCH_SIZE
)

build_train_step_lambda = partial(
    SourceTrainStep,
    build_backbone_lambda=build_backbone_lambda,
    build_top_lambda=build_top_lambda,
    domains=DOMAINS,
    freeze_backbone_flag=True,
    backbone_training_flag=False,
    learning_rate=.001,
    batch_size=BATCH_SIZE
)
trainer = Trainer(
    build_train_step_lambda=build_train_step_lambda,
    n_epochs=1,
    n_train_iterations=500,
    n_validate_iterations=50,
    log_path=LOG_PATH,
    restore_model_flag=False,
    restore_optimizer_flag=False
)
copy_runner(__file__, LOG_PATH)
trainer(train_dataset, validate_dataset)

build_train_step_lambda = partial(
    SourceTrainStep,
    build_backbone_lambda=build_backbone_lambda,
    build_top_lambda=build_top_lambda,
    domains=DOMAINS,
    freeze_backbone_flag=True,
    backbone_training_flag=False,
    learning_rate=.001,
    batch_size=BATCH_SIZE
)
trainer = Trainer(
    build_train_step_lambda=build_train_step_lambda,
    n_epochs=1,
    n_train_iterations=500,
    n_validate_iterations=50,
    log_path=LOG_PATH,
    restore_model_flag=True,
    restore_optimizer_flag=True
)
trainer(train_dataset, validate_dataset)

test_dataset = make_domain_dataset(
    path=os.path.join(DATA_PATH, 'target', 'all'),
    preprocessor=preprocessor,
    batch_size=BATCH_SIZE
)
build_test_step_lambda = partial(
    SourceTestStep,
    build_backbone_lambda=build_backbone_lambda,
    build_top_lambda=build_top_lambda
)
tester = Tester(build_test_step_lambda=build_test_step_lambda, log_path=LOG_PATH)
tester(test_dataset)
