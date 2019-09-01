import os
import tensorflow as tf
from functools import partial

from trainer import Trainer
from tester import Tester
from models import DannTrainStep, SourceTestStep, GradientReverse, SelfEnsemblingPreprocessor, build_backbone
from utils import (
    DOMAINS, N_CLASSES, make_dataset, make_domain_dataset, get_time_string, copy_runner
)
from preprocessor import Preprocessor

DATA_PATH = '/content/data/tfrecords_links'
LOG_PATH = f'/content/data/logs/tmp-dann'  # TODO: <-
N_GPUS = 1
BATCH_SIZE = 32
IMAGE_SIZE = 224
N_PROCESSES = 16
BACKBONE_NAME = 'mobile_net_v2'
DOMAINS = DOMAINS[:4]
print(f'{DOMAINS[:-1]} -> {DOMAINS[-1]}')
CONFIG = [
    {'method': 'keras', 'mode': 'tf'},
    {'method': 'resize', 'height': IMAGE_SIZE, 'width': IMAGE_SIZE}
]


def build_top(n_classes):
    return tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(input_shape=(7, 7, 1280)),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])


def build_discriminator(n_domains):
    return tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(input_shape=(7, 7, 1280)),
        GradientReverse(1.),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(n_domains, activation='softmax')
    ])


build_backbone_lambda = partial(build_backbone, name=BACKBONE_NAME, size=IMAGE_SIZE)
build_top_lambda = partial(build_top, n_classes=N_CLASSES)
build_discriminator_lambda = partial(build_discriminator, n_domains=len(DOMAINS))
preprocessor = Preprocessor(CONFIG)
test_preprocessor = SelfEnsemblingPreprocessor((CONFIG,))

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
    DannTrainStep,
    build_backbone_lambda=build_backbone_lambda,
    build_top_lambda=build_top_lambda,
    build_discriminator_lambda=build_discriminator_lambda,
    domains=DOMAINS,
    learning_rate=0.0001,
    loss_weight=.03,
    batch_size=BATCH_SIZE
)
Trainer(
    build_train_step_lambda=build_train_step_lambda,
    n_epochs=10,
    n_train_iterations=1000,
    log_path=LOG_PATH,
    restore_model_flag=True,
    restore_optimizer_flag=True,
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
