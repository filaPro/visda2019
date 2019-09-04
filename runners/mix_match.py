import os
import tensorflow as tf
from functools import partial

from trainer import Trainer
from tester import Tester
from models import MixMatchTrainStep, SourceTestStep, SelfEnsemblingPreprocessor, build_backbone
from utils import (
    DOMAINS, N_CLASSES, make_multi_source_dataset, make_domain_dataset, get_time_string, copy_runner
)
from preprocessor import Preprocessor

DATA_PATH = '/content/data'
LOG_PATH = f'/content/logs/{get_time_string()}-mix-match'
LOCAL_BATCH_SIZE = 6
N_GPUS = 1
IMAGE_SIZE = 224
N_PROCESSES = 16
BACKBONE_NAME = 'efficient_net_b5'
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
        tf.keras.layers.GlobalAveragePooling2D(input_shape=(7, 7, 2048)),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])


source_domains = DOMAINS[:3]
target_domain = DOMAINS[3]
build_top_lambda = partial(build_top, n_classes=N_CLASSES)
build_backbone_lambda = partial(build_backbone, name=BACKBONE_NAME, size=IMAGE_SIZE)
source_preprocessor = Preprocessor(COMPLEX_CONFIG)
target_preprocessor = SelfEnsemblingPreprocessor((COMPLEX_CONFIG, COMPLEX_CONFIG))
test_preprocessor = SelfEnsemblingPreprocessor((COMPLEX_CONFIG, COMPLEX_CONFIG, COMPLEX_CONFIG, COMPLEX_CONFIG))

train_dataset = make_multi_source_dataset(
    source_domains=source_domains,
    source_phase='all',
    source_preprocessor=source_preprocessor,
    source_batch_size=LOCAL_BATCH_SIZE * N_GPUS // len(source_domains),
    target_domain=target_domain,
    target_phase='all',
    target_preprocessor=target_preprocessor,
    target_batch_size=LOCAL_BATCH_SIZE * N_GPUS,
    path=DATA_PATH,
    n_processes=N_PROCESSES
)

copy_runner(__file__, LOG_PATH)
build_train_step_lambda = partial(
    MixMatchTrainStep,
    build_backbone_lambda=build_backbone_lambda,
    build_top_lambda=build_top_lambda,
    learning_rate=.0001,
    loss_weight=1000.,
    temperature=.5,
    alpha=.75,
    local_batch_size=LOCAL_BATCH_SIZE,
    global_batch_size=LOCAL_BATCH_SIZE * N_GPUS
)
Trainer(
    build_train_step_lambda,
    n_epochs=1,
    n_train_iterations=100,
    log_path=LOG_PATH,
    restore_model_flag=False,
    restore_optimizer_flag=False,
    single_gpu_flag=N_GPUS == 1
)(train_dataset)

test_dataset = make_domain_dataset(
    paths=os.path.join(DATA_PATH, 'multi_source', 'tfrecords', f'{target_domain}_test*'),
    preprocessor=test_preprocessor,
    n_processes=N_PROCESSES
).batch(LOCAL_BATCH_SIZE)
build_test_step_lambda = partial(
    SourceTestStep,
    build_backbone_lambda=build_backbone_lambda,
    build_top_lambda=build_top_lambda
)
Tester(build_test_step_lambda=build_test_step_lambda, log_path=LOG_PATH)(test_dataset)
