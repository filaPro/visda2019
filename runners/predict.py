import os
import tensorflow as tf
from functools import partial

from tester import Tester
from models import SourceTestStep, SelfEnsemblingPreprocessor, build_backbone
from utils import DOMAINS, N_CLASSES, make_domain_dataset, list_tfrecords

DATA_PATH = '/content/data'
LOG_PATH = '/content/logs/tmp-source'
BATCH_SIZE = 128
IMAGE_SIZE = 224
N_PROCESSES = 16
BACKBONE_NAME = 'efficient_net_b5'
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


target_domain = DOMAINS[5]
build_top_lambda = partial(build_top, n_classes=N_CLASSES)
build_backbone_lambda = partial(build_backbone, name=BACKBONE_NAME, size=IMAGE_SIZE)
test_preprocessor = SelfEnsemblingPreprocessor((COMPLEX_CONFIG, COMPLEX_CONFIG, COMPLEX_CONFIG, COMPLEX_CONFIG))

test_paths = list_tfrecords(
    path=os.path.join(DATA_PATH, 'multi_source', 'tfrecords'),
    domain=target_domain,
    phase='test'
)
test_dataset = make_domain_dataset(
    paths=test_paths,
    preprocessor=test_preprocessor
).batch(BATCH_SIZE)
build_test_step_lambda = partial(
    SourceTestStep,
    build_backbone_lambda=build_backbone_lambda,
    build_top_lambda=build_top_lambda
)
paths, predictions = Tester(
    build_test_step_lambda=build_test_step_lambda,
    log_path=LOG_PATH
)(test_dataset)

with open(os.path.join(LOG_PATH, 'result.txt'), 'w') as file:
    for path, prediction in zip(paths, predictions):
        file.write(f'{path.decode()} {prediction}\n')
