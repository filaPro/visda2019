import os
import re
import tensorflow as tf
from functools import partial

from tester import Tester
from models import SourceTestStep, SelfEnsemblingPreprocessor, build_backbone
from utils import N_CLASSES, make_domain_dataset

DATA_PATH = '/content/data/tfrecords_links'
LOG_PATH = '/content/data/logs/tmp-mix-match'
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
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])


build_top_lambda = partial(build_top, n_classes=N_CLASSES)
build_backbone_lambda = partial(build_backbone, name=BACKBONE_NAME, size=IMAGE_SIZE)
test_preprocessor = SelfEnsemblingPreprocessor((COMPLEX_CONFIG, COMPLEX_CONFIG, COMPLEX_CONFIG, COMPLEX_CONFIG))

test_dataset = make_domain_dataset(
    path=os.path.join(DATA_PATH, 'target', 'test'),
    preprocessor=test_preprocessor,
    n_processes=N_PROCESSES
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
pattern = re.compile('[^a-zA-Z]')
paths, predictions = zip(*sorted(zip(paths, predictions), key=lambda t: re.sub(pattern, '|', t[0].decode().lower())))

with open(os.path.join(LOG_PATH, 'result.txt'), 'w') as file:
    for path, prediction in zip(paths, predictions):
        path = '/'.join(path.decode().split('/')[-3:])
        file.write(f'{path} {prediction}\n')
