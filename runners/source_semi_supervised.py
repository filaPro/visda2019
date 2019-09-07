import tensorflow as tf
from functools import partial

from trainer import Trainer
from models import SourceTrainStep, build_backbone
from utils import DOMAINS, N_CLASSES, make_semi_supervised_dataset, copy_runner, get_time_string
from preprocessor import Preprocessor

DATA_PATH = '/content/data'
LOG_PATH = f'/content/logs/{get_time_string()}-source'
N_GPUS = 1
BATCH_SIZE = 8
IMAGE_SIZE = 224
BACKBONE_NAME = 'efficient_net_b5'
CONFIG = [
    {'method': 'keras', 'mode': 'torch'},
    {'method': 'resize', 'height': IMAGE_SIZE, 'width': IMAGE_SIZE, 'n_channels': 3}
]
COMPLEX_CONFIG = [
    {'method': 'resize', 'height': 256, 'width': 256},
    {'method': 'random_flip_left_right'},
    {'method': 'keras', 'mode': 'torch'},
    {
        'method': 'random_size_crop',
        'min_height': 160,
        'max_height': 256,
        'min_width': 160,
        'max_width': 256,
        'n_channels': 3
    },
    {'method': 'resize', 'height': IMAGE_SIZE, 'width': IMAGE_SIZE}
]


def build_top(n_classes):
    return tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(input_shape=(7, 7, 2048)),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])


source_domain = DOMAINS[0]
target_domain = DOMAINS[3]
build_top_lambda = partial(build_top, n_classes=N_CLASSES)
build_backbone_lambda = partial(build_backbone, name=BACKBONE_NAME, size=IMAGE_SIZE)
preprocessor = Preprocessor(CONFIG)
complex_preprocessor = Preprocessor(COMPLEX_CONFIG)

train_dataset = make_semi_supervised_dataset(
    source_domain=source_domain,
    source_phase='all',
    source_preprocessor=complex_preprocessor,
    source_batch_size=BATCH_SIZE,
    target_domain=target_domain,
    labeled_preprocessor=complex_preprocessor,
    labeled_batch_size=BATCH_SIZE,
    unlabeled_preprocessor=preprocessor,
    unlabeled_batch_size=BATCH_SIZE,
    path=DATA_PATH
)

copy_runner(__file__, LOG_PATH)
build_train_step_lambda = partial(
    SourceTrainStep,
    build_backbone_lambda=build_backbone_lambda,
    build_top_lambda=build_top_lambda,
    domains=('source', 'labeled', 'unlabeled'),
    learning_rate=.0001,
    batch_size=BATCH_SIZE
)
Trainer(
    build_train_step_lambda=build_train_step_lambda,
    n_epochs=1000,
    n_train_iterations=1000,
    log_path=LOG_PATH,
    restore_model_flag=False,
    restore_optimizer_flag=False,
    single_gpu_flag=N_GPUS == 1
)(train_dataset)
