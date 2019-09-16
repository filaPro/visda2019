import tensorflow as tf
from functools import partial

from trainer import Trainer
from models import SelfEnsemblingPreprocessor, MixMatchV3TrainStep, efficient_net_b4
from utils import DOMAINS, N_CLASSES, make_multi_source_dataset, get_time_string, copy_runner
from preprocessor import Preprocessor

DATA_PATH = '/content/data'
LOG_PATH = f'/content/logs/tmp-mix-match'  # TODO: <-
LOCAL_BATCH_SIZE = 16  # TODO: <-
N_GPUS = 1
IMAGE_SIZE = 128  # TODO: <-
BACKBONE_NAME = efficient_net_b4  # TODO: <-
CONFIG = [
    {'method': 'keras', 'mode': 'torch'},
    {'method': 'resize', 'height': IMAGE_SIZE, 'width': IMAGE_SIZE}
]
COMPLEX_CONFIG = [
    {'method': 'resize', 'height': 156, 'width': 156},  # TODO: <-
    {'method': 'random_flip_left_right'},
    {'method': 'keras', 'mode': 'torch'},
    {'method': 'random_crop', 'height': IMAGE_SIZE, 'width': IMAGE_SIZE, 'n_channels': 3}
]


def build_top(n_classes):
    return tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(input_shape=(7, 7, 1792)),  # TODO: <-
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])


source_domains = DOMAINS[:3]
target_domain = DOMAINS[3]
build_top_lambda = partial(build_top, n_classes=N_CLASSES)
build_backbone_lambda = partial(BACKBONE_NAME, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
source_preprocessor = Preprocessor(COMPLEX_CONFIG)
target_preprocessor = SelfEnsemblingPreprocessor((COMPLEX_CONFIG, COMPLEX_CONFIG))

train_dataset = make_multi_source_dataset(
    source_domains=source_domains,
    source_phase='all',
    source_preprocessor=source_preprocessor,
    source_batch_size=LOCAL_BATCH_SIZE * N_GPUS,
    target_domain=target_domain,
    target_phase='all',
    target_preprocessor=target_preprocessor,
    target_batch_size=LOCAL_BATCH_SIZE * N_GPUS,
    path=DATA_PATH
)

copy_runner(__file__, LOG_PATH)
build_train_step_lambda = partial(
    MixMatchV3TrainStep,
    build_backbone_lambda=build_backbone_lambda,
    build_top_lambda=build_top_lambda,
    learning_rate=.0001,
    source_domains=(0, 0, 0),  # range(1, len(source_domains) + 1),  # TODO: <-
    loss_weight=10.,  # TODO: <-
    temperature=.5,
    alpha=.75,
    local_batch_size=LOCAL_BATCH_SIZE,
    global_batch_size=LOCAL_BATCH_SIZE * N_GPUS
)
Trainer(
    build_train_step_lambda,
    n_epochs=1000,
    n_train_iterations=1000,
    log_path=LOG_PATH,
    restore_model_flag=False,
    restore_optimizer_flag=False,
    single_gpu_flag=N_GPUS == 1
)(train_dataset)
