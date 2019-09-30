import tensorflow as tf
from functools import partial

from trainer import Trainer
from models import MixMatchDannTrainStep, SelfEnsemblingPreprocessor, build_backbone, GradientReverse
from utils import DOMAINS, N_CLASSES, make_combined_multi_source_dataset, get_time_string, copy_runner
from preprocessor import Preprocessor

DATA_PATH = '/content/data'
LOG_PATH = f'/content/logs/{get_time_string()}-mix-match-dann'
LOCAL_BATCH_SIZE = 6
N_GPUS = 1
IMAGE_SIZE = 224
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


def build_discriminator(n_domains):
    return tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(input_shape=(7, 7, 2048)),
        GradientReverse(1.),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(n_domains, activation='softmax')
    ])


source_domains = DOMAINS[:3]
target_domain = DOMAINS[3]
build_top_lambda = partial(build_top, n_classes=N_CLASSES)
build_backbone_lambda = partial(build_backbone, name=BACKBONE_NAME, size=IMAGE_SIZE)
build_discriminator_lambda = partial(build_discriminator, n_domains=len(source_domains) + 1)
source_preprocessor = Preprocessor(COMPLEX_CONFIG)
target_preprocessor = SelfEnsemblingPreprocessor((COMPLEX_CONFIG, COMPLEX_CONFIG))

train_dataset = make_combined_multi_source_dataset(
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
    MixMatchDannTrainStep,
    build_backbone_lambda=build_backbone_lambda,
    build_top_lambda=build_top_lambda,
    build_discriminator_lambda=build_discriminator_lambda,
    n_domains=len(source_domains) + 1,
    learning_rate=.0001,
    loss_weight=333.,
    domain_loss_weight=.03,
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
