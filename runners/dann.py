import tensorflow as tf
from functools import partial

from trainer import Trainer
from models import DannTrainStep, GradientReverse, SelfEnsemblingPreprocessor, build_backbone
from utils import DOMAINS, N_CLASSES, make_multi_source_dataset, get_time_string, copy_runner
from preprocessor import Preprocessor

DATA_PATH = '/content/data'
LOG_PATH = f'/content/logs/{get_time_string()}-dann'
N_GPUS = 1
BATCH_SIZE = 32
IMAGE_SIZE = 224
BACKBONE_NAME = 'mobile_net_v2'
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


source_domains = DOMAINS[:3]
target_domain = DOMAINS[3]
build_backbone_lambda = partial(build_backbone, name=BACKBONE_NAME, size=IMAGE_SIZE)
build_top_lambda = partial(build_top, n_classes=N_CLASSES)
build_discriminator_lambda = partial(build_discriminator, n_domains=len(source_domains) + 1)
preprocessor = Preprocessor(CONFIG)
test_preprocessor = SelfEnsemblingPreprocessor((CONFIG,))

train_dataset = make_multi_source_dataset(
    source_domains=source_domains,
    source_phase='all',
    source_preprocessor=preprocessor,
    source_batch_size=BATCH_SIZE,
    target_domain=target_domain,
    target_phase='all',
    target_preprocessor=preprocessor,
    target_batch_size=BATCH_SIZE,
    path=DATA_PATH
)
copy_runner(__file__, LOG_PATH)
build_train_step_lambda = partial(
    DannTrainStep,
    build_backbone_lambda=build_backbone_lambda,
    build_top_lambda=build_top_lambda,
    build_discriminator_lambda=build_discriminator_lambda,
    domains=source_domains + (target_domain,),
    learning_rate=0.0001,
    loss_weight=.03,
    batch_size=BATCH_SIZE
)
Trainer(
    build_train_step_lambda=build_train_step_lambda,
    n_epochs=1,
    n_train_iterations=1000,
    log_path=LOG_PATH,
    restore_model_flag=True,
    restore_optimizer_flag=True,
    single_gpu_flag=N_GPUS == 1
)(train_dataset)
