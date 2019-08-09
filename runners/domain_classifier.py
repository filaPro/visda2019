import tensorflow as tf
from functools import partial

from trainer import Trainer
from models import DomainClassifierTrainStep, get_backbone
from utils import DOMAINS, N_CLASSES, read_paths_and_labels, make_dataset

RAW_DATA_PATH = '/content/data/raw'
LOG_PATH = '/content/data/logs'
BATCH_SIZE = 32
IMAGE_SIZE = 96
BACKBONE_NAME = 'vgg19'
TRAIN_CONFIG = [
    {'method': 'keras', 'mode': 'caffe'},
    {'method': 'resize', 'height': 128, 'width': 128},
    {'method': 'random_flip_left_right'},
    {'method': 'random_contrast', 'delta': .2},
    {'method': 'random_brightness', 'delta': .2},
    {'method': 'random_crop', 'height': IMAGE_SIZE, 'width': IMAGE_SIZE, 'n_channels': 3}
]
CONFIG = [
    {'method': 'keras', 'mode': 'caffe'},
    {'method': 'resize', 'height': IMAGE_SIZE, 'width': IMAGE_SIZE}
]


def build_discriminator(n_domains):
    return tf.keras.Sequential([
        tf.keras.layers.Dropout(.99, input_shape=(8192,)),
        tf.keras.layers.Dense(n_domains, activation='softmax')
    ])


def build_generator(image_size, name):
    return tf.keras.Sequential([
        get_backbone(name)(
            input_shape=(image_size, image_size, 3),
            include_top=False,
            weights='imagenet'
        ),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(8192, activation='relu')
    ])


def build_classifier(n_classes):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(n_classes, input_shape=(8192,), activation='softmax')
    ])


build_discriminator_lambda = partial(build_discriminator, n_domains=len(DOMAINS))
build_generator_lambda = partial(build_generator, image_size=IMAGE_SIZE, name=BACKBONE_NAME)
build_classifier_lambda = partial(build_classifier, n_classes=N_CLASSES)

paths_and_labels = read_paths_and_labels(RAW_DATA_PATH, DOMAINS, 3)
target_paths = paths_and_labels['target']['train']['paths'] + paths_and_labels['target']['test']['paths']
target_labels = paths_and_labels['target']['train']['labels'] + paths_and_labels['target']['test']['labels']
train_dataset = iter(make_dataset(
    source_paths=paths_and_labels['source']['train']['paths'],
    source_labels=paths_and_labels['source']['train']['labels'],
    source_config=TRAIN_CONFIG,
    target_paths=target_paths,
    target_labels=target_labels,
    target_config=CONFIG,
    batch_size=BATCH_SIZE,
))
validate_dataset = iter(make_dataset(
    source_paths=paths_and_labels['source']['test']['paths'],
    source_labels=paths_and_labels['source']['test']['labels'],
    source_config=CONFIG,
    target_paths=target_paths,
    target_labels=target_labels,
    target_config=CONFIG,
    batch_size=BATCH_SIZE,
))

train_step = DomainClassifierTrainStep(
    build_discriminator_lambda=build_discriminator_lambda,
    build_generator_lambda=build_generator_lambda,
    build_classifier_lambda=build_classifier_lambda,
    domains=DOMAINS,
    n_frozen_layers=100,
    learning_rate=0.001,
    loss_weight=.0,
)
trainer = Trainer(
    train_step=train_step,
    n_iterations=200,
    n_log_iterations=100,
    n_save_iterations=200,
    n_validate_iterations=10,
    log_path=LOG_PATH,
    restore_model_flag=False,
    restore_optimizer_flag=False
)
trainer(train_dataset, validate_dataset)

train_step = DomainClassifierTrainStep(
    build_discriminator_lambda=build_discriminator_lambda,
    build_generator_lambda=build_generator_lambda,
    build_classifier_lambda=build_classifier_lambda,
    domains=DOMAINS,
    n_frozen_layers=0,
    learning_rate=0.0001,
    loss_weight=.001,
)
trainer = Trainer(
    train_step=train_step,
    n_iterations=10000,
    n_log_iterations=100,
    n_save_iterations=0,
    n_validate_iterations=10,
    log_path=LOG_PATH,
    restore_model_flag=True,
    restore_optimizer_flag=False
)
trainer(train_dataset, validate_dataset)
