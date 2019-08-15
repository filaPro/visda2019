import tensorflow as tf
from functools import partial

from trainer import Trainer
from tester import Tester
from models import TwiceAugmentedTrainStep, TwiceAugmentedPreprocessor, SourceTestStep, build_backbone
from utils import DOMAINS, N_CLASSES, read_paths_and_labels, make_dataset, make_domain_dataset
from preprocessor import Preprocessor

RAW_DATA_PATH = '/content/data/raw'
LOG_PATH = '/content/data/logs'
BATCH_SIZE = 24
IMAGE_SIZE = 224
BACKBONE_NAME = 'mobilenet_v2'
CONFIG = [
    {'method': 'keras', 'mode': 'tf'},
    {'method': 'resize', 'height': IMAGE_SIZE, 'width': IMAGE_SIZE}
]
TWICE_CONFIG = [
    {'method': 'resize', 'height': 256, 'width': 256},
    # {'method': 'random_flip_left_right'},
    # {'method': 'random_contrast', 'delta': .1},
    # {'method': 'random_brightness', 'delta': .1},
    # {'method': 'random_rotate', 'angle': 20.},
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
twice_preprocessor = TwiceAugmentedPreprocessor(first_config=CONFIG, second_config=TWICE_CONFIG)

paths_and_labels = read_paths_and_labels(RAW_DATA_PATH, DOMAINS)
target_paths = paths_and_labels['target']['train']['paths'] + paths_and_labels['target']['test']['paths']
target_labels = paths_and_labels['target']['train']['labels'] + paths_and_labels['target']['test']['labels']
train_dataset = iter(make_dataset(
    source_paths=paths_and_labels['source']['train']['paths'],
    source_labels=paths_and_labels['source']['train']['labels'],
    source_preprocessor=preprocessor,
    target_paths=target_paths,
    target_labels=target_labels,
    target_preprocessor=twice_preprocessor,
    batch_size=BATCH_SIZE
))

train_step = TwiceAugmentedTrainStep(
    build_backbone_lambda=build_backbone_lambda,
    build_top_lambda=build_top_lambda,
    domains=DOMAINS,
    freeze_backbone_flag=True,
    backbone_training_flag=False,
    learning_rate=0.001,
    loss_weight=1.
)
trainer = Trainer(
    train_step=train_step,
    n_iterations=500,
    n_log_iterations=100,
    n_save_iterations=500,
    n_validate_iterations=0,
    log_path=LOG_PATH,
    restore_model_flag=False,
    restore_optimizer_flag=False
)
trainer(train_dataset, None)

train_step = TwiceAugmentedTrainStep(
    build_backbone_lambda=build_backbone_lambda,
    build_top_lambda=build_top_lambda,
    domains=DOMAINS,
    freeze_backbone_flag=False,
    backbone_training_flag=False,
    learning_rate=0.0001,
    loss_weight=1.
)
trainer = Trainer(
    train_step=train_step,
    n_iterations=2000,
    n_log_iterations=100,
    n_save_iterations=2000,
    n_validate_iterations=0,
    log_path=LOG_PATH,
    restore_model_flag=True,
    restore_optimizer_flag=False
)
trainer(train_dataset, None)

test_dataset = iter(make_domain_dataset(
    paths=target_paths,
    labels=target_labels,
    preprocessor=preprocessor,
    batch_size=BATCH_SIZE,
))
test_step = SourceTestStep(
    build_backbone_lambda=build_backbone_lambda,
    build_top_lambda=build_top_lambda
)
tester = Tester(test_step=test_step, log_path=LOG_PATH)
tester(test_dataset)
