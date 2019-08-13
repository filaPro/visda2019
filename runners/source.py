import tensorflow as tf
from functools import partial

from trainer import Trainer
from tester import Tester
from models import SourceTrainStep, SourceTestStep, build_backbone
from utils import DOMAINS, N_CLASSES, read_paths_and_labels, make_dataset, make_domain_dataset
from preprocessor import Preprocessor

RAW_DATA_PATH = '/content/data/raw'
LOG_PATH = '/content/data/logs'
BATCH_SIZE = 32
IMAGE_SIZE = 224
BACKBONE_NAME = 'mobilenet_v2'
CONFIG = [
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

paths_and_labels = read_paths_and_labels(RAW_DATA_PATH, DOMAINS)
target_paths = paths_and_labels['target']['train']['paths'] + paths_and_labels['target']['test']['paths']
target_labels = paths_and_labels['target']['train']['labels'] + paths_and_labels['target']['test']['labels']
train_dataset = iter(make_dataset(
    source_paths=paths_and_labels['source']['train']['paths'],
    source_labels=paths_and_labels['source']['train']['labels'],
    source_config=CONFIG,
    target_paths=target_paths,
    target_labels=target_labels,
    target_config=CONFIG,
    batch_size=BATCH_SIZE
))
validate_dataset = iter(make_dataset(
    source_paths=paths_and_labels['source']['test']['paths'],
    source_labels=paths_and_labels['source']['test']['labels'],
    source_config=CONFIG,
    target_paths=target_paths,
    target_labels=target_labels,
    target_config=CONFIG,
    batch_size=BATCH_SIZE
))

train_step = SourceTrainStep(
    build_backbone_lambda=build_backbone_lambda,
    build_top_lambda=build_top_lambda,
    domains=DOMAINS,
    freeze_backbone_flag=True,
    backbone_training_flag=False,
    learning_rate=0.001
)
trainer = Trainer(
    train_step=train_step,
    n_iterations=500,
    n_log_iterations=100,
    n_save_iterations=500,
    n_validate_iterations=10,
    log_path=LOG_PATH,
    restore_model_flag=False,
    restore_optimizer_flag=False
)
trainer(validate_dataset, train_dataset)

train_step = SourceTrainStep(
    build_backbone_lambda=build_backbone_lambda,
    build_top_lambda=build_top_lambda,
    domains=DOMAINS,
    freeze_backbone_flag=False,
    backbone_training_flag=False,
    learning_rate=0.0001
)
trainer = Trainer(
    train_step=train_step,
    n_iterations=2000,
    n_log_iterations=100,
    n_save_iterations=2000,
    n_validate_iterations=10,
    log_path=LOG_PATH,
    restore_model_flag=True,
    restore_optimizer_flag=False
)
trainer(train_dataset, validate_dataset)

test_dataset = iter(make_domain_dataset(
    paths=target_paths,
    labels=target_labels,
    preprocessor=Preprocessor(CONFIG),
    batch_size=BATCH_SIZE,
    seed=None
))
test_step = SourceTestStep(
    build_backbone_lambda=build_backbone_lambda,
    build_top_lambda=build_top_lambda
)
tester = Tester(test_step=test_step, log_path=LOG_PATH)
tester(test_dataset)
# >>> acc: 2.98042e-01
