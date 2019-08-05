from trainer import Trainer
from tester import Tester
from models import M3sdaTrainStep, M3sdaTestStep
from utils import DOMAINS, N_CLASSES, read_source_target_paths_and_labels, make_dataset, make_domain_dataset

raw_data_path = '/content/data/raw'
log_path = '/content/data/logs'
batch_size = 32
image_size = 96

sources_paths, sources_labels, target_paths, target_labels = \
    read_source_target_paths_and_labels(raw_data_path, DOMAINS, 3)
dataset = make_dataset(sources_paths, target_paths, sources_labels, target_labels, batch_size, image_size)

train_step = M3sdaTrainStep(
    n_classes=N_CLASSES,
    domains=DOMAINS,
    image_size=image_size,
    n_moments=5,
    n_frozen_layers=143,
    learning_rate=0.001,
    loss_weight=0.001
)
trainer = Trainer(
    train_step=train_step,
    n_iterations=500,
    n_log_iterations=100,
    n_save_iterations=500,
    log_path=log_path,
    restore_model_flag=False,
    restore_optimizer_flag=False
)
trainer(dataset)

train_step = M3sdaTrainStep(
    n_classes=N_CLASSES,
    domains=DOMAINS,
    image_size=image_size,
    n_moments=5,
    n_frozen_layers=0,
    learning_rate=0.0001,
    loss_weight=0.03
)
trainer = Trainer(
    train_step=train_step,
    n_iterations=1000,
    n_log_iterations=100,
    n_save_iterations=1000,
    log_path=log_path,
    restore_model_flag=True,
    restore_optimizer_flag=False
)
trainer(dataset)

test_dataset = make_domain_dataset(target_paths, target_labels, batch_size, image_size)
test_step = M3sdaTestStep(n_classes=N_CLASSES, domains=DOMAINS, image_size=image_size)
tester = Tester(test_step=test_step, log_path=log_path)
tester(test_dataset)
# >>> accuracy: 8.52442e-02
