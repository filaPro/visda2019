from trainer import Trainer
from models import M3sdaModel

from utils import DOMAINS, N_CLASSES, download_raw_data, read_source_target_paths_and_labels, make_dataset

raw_data_path = 'raw_data'
batch_size = 32
image_size = 96

download_raw_data(raw_data_path, DOMAINS)
sources_paths, sources_labels, target_paths, target_labels = \
    read_source_target_paths_and_labels(raw_data_path, DOMAINS, 3)
dataset = make_dataset(sources_paths, target_paths, sources_labels, target_labels, batch_size, image_size)
model = M3sdaModel(beta=False, n_classes=N_CLASSES, domains=DOMAINS, image_size=image_size, n_moments=5)
trainer = Trainer(500, 100)
trainer.train(model, dataset)