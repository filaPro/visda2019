import os
import random
import shutil
import tensorflow as tf
from datetime import datetime


DOMAINS = ('infograph', 'quickdraw', 'real', 'sketch')
N_CLASSES = 345


def download_raw_data(path, domains):
    base_url = 'http://csr.bu.edu/ftp/visda/2019/multi-source/'

    if os.path.exists(path):
        print(
            'Raw data path already exists. '
            'Be careful with downloading twice. '
            'If nessesary remove it and retry downloading.'
        )
        return

    os.makedirs(path)
    for domain in domains:
        urls = (
            f'{base_url}{domain}.zip',
            f'{base_url}txt/{domain}_train.txt',
            f'{base_url}txt/{domain}_test.txt'
        )
        for url in urls:
            print(f'downloading: {url}')
            os.system(f'wget -P {path} {url}')


def unzip_raw_data(path, domains):
    for domain in domains:
        print(f'extracting: {domain}')
        os.system(f'cd {path} && unzip -qo {domain}.zip')


def read_domain_paths_and_labels(path, domain, phase):
    """
    :param phase: 'train' or 'test'
    """
    print('>', domain, phase)
    with open(os.path.join(path, f'{domain}_{phase}.txt')) as file:
        paths_and_labels = list(map(lambda s: s.split(), file.readlines()))
    random.shuffle(paths_and_labels)
    paths, labels = zip(*paths_and_labels)
    paths = list(map(lambda s: os.path.join(path, s), paths))
    labels = list(map(int, labels))
    return paths, labels


def read_paths_and_labels(path, domains):
    print('source:')
    paths_and_labels = {
        'source': {
            'train': {
                'labels': [],
                'paths': []
            },
            'test': {
                'labels': [],
                'paths': []
            }
        },
        'target': {
            'train': {
                'labels': [],
                'paths': []
            },
            'test': {
                'labels': [],
                'paths': []
            }
        }
    }
    for domain_id in range(len(domains) - 1):
        for phase in ('train', 'test'):
            paths, labels = read_domain_paths_and_labels(path, domains[domain_id], phase)
            paths_and_labels['source'][phase]['paths'].append(paths)
            paths_and_labels['source'][phase]['labels'].append(labels)
    print('target:')
    for phase in ('train', 'test'):
        paths, labels = read_domain_paths_and_labels(path, domains[-1], phase)
        paths_and_labels['target'][phase]['paths'] = paths
        paths_and_labels['target'][phase]['labels'] = labels
    return paths_and_labels


@tf.function
def decode_image(path):
    raw = tf.io.read_file(path)
    if tf.image.is_jpeg(raw):
        image = tf.image.decode_jpeg(raw, channels=3)
    else:
        image = tf.image.decode_png(raw, channels=3)
    return tf.cast(image, tf.float32)


def make_domain_dataset(paths, labels, preprocessor, batch_size):
    return tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(paths),
        tf.data.Dataset.from_tensor_slices(labels)
    )).shuffle(
        23456
    ).map(
        lambda path, label: (preprocessor(decode_image(path)), label)
    ).batch(batch_size)


def make_dataset(
    source_paths, source_labels, source_preprocessor,
    target_paths, target_labels, target_preprocessor,
    batch_size
):
    datasets = []
    for paths, labels in zip(source_paths, source_labels):
        datasets.append(make_domain_dataset(paths, labels, source_preprocessor, batch_size))
    datasets.append(make_domain_dataset(target_paths, target_labels, target_preprocessor, batch_size))
    return tf.data.Dataset.zip(tuple(datasets)).repeat()


def get_time_string():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")


def copy_runner(file, path):
    os.makedirs(path)
    shutil.copy(os.path.realpath(file), path)
