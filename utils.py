import os
import random
import numpy as np
import tensorflow as tf


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
        print(f'extracting: {domain}')
        os.system(f'cd {path} && unzip -qo {domain}.zip')


def read_paths_and_labels(path, domain):
    print('>', domain)
    paths_and_labels = []
    for suffix in ('train', 'test'):
        with open(os.path.join(path, f'{domain}_{suffix}.txt')) as file:
            paths_and_labels += list(map(lambda s: s.split(), file.readlines()))
    random.shuffle(paths_and_labels)
    paths, labels = zip(*paths_and_labels)
    paths = list(map(lambda s: os.path.join(path, s), paths))
    labels = list(map(int, labels))
    return paths, labels


# TODO: enable train / test split
def read_source_target_paths_and_labels(path, domains, target_domain_id):
    print('source:')
    sources_paths = []
    sources_labels = []
    source_domain_ids = np.setdiff1d(np.arange(len(domains)), (target_domain_id,))
    for domain_id in source_domain_ids:
        paths, labels = read_paths_and_labels(path, domains[domain_id])
        sources_paths.append(paths)
        sources_labels.append(labels)
    print('target:')
    target_paths, target_labels = read_paths_and_labels(path, domains[target_domain_id])
    return sources_paths, sources_labels, target_paths, target_labels


@tf.function
def decode_image(path):
    raw = tf.io.read_file(path)
    if tf.image.is_jpeg(raw):
        return tf.image.decode_jpeg(raw, channels=3)
    return tf.image.decode_png(raw, channels=3)


def make_domain_dataset(paths, labels, batch_size, image_size):
    return tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(paths),
        tf.data.Dataset.from_tensor_slices(labels)
    )).shuffle(
        23456
    ).map(
        lambda path, label: (decode_image(path), label)
    ).map(
        lambda image, label: (tf.keras.applications.mobilenet.preprocess_input(
            tf.image.resize(image, (image_size, image_size))), label
        )
    ).batch(batch_size)


def make_dataset(sources_paths, target_paths, sources_labels, target_labels, batch_size, image_size):
    return tf.data.Dataset.zip(tuple(
        make_domain_dataset(paths, labels, batch_size, image_size)
        for paths, labels in zip(
            sources_paths + [target_paths], sources_labels + [target_labels]
        )
    )).repeat()


