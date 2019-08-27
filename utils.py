import os
import random
import shutil
import tensorflow as tf
from functools import partial
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


@tf.function
def decode_image(image_bytes):
    if tf.image.is_jpeg(image_bytes):
        image = tf.image.decode_jpeg(image_bytes, channels=3)
    else:
        image = tf.image.decode_png(image_bytes, channels=3)
    return tf.cast(image, tf.float32)


def parse_example(example, preprocessor):
    feature = {
        'image': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.int64),
        'path': tf.io.FixedLenFeature((), tf.string)
    }
    data = tf.io.parse_single_example(example, feature)
    data['image'] = preprocessor(decode_image(data['image']))
    return data


def make_domain_dataset(path, preprocessor, n_processes):
    return tf.data.Dataset.list_files(
        os.path.join(path, '*')
    ).interleave(
        tf.data.TFRecordDataset,
        cycle_length=n_processes,
        num_parallel_calls=n_processes
    ).map(
        partial(parse_example, preprocessor=preprocessor),
        num_parallel_calls=n_processes
    )


def make_dataset(
    source_path, source_preprocessor, source_batch_size,
    target_path, target_preprocessor, target_batch_size,
    domains, n_processes
):
    buffer_size = 128
    datasets = []
    for domain in domains[:-1]:
        datasets.append(make_domain_dataset(
            path=os.path.join(source_path, domain),
            preprocessor=source_preprocessor,
            n_processes=n_processes
        ).repeat().shuffle(buffer_size).batch(source_batch_size))
    datasets.append(make_domain_dataset(
        path=target_path,
        preprocessor=target_preprocessor,
        n_processes=n_processes
    ).repeat().shuffle(buffer_size).batch(target_batch_size))
    return tf.data.Dataset.zip(tuple(datasets)).prefetch(buffer_size)


def link_tfrecords(in_path, out_path, domains):
    os.makedirs(out_path, exist_ok=True)
    for file_name in os.listdir(in_path):
        domain = file_name.split('_')[0]
        phase = file_name.split('_')[1]
        split = 'target' if domain == domains[-1] else 'source'
        for all_phase in (phase, 'all'):
            path = os.path.join(out_path, split, all_phase)
            if split == 'source':
                path = os.path.join(path, domain)
            os.makedirs(path, exist_ok=True)
            os.system(f'ln -s {os.path.join(in_path, file_name)} {path}')


def get_time_string():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")


def copy_runner(file, path):
    os.makedirs(path, exist_ok=True)
    shutil.copy(os.path.realpath(file), path)
