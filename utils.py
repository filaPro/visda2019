import os
import shutil
import tensorflow as tf
from functools import partial
from datetime import datetime


DOMAINS = ('real', 'infograph', 'quickdraw', 'sketch', 'clipart', 'painting')
N_CLASSES = 345
BUFFER_SIZE = 128


def download_raw_data(path, domains):
    base_url = 'http://csr.bu.edu/ftp/visda/2019/multi-source/'

    if os.path.exists(path):
        print(
            'Raw data path already exists. '
            'Be careful with downloading twice. '
            'If necessary remove it and retry downloading.'
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


def make_domain_dataset(paths, preprocessor, n_processes):
    print(f'paths: {paths}')
    return tf.data.Dataset.list_files(
        paths
    ).interleave(
        tf.data.TFRecordDataset,
        cycle_length=n_processes,
        num_parallel_calls=n_processes
    ).map(
        partial(parse_example, preprocessor=preprocessor),
        num_parallel_calls=n_processes
    )


def phase_to_template(phase):
    return '*' if phase == 'all' else f'{phase}*'


def make_multi_source_dataset(
    source_domains, source_phase, source_preprocessor, source_batch_size,
    target_domain, target_phase, target_preprocessor, target_batch_size,
    path, n_processes
):
    datasets = list()
    for domain in source_domains:
        datasets.append(make_domain_dataset(
            paths=os.path.join(path, 'multi_source', 'tfrecords', f'{domain}_{phase_to_template(source_phase)}'),
            preprocessor=source_preprocessor,
            n_processes=n_processes
        ).repeat().shuffle(BUFFER_SIZE).batch(source_batch_size))
    datasets.append(make_domain_dataset(
        paths=os.path.join(path, 'multi_source', 'tfrecords', f'{target_domain}_{phase_to_template(target_phase)}'),
        preprocessor=target_preprocessor,
        n_processes=n_processes
    ).repeat().shuffle(BUFFER_SIZE).batch(target_batch_size))
    return tf.data.Dataset.zip(tuple(datasets)).prefetch(n_processes)


def make_semi_supervised_dataset(
    source_domain, source_phase, source_preprocessor, source_batch_size,
    labeled_preprocessor, labeled_batch_size,
    unlabeled_preprocessor, unlabeled_batch_size,
    target_domain, path, n_processes
):
    datasets = list()
    datasets.append(make_domain_dataset(
        paths=os.path.join(path, 'multi_source', 'tfrecords', f'{source_domain}_{phase_to_template(source_phase)}'),
        preprocessor=source_preprocessor,
        n_processes=n_processes
    ).repeat().shuffle(BUFFER_SIZE).batch(source_batch_size))
    datasets.append(make_domain_dataset(
        paths=os.path.join(path, 'semi_supervised', 'tfrecords', f'{target_domain}_labeled*'),
        preprocessor=labeled_preprocessor,
        n_processes=n_processes
    ).repeat().shuffle(BUFFER_SIZE).batch(labeled_batch_size))
    datasets.append(make_domain_dataset(
        paths=os.path.join(path, 'semi_supervised', 'tfrecords', f'{target_domain}_unlabeled*'),
        preprocessor=unlabeled_preprocessor,
        n_processes=n_processes
    ).repeat().shuffle(BUFFER_SIZE).batch(unlabeled_batch_size))
    return tf.data.Dataset.zip(tuple(datasets)).prefetch(n_processes)


def get_time_string():
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')


def copy_runner(file, path):
    os.makedirs(path, exist_ok=True)
    shutil.copy(os.path.realpath(file), path)
