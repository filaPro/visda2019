import os
import random
import shutil
import tensorflow as tf
from functools import partial
from datetime import datetime


DOMAINS = ('real', 'infograph', 'quickdraw', 'sketch', 'clipart', 'painting')
N_CLASSES = 345
BUFFER_SIZE = 128
N_PROCESSES = 16


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


def list_tfrecords(path, domain, phase):
    file_names = os.listdir(path)
    random.shuffle(file_names)
    results = []
    for file_name in file_names:
        splits = file_name.split('_')
        if splits[0] == domain and (phase == splits[1] or phase == 'all'):
            results.append(os.path.join(path, file_name))
    return results


def make_domain_dataset(paths, preprocessor):
    print(f'paths: {paths}')
    return tf.data.Dataset.from_tensor_slices(
        paths
    ).interleave(
        tf.data.TFRecordDataset,
        cycle_length=max(len(paths), N_PROCESSES),
        num_parallel_calls=N_PROCESSES
    ).map(
        partial(parse_example, preprocessor=preprocessor),
        num_parallel_calls=N_PROCESSES
    )


def make_multi_source_dataset(
    source_domains, source_phase, source_preprocessor, source_batch_size,
    target_domain, target_phase, target_preprocessor, target_batch_size,
    path
):
    datasets = list()
    for domain in source_domains:
        paths = list_tfrecords(
            path=os.path.join(path, 'multi_source', 'tfrecords'),
            domain=domain,
            phase=source_phase
        )
        datasets.append(make_domain_dataset(
            paths=paths,
            preprocessor=source_preprocessor
        ).repeat().shuffle(BUFFER_SIZE).batch(source_batch_size))
    target_paths = list_tfrecords(
        path=os.path.join(path, 'multi_source', 'tfrecords'),
        domain=target_domain,
        phase=target_phase
    )
    datasets.append(make_domain_dataset(
        paths=target_paths,
        preprocessor=target_preprocessor
    ).repeat().shuffle(BUFFER_SIZE).batch(target_batch_size))
    return tf.data.Dataset.zip(tuple(datasets)).prefetch(N_PROCESSES)


def make_combined_multi_source_dataset(
    source_domains, source_phase, source_preprocessor, source_batch_size,
    target_domain, target_phase, target_preprocessor, target_batch_size,
    path
):
    datasets = list()
    for domain in source_domains:
        paths = list_tfrecords(
            path=os.path.join(path, 'multi_source', 'tfrecords'),
            domain=domain,
            phase=source_phase
        )
        datasets.append(make_domain_dataset(
            paths=paths,
            preprocessor=source_preprocessor
        ).repeat().shuffle(BUFFER_SIZE))
    source_dataset = tf.data.experimental.sample_from_datasets(datasets).batch(source_batch_size)
    target_paths = list_tfrecords(
        path=os.path.join(path, 'multi_source', 'tfrecords'),
        domain=target_domain,
        phase=target_phase
    )
    target_dataset = make_domain_dataset(
        paths=target_paths,
        preprocessor=target_preprocessor
    ).repeat().shuffle(BUFFER_SIZE).batch(target_batch_size)
    return tf.data.Dataset.zip((source_dataset, target_dataset)).prefetch(N_PROCESSES)


def make_semi_supervised_dataset(
    source_domain, source_phase, source_preprocessor, source_batch_size,
    labeled_preprocessor, labeled_batch_size,
    unlabeled_preprocessor, unlabeled_batch_size,
    target_domain, path
):
    datasets = list()
    source_paths = list_tfrecords(
        path=os.path.join(path, 'multi_source', 'tfrecords'),
        domain=source_domain,
        phase=source_phase
    )
    datasets.append(make_domain_dataset(
        paths=source_paths,
        preprocessor=source_preprocessor
    ).repeat().shuffle(BUFFER_SIZE).batch(source_batch_size))
    labeled_paths = list_tfrecords(
        path=os.path.join(path, 'semi_supervised', 'tfrecords'),
        domain=target_domain,
        phase='labeled'
    )
    datasets.append(make_domain_dataset(
        paths=labeled_paths,
        preprocessor=labeled_preprocessor
    ).repeat().shuffle(BUFFER_SIZE).batch(labeled_batch_size))
    unlabeled_paths = list_tfrecords(
        path=os.path.join(path, 'semi_supervised', 'tfrecords'),
        domain=target_domain,
        phase='unlabeled'
    )
    datasets.append(make_domain_dataset(
        paths=unlabeled_paths,
        preprocessor=unlabeled_preprocessor
    ).repeat().shuffle(BUFFER_SIZE).batch(unlabeled_batch_size))
    return tf.data.Dataset.zip(tuple(datasets)).prefetch(N_PROCESSES)


def get_time_string():
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')


def copy_runner(file, path):
    os.makedirs(path, exist_ok=True)
    shutil.copy(os.path.realpath(file), path)
