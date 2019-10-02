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
IMAGE_SIZE = 224
N_TRAIN_ITERATIONS = 1000
LEARNING_RATE = .0001


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


def concatenate(*args):
    result = {}
    for key in args[0].keys():
        result[key] = tf.concat(tuple(args[i][key] for i in range(len(args))), axis=0)
    return result


def make_domain_dataset(paths, preprocessor):
    print(f'paths: {paths}')
    return tf.data.Dataset.from_tensor_slices(
        sorted(paths)
    ).interleave(
        tf.data.TFRecordDataset,
        cycle_length=N_PROCESSES,
        num_parallel_calls=N_PROCESSES
    ).map(
        partial(parse_example, preprocessor=preprocessor),
        num_parallel_calls=N_PROCESSES
    )


def make_multi_source_datasets(
    source_domains, source_phase, source_preprocessor,
    target_domain, target_phase, target_preprocessor,
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
    target_paths = list_tfrecords(
        path=os.path.join(path, 'multi_source', 'tfrecords'),
        domain=target_domain,
        phase=target_phase
    )
    datasets.append(make_domain_dataset(
        paths=target_paths,
        preprocessor=target_preprocessor
    ).repeat().shuffle(BUFFER_SIZE))
    return tuple(datasets)


def make_multi_source_dataset(
    source_domains, source_phase, source_preprocessor, source_batch_size,
    target_domain, target_phase, target_preprocessor, target_batch_size,
    path
):
    datasets = make_multi_source_datasets(
        source_domains, source_phase, source_preprocessor,
        target_domain, target_phase, target_preprocessor,
        path
    )
    source_datasets = tuple(d.batch(source_batch_size) for d in datasets[:-1])
    target_dataset = datasets[-1].batch(target_batch_size)
    return tf.data.Dataset.zip(source_datasets + (target_dataset,)).prefetch(N_PROCESSES)


def make_combined_multi_source_dataset(
    source_domains, source_phase, source_preprocessor, source_batch_size,
    target_domain, target_phase, target_preprocessor, target_batch_size,
    path
):
    datasets = make_multi_source_datasets(
        source_domains, source_phase, source_preprocessor,
        target_domain, target_phase, target_preprocessor,
        path
    )
    # source_dataset = tf.data.experimental.sample_from_datasets(datasets[:-1]).batch(source_batch_size)
    # This is a dirty hack.
    # Reason 1: tf.data.experimental causes segmentation fault on multi GPU for tensorflow-gpu==2.0.0beta*.
    # Reason 2: tensorflow can not be easily updated because will need cudnn7.6, while cluster has <7.5.
    source_dataset = tf.data.Dataset.zip(
        datasets[:-1]
    ).batch(
        1
    ).map(
        concatenate,
        num_parallel_calls=N_PROCESSES
    ).unbatch(
    ).batch(
        source_batch_size
    )
    target_dataset = datasets[-1].batch(target_batch_size)
    return tf.data.Dataset.zip((source_dataset, target_dataset)).prefetch(N_PROCESSES)


def make_semi_supervised_datasets(
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
    return tuple(datasets)


def make_semi_supervised_dataset(
    source_domain, source_phase, source_preprocessor, source_batch_size,
    labeled_preprocessor, labeled_batch_size,
    unlabeled_preprocessor, unlabeled_batch_size,
    target_domain, path
):
    datasets = make_semi_supervised_datasets(
        source_domain, source_phase, source_preprocessor, source_batch_size,
        labeled_preprocessor, labeled_batch_size,
        unlabeled_preprocessor, unlabeled_batch_size,
        target_domain, path
    )
    return tf.data.Dataset.zip(datasets).prefetch(N_PROCESSES)


def make_combined_semi_supervised_dataset(
    source_domain, source_phase, source_preprocessor, source_batch_size,
    labeled_preprocessor, labeled_batch_size,
    unlabeled_preprocessor, unlabeled_batch_size,
    target_domain, path
):
    datasets = make_semi_supervised_datasets(
        source_domain, source_phase, source_preprocessor, source_batch_size,
        labeled_preprocessor, labeled_batch_size,
        unlabeled_preprocessor, unlabeled_batch_size,
        target_domain, path
    )
    source_dataset = tf.data.Dataset.zip(
        datasets[:2]
    ).map(
        concatenate,
        num_parallel_calls=N_PROCESSES
    )
    target_dataset = datasets[-1]
    return tf.data.Dataset.zip((source_dataset, target_dataset)).prefetch(N_PROCESSES)


def get_time_string():
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')


def copy_runner(file, path):
    os.makedirs(path, exist_ok=True)
    shutil.copy(os.path.realpath(file), path)


def get_preprocessor_config(normalization):
    return [
        {'method': 'resize', 'height': 256, 'width': 256},
        {'method': 'random_flip_left_right'},
        {'method': 'keras', 'mode': normalization},
        {'method': 'random_crop', 'height': IMAGE_SIZE, 'width': IMAGE_SIZE, 'n_channels': 3}
    ]


def get_track_information(track):
    phases = ('train', 'test') if track == 0 else ('labeled', 'unlabeled')
    domains = DOMAINS if track == 0 else DOMAINS[3:]
    name = 'multi_source' if track == 0 else 'semi_supervised'
    return phases, domains, name
