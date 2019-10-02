import os
import random
import tensorflow as tf
from argparse import ArgumentParser

from utils import get_track_information


class Writer:
    def __init__(self, path, size):
        self.path = path
        self.size = size
        self.n_examples = 0
        self.n_records = 0
        self.name = None
        self.writer = None

    def write(self, string):
        self.writer.write(string)
        self.n_examples += 1
        if self.n_examples == self.size:
            self.reset(None)

    def reset(self, name):
        print(self.name, self.n_records, self.n_examples)
        if self.writer:
            self.writer.close()
        if name:
            self.name = name
            self.n_records = 0
        self.n_examples = 0
        self.n_records += 1
        path = os.path.join(self.path, f'{self.name}_{str(self.n_records - 1).zfill(3)}.tfrecord')
        self.writer = tf.io.TFRecordWriter(path)


def path_and_label_to_string(in_path, path, label):
    image_bytes = open(os.path.join(in_path, path), 'rb').read()
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=(image_bytes,))),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=(label,))),
        'path': tf.train.Feature(bytes_list=tf.train.BytesList(value=(path.encode(),)))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def read_domain_paths_and_labels(path, domain, phase):
    print('>', domain, phase)
    with open(os.path.join(path, f'{domain}_{phase}.txt')) as file:
        paths_and_labels = list(map(lambda s: s.split(), file.readlines()))
    random.shuffle(paths_and_labels)

    # For now 'clipart' and 'painting' domains don't have ground truth.
    # After the end of the competition this condition must be removed.
    if len(paths_and_labels[0]) == 1:
        paths = list(map(lambda s: s[0], paths_and_labels))
        labels = ['0'] * len(paths)
    else:
        paths, labels = zip(*paths_and_labels)

    # Semi-supervised part still has missing domain name in path.
    paths = list(map(lambda s: os.path.join(path, s if '/' in s else f'{domain}/{s}'), paths))
    labels = list(map(int, labels))
    return paths, labels


def run(path, size, track):
    phases, domains, name = get_track_information(track)
    in_path = os.path.join(path, name, 'raw')
    out_path = os.path.join(path, name, 'tfrecords')
    assert os.path.exists(in_path)
    assert not os.path.exists(out_path)
    os.mkdir(out_path)
    writer = Writer(out_path, size)
    for domain in domains:
        for phase in phases:
            writer.reset(f'{domain}_{phase}')
            paths, labels = read_domain_paths_and_labels(in_path, domain, phase)
            for path, label in zip(paths, labels):
                string = path_and_label_to_string(in_path, path, label)
                writer.write(string)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default='/content/data')
    parser.add_argument('--size', type=int, default=15000)
    options = vars(parser.parse_args())

    for i in (0, 1):
        run(path=options['path'], size=options['size'], track=i)
