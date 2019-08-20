import os
import tensorflow as tf
from argparse import ArgumentParser

from utils import DOMAINS, read_domain_paths_and_labels


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


def run(in_path, out_path, domains, size):
    os.makedirs(out_path, exist_ok=True)
    writer = Writer(out_path, size)
    for domain in domains:
        for phase in ('train', 'test'):
            writer.reset(f'{domain}_{phase}')
            paths, labels = read_domain_paths_and_labels(in_path, domain, phase)
            for path, label in zip(paths, labels):
                string = path_and_label_to_string(in_path, path, label)
                writer.write(string)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--size', type=int, required=True)
    options = vars(parser.parse_args())
    run(
        in_path=options['in_path'],
        out_path=options['out_path'],
        domains=DOMAINS,
        size=options['size']
    )
