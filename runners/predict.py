import os
import numpy as np
from functools import partial
from argparse import ArgumentParser

from tester import Tester
from models import (
    SourceTestStep, SelfEnsemblingPreprocessor, build_backbone, get_backbone_normalization, get_n_backbone_channels,
    build_top
)
from utils import DOMAINS, IMAGE_SIZE, N_CLASSES, make_domain_dataset, list_tfrecords, get_preprocessor_config


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in-path', type=str, default='/content/data')
    parser.add_argument('--out-path', type=str, default='/content/logs')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--domain', type=str, default=3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--backbone', type=str, default='efficient_net_b5')
    parser.add_argument('--n-augmentations', type=int, default=7)
    parser.add_argument('--phase', type=str, required=True, help='all, train, test, labeled, unlabeled')
    parser.add_argument('--track', type=int, required=True, help='0: multi-source, 1: semi-supervised')
    options = vars(parser.parse_args())

    backbone_name = options['backbone']
    normalization = get_backbone_normalization(backbone_name)
    n_channels = get_n_backbone_channels(backbone_name)
    build_top_lambda = partial(build_top, n_channels=n_channels, n_classes=N_CLASSES)
    build_backbone_lambda = partial(build_backbone, name=backbone_name, size=IMAGE_SIZE)
    log_path = os.path.join(options['out_path'], options['name'])
    track = 'semi_supervised' if options['track'] else 'multi_source'
    preprocessor = SelfEnsemblingPreprocessor((get_preprocessor_config(normalization),) * options['n_augmentations'])
    paths = list_tfrecords(
        path=os.path.join(options['in_path'], track, 'tfrecords'),
        domain=DOMAINS[options['domain']],
        phase=options['phase']
    )
    dataset = make_domain_dataset(
        paths=paths,
        preprocessor=preprocessor
    ).batch(options['batch_size'])
    build_test_step_lambda = partial(
        SourceTestStep,
        build_backbone_lambda=build_backbone_lambda,
        build_top_lambda=build_top_lambda
    )
    paths, predictions, probabilities = Tester(
        build_test_step_lambda=build_test_step_lambda,
        log_path=log_path
    )(dataset)

    with open(os.path.join(log_path, 'result.txt'), 'w') as file:
        for path, prediction in zip(paths, predictions):
            file.write(f'{path.decode()} {prediction}\n')
    np.save(os.path.join(log_path, 'probability'), probabilities)
