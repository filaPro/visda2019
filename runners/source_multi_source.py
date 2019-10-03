import os
from functools import partial
from argparse import ArgumentParser

from trainer import Trainer
from models import SourceTrainStep, build_backbone, get_n_backbone_channels, get_backbone_normalization, build_top
from utils import (
    DOMAINS, N_CLASSES, N_TRAIN_ITERATIONS, IMAGE_SIZE, LEARNING_RATE,
    make_multi_source_dataset, copy_runner, get_time_string, get_preprocessor_config
)
from preprocessor import Preprocessor


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in-path', type=str, default='/content/data')
    parser.add_argument('--out-path', type=str, default='/content/logs')
    parser.add_argument('--name', type=str, default='tmp')
    parser.add_argument('--time', type=int, default=0)
    parser.add_argument('--n-gpus', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--n-epochs', type=int, default=100)
    parser.add_argument('--backbone', type=str, default='efficient_net_b5')
    parser.add_argument('--source-domains', type=str, default='0,1,2')
    parser.add_argument('--target-domain', type=int, default=3)
    parser.add_argument('--phase', type=str, default='train')
    options = vars(parser.parse_args())

    source_domains = tuple(map(lambda x: DOMAINS[int(x)], options['source_domains'].split(',')))
    target_domain = DOMAINS[options['target_domain']]
    backbone_name = options['backbone']
    normalization = get_backbone_normalization(backbone_name)
    n_channels = get_n_backbone_channels(backbone_name)
    build_top_lambda = partial(build_top, n_channels=n_channels, n_classes=N_CLASSES)
    build_backbone_lambda = partial(build_backbone, name=backbone_name, size=IMAGE_SIZE)
    preprocessor = Preprocessor(get_preprocessor_config(normalization))
    out_path = options['out_path']
    time = f'{get_time_string()}-' if options['time'] else ''
    name = options['name']
    log_path = os.path.join(out_path, f'{time}{name}')
    batch_size = options['batch_size']
    n_gpus = options['n_gpus']

    dataset = make_multi_source_dataset(
        source_domains=source_domains,
        source_phase=options['phase'],
        source_preprocessor=preprocessor,
        source_batch_size=batch_size * n_gpus,
        target_domain=target_domain,
        target_phase=options['phase'],
        target_preprocessor=preprocessor,
        target_batch_size=batch_size * n_gpus,
        path=options['in_path']
    )

    copy_runner(__file__, log_path)
    build_train_step_lambda = partial(
        SourceTrainStep,
        build_backbone_lambda=build_backbone_lambda,
        build_top_lambda=build_top_lambda,
        domains=source_domains + (target_domain,),
        learning_rate=LEARNING_RATE,
        batch_size=batch_size * n_gpus
    )
    Trainer(
        build_train_step_lambda=build_train_step_lambda,
        n_epochs=options['n_epochs'],
        n_train_iterations=N_TRAIN_ITERATIONS,
        log_path=log_path,
        restore_model_flag=False,
        restore_optimizer_flag=False,
        single_gpu_flag=n_gpus == 1
    )(dataset)
