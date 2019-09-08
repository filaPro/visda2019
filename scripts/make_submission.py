import os
import numpy as np
from argparse import ArgumentParser


def run_domain(in_path, domain_path, track, domain):
    if track == 0:
        track_name = 'multi_source'
        phase = 'test'
    else:
        track_name = 'semi_supervised'
        phase = 'unl'
    test_path = os.path.join(in_path, track_name, 'raw', f'{domain}_{phase}.txt')
    with open(test_path) as file:
        true_paths = np.array(tuple(map(lambda x: x.split()[0].split('/')[-1], file.readlines())))
    with open(os.path.join(domain_path, 'result.txt')) as file:
        lines = file.readlines()
        paths = np.array(tuple(map(lambda x: x.split()[0].split('/')[-1], lines)))
        predictions = np.array(tuple(map(lambda x: int(x.split()[1]), lines)))
    sorted_indexes = np.argsort(paths)
    indexes = sorted_indexes[np.searchsorted(paths[sorted_indexes], true_paths)]
    predictions = predictions[indexes]
    return predictions


def run(in_path, out_path, clipart_path, painting_path, track):
    clipart_predictions = run_domain(in_path, clipart_path, track, 'clipart')
    painting_predictions = run_domain(in_path, painting_path, track, 'painting')
    with open(os.path.join(out_path, 'result.txt'), 'w') as file:
        for p in np.concatenate((clipart_predictions, painting_predictions)):
            file.write(f'{p}\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in-path', type=str, default='/content/data')
    parser.add_argument('--out-path', type=str, default='/content/logs/tmp')
    parser.add_argument('--clipart-path', type=str, default='/content/logs/tmp-mix-match-clipart')
    parser.add_argument('--painting-path', type=str, default='/content/logs/tmp-mix-match-painting')
    parser.add_argument('--track', type=int, required=True, help='0: multi source, 1: semi supervised')
    options = vars(parser.parse_args())
    run(
        in_path=options['in_path'],
        out_path=options['out_path'],
        clipart_path=options['clipart_path'],
        painting_path=options['painting_path'],
        track=options['track']
    )
