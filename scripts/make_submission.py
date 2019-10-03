import os
import numpy as np
from argparse import ArgumentParser

from utils import get_track_information


def run_domain(in_path, out_path, name, track, domain):
    phases, _, track_name = get_track_information(track)
    test_path = os.path.join(in_path, track_name, 'raw', f'{domain}_{phases[1]}.txt')
    with open(test_path) as file:
        true_paths = np.array(tuple(map(lambda x: x.split()[0].split('/')[-1], file.readlines())))
    with open(os.path.join(out_path, name, 'result.txt')) as file:
        lines = file.readlines()
        paths = np.array(tuple(map(lambda x: x.split()[0].split('/')[-1], lines)))
        predictions = np.array(tuple(map(lambda x: int(x.split()[1]), lines)))
    sorted_indexes = np.argsort(paths)
    indexes = sorted_indexes[np.searchsorted(paths[sorted_indexes], true_paths)]
    predictions = predictions[indexes]
    return predictions


def run(in_path, out_path, clipart_name, painting_name, out_name, track):
    clipart_predictions = run_domain(in_path, out_path, clipart_name, track, 'clipart')
    painting_predictions = run_domain(in_path, out_path, painting_name, track, 'painting')
    with open(os.path.join(out_path, out_name, 'result.txt'), 'w') as file:
        for p in np.concatenate((clipart_predictions, painting_predictions)):
            file.write(f'{p}\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in-path', type=str, default='/content/data')
    parser.add_argument('--out-path', type=str, default='/content/logs')
    parser.add_argument('--clipart-name', type=str, required=True)
    parser.add_argument('--painting-name', type=str, required=True)
    parser.add_argument('--out-name', type=str, default='tmp')
    parser.add_argument('--track', type=int, required=True, help='0: multi-source, 1: semi-supervised')
    options = vars(parser.parse_args())
    run(
        in_path=options['in_path'],
        out_path=options['out_path'],
        clipart_name=options['clipart_name'],
        painting_name=options['painting_name'],
        out_name=options['out_name'],
        track=options['track']
    )
