import os
import numpy as np
from argparse import ArgumentParser


def run(in_paths, weights, out_path):
    in_paths = in_paths.split(',')
    weights = tuple(map(float, weights.split(',')))
    assert len(in_paths) == len(weights)
    paths = None
    predictions = None
    for in_path, weight in zip(in_paths, weights):
        with open(os.path.join(in_path, 'result.txt')) as file:
            current_paths = np.array(tuple(map(lambda x: x.split()[0], file.readlines())))
            if paths is None:
                paths = current_paths
            assert np.array_equal(paths, current_paths)
        probablility = np.load(os.path.join(in_path, 'probability.npy'))
        if predictions is None:
            predictions = np.zeros_like(probablility)
        predictions += probablility * weight
    predictions = np.argmax(predictions, axis=1)
    with open(os.path.join(out_path, 'result.txt'), 'w') as file:
        for path, prediction in zip(paths, predictions):
            file.write(f'{path} {prediction}\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in-paths', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--out-path', type=str, default='/content/logs/tmp')
    options = vars(parser.parse_args())
    run(
        in_paths=options['in_paths'],
        weights=options['weights'],
        out_path=options['out_path'],
    )
