import os
import numpy as np
from argparse import ArgumentParser


def run(out_path, in_names, out_name, weights):
    in_names = in_names.split(',')
    weights = tuple(map(float, weights.split(',')))
    assert len(in_names) == len(weights)
    paths = None
    predictions = None
    for in_name, weight in zip(in_names, weights):
        with open(os.path.join(out_path, in_name, 'result.txt')) as file:
            current_paths = np.array(tuple(map(lambda x: x.split()[0], file.readlines())))
            if paths is None:
                paths = current_paths
            assert np.array_equal(paths, current_paths)
        probablility = np.load(os.path.join(out_path, in_name, 'probability.npy'))
        if predictions is None:
            predictions = np.zeros_like(probablility)
        predictions += probablility * weight
    predictions = np.argmax(predictions, axis=1)
    with open(os.path.join(out_path, out_name, 'result.txt'), 'w') as file:
        for path, prediction in zip(paths, predictions):
            file.write(f'{path} {prediction}\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--out-path', type=str, default='/content/logs')
    parser.add_argument('--in-names', type=str, required=True)
    parser.add_argument('--out-name', type=str, default='tmp')
    parser.add_argument('--weights', type=str, required=True)
    options = vars(parser.parse_args())
    run(
        out_path=options['out_path'],
        in_names=options['in_names'],
        out_name=options['out_name'],
        weights=options['weights'],
    )
