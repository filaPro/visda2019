import os
import numpy as np
import tensorflow as tf

from utils import get_time_string


class Tester:
    def __init__(self, build_test_step_lambda, log_path):
        """
        :param test_step: requires `.iteration`, `.metrics`, `.test(batch)`
        """
        self.build_test_step_lambda = build_test_step_lambda
        self.log_path = os.path.join(log_path, 'log.txt')
        self.checkpoint_path = os.path.join(log_path, 'checkpoint')

    def __call__(self, dataset):
        paths = []
        predictions = []
        probabilities = []
        test_step = self.build_test_step_lambda()
        tf.train.Checkpoint(**test_step.models).restore(tf.train.latest_checkpoint(self.checkpoint_path))
        for batch in dataset:
            iteration = test_step.iteration.numpy()
            path, probability = test_step.test(batch)
            probability = probability.numpy()
            paths += path.numpy().tolist()
            predictions += np.argmax(probability, axis=1).tolist()
            probabilities.append(probability)
            string = f'\riteration: {iteration + 1}'
            for name, metric in test_step.metrics.items():
                string += f', {name}: {metric.result().numpy():.5e}'
            print(string, end='')
        print()
        with open(self.log_path, 'a') as file:
            file.write(f'{get_time_string()} Tester: {string[1:]}\n')
        return paths, predictions, np.concatenate(probabilities)
