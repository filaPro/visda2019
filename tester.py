import os
import tensorflow as tf

from utils import get_time_string


class Tester:
    def __init__(self, test_step, log_path):
        """
        :param test_step: requires `.iteration`, `.metrics`, `.test(batch)`
        """
        self.test_step = test_step
        self.log_path = os.path.join(log_path, 'log.txt')
        checkpoint = tf.train.Checkpoint(**test_step.models)
        checkpoint_path = os.path.join(log_path, 'checkpoint')
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

    def __call__(self, dataset):
        """
        :param dataset: `.__call__` returns batch
        """
        for batch in dataset:
            iteration = self.test_step.iteration.numpy()
            self.test_step.test(batch)
            string = f'\riteration: {iteration + 1}'
            for name, metric in self.test_step.metrics.items():
                string += f', {name}: {metric.result().numpy():.5e}'
            print(string, end='')
        print()
        with open(self.log_path, 'a') as file:
            file.write(f'{get_time_string()} Tester: {string[1:]}\n')
