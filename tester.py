import os
import tensorflow as tf


class Tester:
    def __init__(self, test_step, log_path):
        """
        :param test_step: requires `.iteration`, `.metrics`, `.__call__(batch)`
        """
        self.test_step = test_step
        checkpoint = tf.train.Checkpoint(**test_step.models)
        checkpoint_path = os.path.join(log_path, 'checkpoint')
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

    def __call__(self, dataset):
        """
        :param dataset: `.__call__` returns batch
        """
        for batch in dataset:
            iteration = self.test_step.iteration.numpy()
            self.test_step(batch)
            string = f'\riteration: {iteration + 1}'
            for name, metric in self.test_step.metrics.items():
                string += f', {name}: {metric.result().numpy():.5e}'
            print(string, end='')
        print()
