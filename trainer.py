import os
import tensorflow as tf


class Trainer:
    def __init__(
        self, train_step, n_iterations, n_log_iterations, n_save_iterations, log_path,
        restore_model_flag, restore_optimizer_flag
    ):
        """
        :param train_step: requires `.iteration`, `.metrics`, `.losses`, `.optimizers`, `.__call__(batch)`
        """
        self.train_step = train_step
        self.n_iterations = n_iterations
        self.n_log_iterations = n_log_iterations
        self.n_save_iterations = n_save_iterations
        self.checkpoint = tf.train.Checkpoint(**train_step.models, **train_step.optimizers)
        checkpoint_path = os.path.join(log_path, 'checkpoint')
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_path, max_to_keep=5)
        if restore_model_flag or restore_optimizer_flag:
            restore_objects = {}
            if restore_model_flag:
                restore_objects.update(train_step.models)
            if restore_optimizer_flag:
                restore_objects.update(train_step.optimizers)
            restore_checkpoint = tf.train.Checkpoint(**restore_objects)
            restore_checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

    def __call__(self, dataset):
        """
        :param dataset: `.__call__` returns batch
        """
        for batch in dataset:
            iteration = self.train_step.iteration.numpy()
            self.train_step(batch)
            string = f'\riteration: {iteration + 1}'
            for name, metric in self.train_step.metrics.items():
                string += f', {name}: {metric.result().numpy():.5e}'
                if self.n_log_iterations and iteration % self.n_log_iterations == self.n_log_iterations - 1:
                    metric.reset_states()
            print(string, end='')
            if self.n_log_iterations and iteration % self.n_log_iterations == self.n_log_iterations - 1:
                print()
            if self.n_save_iterations and iteration % self.n_save_iterations == self.n_save_iterations - 1:
                checkpoint_path = self.checkpoint_manager.save()
                print(f'iteration: {iteration + 1}, save: {checkpoint_path}')
            if iteration == self.n_iterations - 1:
                break
