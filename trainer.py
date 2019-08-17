import os
import tensorflow as tf

from utils import get_time_string


class Trainer:
    def __init__(
        self, train_step, n_iterations, n_log_iterations, n_save_iterations, n_validate_iterations,
        log_path, restore_model_flag, restore_optimizer_flag
    ):
        """
        :param train_step: requires `.iteration`, `.metrics`, `.losses`, `.optimizers`, `.train(batch)`,
            `.validate(batch)`
        """
        self.train_step = train_step
        self.n_iterations = n_iterations
        self.n_log_iterations = n_log_iterations
        self.n_save_iterations = n_save_iterations
        self.n_validate_iterations = n_validate_iterations
        self.log_path = os.path.join(log_path, 'log.txt')
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

    def __call__(self, train_dataset, validate_dataset):
        """
        :param train_dataset: `next()` returns batch
        :param validate_dataset: `next()` returns batch; None if not self.n_validate_iterations
        """
        for _ in range(self.n_iterations):
            iteration = self.train_step.iteration.numpy()
            self.train_step.train(next(train_dataset))
            if self.n_validate_iterations and iteration % self.n_validate_iterations == self.n_validate_iterations - 1:
                self.train_step.validate(next(validate_dataset))
            string = f'\riteration: {iteration + 1}'
            for name, metric in self.train_step.metrics.items():
                string += f', {name}: {metric.result().numpy():.5e}'
                if self.n_log_iterations and iteration % self.n_log_iterations == self.n_log_iterations - 1:
                    metric.reset_states()
            print(string, end='')
            if self.n_log_iterations and iteration % self.n_log_iterations == self.n_log_iterations - 1:
                print()
                with open(self.log_path, 'a') as file:
                    file.write(f'{get_time_string()} Trainer: {string[1:]}\n')
            if self.n_save_iterations and iteration % self.n_save_iterations == self.n_save_iterations - 1:
                checkpoint_path = self.checkpoint_manager.save()
                print(f'iteration: {iteration + 1}, save: {checkpoint_path}')
