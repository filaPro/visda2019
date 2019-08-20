import os
import tensorflow as tf

from utils import get_time_string


class Trainer:
    def __init__(
        self, build_train_step_lambda, n_epochs, n_train_iterations, n_validate_iterations,
        log_path, restore_model_flag, restore_optimizer_flag
    ):
        """
        :param build_train_step_lambda: requires `.iteration`, `.metrics`, `.losses`, `.optimizers`, `.train(batch)`,
            `.validate(batch)`
        """
        self.build_train_step_lambda = build_train_step_lambda
        self.n_epochs = n_epochs
        self.n_train_iterations = n_train_iterations
        self.n_validate_iterations = n_validate_iterations
        self.restore_model_flag = restore_model_flag
        self.restore_optimizer_flag = restore_optimizer_flag
        self.log_path = os.path.join(log_path, 'log.txt')
        self.checkpoint_path = os.path.join(log_path, 'checkpoint')

    def __call__(self, train_dataset, validate_dataset):
        strategy = tf.distribute.MirroredStrategy()
        print(f'n_devices in strategy: {strategy.num_replicas_in_sync}')
        train_dataset = self._distribute(train_dataset, strategy)
        validate_dataset = self._distribute(validate_dataset, strategy)

        with strategy.scope():
            train_step = self.build_train_step_lambda()
            self.checkpoint = tf.train.Checkpoint(**train_step.models, **train_step.optimizers)
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_path, max_to_keep=5)
            self._restore(
                models=train_step.models,
                optimizers=train_step.optimizers,
                model_flag=self.restore_model_flag,
                optimizer_flag=self.restore_optimizer_flag,
                path=self.checkpoint_path
            )

            train_step_lambda = tf.function(lambda x: strategy.experimental_run_v2(train_step.train, (x,)))
            if self.n_validate_iterations:
                validate_step_lambda = tf.function(lambda x: strategy.experimental_run_v2(train_step.validate, (x,)))

            for epoch in range(self.n_epochs):
                for _ in range(self.n_train_iterations):
                    iteration = train_step.iteration.numpy()
                    train_step_lambda(next(train_dataset))
                    string = f'\riteration: {iteration + 1}'
                    for name, metric in train_step.metrics.items():
                        if 'val_' not in name :
                            string += f', {name}: {metric.result().numpy():.5e}'
                    print(string, end='')
                print()
                with open(self.log_path, 'a') as file:
                    file.write(f'{get_time_string()} Trainer: train: {string[1:]}\n')

                if self.n_validate_iterations:
                    for _ in range(self.n_validate_iterations):
                        validate_step_lambda(next(validate_dataset))
                        string = f'\riteration: {iteration + 1}'
                        for name, metric in train_step.metrics.items():
                            if 'val_' in name:
                                string += f', {name}: {metric.result().numpy():.5e}'
                        print(string, end='')
                    print()
                    with open(self.log_path, 'a') as file:
                        file.write(f'{get_time_string()} Trainer: validate: {string[1:]}\n')

                for _, metric in train_step.metrics.items():
                    metric.reset_states()
                checkpoint_path = self.checkpoint_manager.save()
                print(f'iteration: {iteration + 1}, save: {checkpoint_path}')

    @staticmethod
    def _distribute(dataset, strategy):
        if dataset:
            return iter(strategy.experimental_distribute_dataset(dataset))

    @staticmethod
    def _restore(models, optimizers, model_flag, optimizer_flag, path):
        if model_flag and optimizer_flag:
            objects = {}
            if model_flag:
                objects.update(models)
            if optimizer_flag:
                objects.update(optimizers)
            tf.train.Checkpoint(**objects).restore(tf.train.latest_checkpoint(path))
