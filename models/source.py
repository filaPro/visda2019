import tensorflow as tf

from .common import ClassificationLoss


class SourceTrainStep:
    def __init__(self, build_model_lambda, domains, n_frozen_layers, learning_rate):
        self.n_sources = len(domains) - 1
        self.domains = domains
        self.iteration = tf.Variable(0, name='iteration')
        self.models = self._init_models(build_model_lambda, n_frozen_layers)
        self.losses = self._init_losses()
        self.metrics = self._init_metrics()
        self.optimizers = self._init_optimizers(learning_rate)

    @tf.function
    def train(self, batch):
        self.iteration.assign_add(1)

        with tf.GradientTape() as tape:
            source_predictions = tuple(self.models['model'](batch[i][0]) for i in range(self.n_sources))
            target_predictions = self.models['model'](batch[-1][0])
            loss = self.losses['classification'](
                tuple(zip(*batch[:self.n_sources]))[1], source_predictions
            )

        trainable_variables = self.models['model'].trainable_variables
        self.optimizers['optimizer'].apply_gradients(zip(tape.gradient(loss, trainable_variables), trainable_variables))

        self.metrics['scce'].update_state(loss)
        self.metrics['target_acc'].update_state(batch[-1][1], target_predictions)
        for i in range(self.n_sources):
            self.metrics[f'{self.domains[i]}_acc'].update_state(batch[i][1], source_predictions[i])

    @tf.function
    def validate(self, batch):
        predictions = tuple(self.models['model'](batch[i][0]) for i in range(self.n_sources))
        for i in range(self.n_sources):
            self.metrics[f'{self.domains[i]}_val_acc'].update_state(batch[i][1], predictions[i])

    @staticmethod
    def _init_models(build_model_lambda, n_frozen_layers):
        models = {
            'model': build_model_lambda()
        }
        backbone = models['model'].layers[0]
        for layer in backbone.layers[:n_frozen_layers]:
            layer.trainable = False
        return models

    @staticmethod
    def _init_losses():
        return {
            'classification': ClassificationLoss()
        }

    def _init_metrics(self):
        metrics = {
            'scce': tf.keras.metrics.Mean(),
            'target_acc': tf.keras.metrics.SparseCategoricalAccuracy(),
        }
        for i in range(self.n_sources):
            metrics[f'{self.domains[i]}_acc'] = tf.keras.metrics.SparseCategoricalAccuracy()
            metrics[f'{self.domains[i]}_val_acc'] = tf.keras.metrics.SparseCategoricalAccuracy()
        return metrics

    @staticmethod
    def _init_optimizers(learning_rate):
        return {
            'optimizer': tf.keras.optimizers.Adam(learning_rate)
        }


class SourceTestStep:
    def __init__(self, build_model_lambda, domains):
        self.n_sources = len(domains) - 1
        self.iteration = tf.Variable(0, name='iteration')
        self.models = self._init_models(build_model_lambda)
        self.metrics = self._init_metrics()

    @tf.function
    def __call__(self, batch):
        self.iteration.assign_add(1)
        predictions = self.models['model'](batch[0])
        self.metrics['acc'].update_state(batch[1], predictions)

    @staticmethod
    def _init_models(build_model_lambda):
        return {
            'model': build_model_lambda()
        }

    @staticmethod
    def _init_metrics():
        return {
            'acc': tf.keras.metrics.SparseCategoricalAccuracy()
        }
