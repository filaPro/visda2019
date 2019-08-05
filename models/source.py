import tensorflow as tf

from .common import ClassificationLoss


def build_model(image_size, n_classes):
    return tf.keras.Sequential([
        tf.keras.applications.MobileNetV2(
            input_shape=(image_size, image_size, 3),
            include_top=False,
            weights='imagenet'
        ),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(n_classes, input_shape=(4096,), activation='softmax')
    ])


class SourceTrainStep:
    def __init__(self, n_classes, domains, image_size, n_frozen_layers, learning_rate):
        self.n_sources = len(domains) - 1
        self.domains = domains
        self.iteration = tf.Variable(0, name='iteration')
        self.models = self._init_models(image_size, n_frozen_layers, n_classes)
        self.losses = self._init_losses()
        self.metrics = self._init_metrics()
        self.optimizers = self._init_optimizers(learning_rate)

    @tf.function
    def __call__(self, batch):
        self.iteration.assign_add(1)

        with tf.GradientTape() as tape:
            source_predictions = tuple(self.models['model'](batch[i][0]) for i in range(self.n_sources))
            target_predictions = self.models['model'](batch[-1][0])
            loss = self.losses['classification'](
                tuple(zip(*batch[:self.n_sources]))[1], source_predictions
            )

        trainable_variables = self.models['model'].trainable_variables
        self.optimizers['optimizer'].apply_gradients(zip(tape.gradient(loss, trainable_variables), trainable_variables))

        self.metrics['classification'].update_state(loss)
        self.metrics['target_accuracy'].update_state(batch[-1][1], target_predictions)
        for i in range(self.n_sources):
            self.metrics[f'{self.domains[i]}_accuracy'].update_state(batch[i][1], source_predictions[i])

    @staticmethod
    def _init_models(image_size, n_frozen_layers, n_classes):
        models = {
            'model': build_model(image_size, n_classes)
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
            'classification': tf.keras.metrics.Mean(),
            'target_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        }
        for i in range(self.n_sources):
            metrics[f'{self.domains[i]}_accuracy'] = tf.keras.metrics.SparseCategoricalAccuracy()
        return metrics

    @staticmethod
    def _init_optimizers(learning_rate):
        return {
            'optimizer': tf.keras.optimizers.Adam(learning_rate)
        }


class SourceTestStep:
    def __init__(self, n_classes, domains, image_size):
        self.n_sources = len(domains) - 1
        self.iteration = tf.Variable(0, name='iteration')
        self.models = self._init_models(image_size, n_classes)
        self.metrics = self._init_metrics()

    @tf.function
    def __call__(self, batch):
        self.iteration.assign_add(1)
        predictions = self.models['model'](batch[0])
        self.metrics[f'accuracy'].update_state(batch[1], predictions)

    @staticmethod
    def _init_models(image_size, n_classes):
        return {
            'model': build_model(image_size, n_classes)
        }

    @staticmethod
    def _init_metrics():
        return {
            'accuracy': tf.keras.metrics.SparseCategoricalAccuracy()
        }
