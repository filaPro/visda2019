import tensorflow as tf

from .common import ClassificationLoss


class DomainClassifierTrainStep:
    def __init__(
        self, build_discriminator_lambda, build_generator_lambda, build_classifier_lambda, domains, n_frozen_layers,
        learning_rate, loss_weight
    ):
        self.n_sources = len(domains) - 1
        self.domains = domains
        self.loss_weight = loss_weight
        self.iteration = tf.Variable(0, name='iteration')
        self.models = self._init_models(
            build_discriminator_lambda, build_generator_lambda, build_classifier_lambda, n_frozen_layers
        )
        self.losses = self._init_losses()
        self.metrics = self._init_metrics()
        self.optimizers = self._init_optimizers(learning_rate)

    @tf.function
    def train(self, batch):
        self.iteration.assign_add(1)

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            source_features = tuple(self.models['generator'](batch[i][0], training=True) for i in range(self.n_sources))
            target_features = self.models['generator'](batch[-1][0], training=True)
            source_predictions = tuple(
                self.models[f'classifier_{i}'](source_features[i], training=True) for i in range(self.n_sources)
            )
            target_predictions = tuple(
                self.models[f'classifier_{i}'](target_features, training=True) for i in range(self.n_sources)
            )
            source_discriminator_predictions = tuple(
                self.models['discriminator'](source_features[i]) for i in range(self.n_sources)
            )
            target_discriminator_predictions = self.models['discriminator'](target_features, training=True)
            discriminator_predictions = source_discriminator_predictions + (target_discriminator_predictions,)
            discriminator_labels = tuple(
                tf.ones((discriminator_predictions[i].shape[0],), dtype=tf.uint8)for i in range(self.n_sources + 1)
            )
            classification_loss = self.losses['classification'](
                tuple(zip(*batch[:self.n_sources]))[1], source_predictions
            )
            domain_loss = self.losses['domain'](discriminator_labels, discriminator_predictions)
            generator_loss = classification_loss - domain_loss * self.loss_weight
            discriminator_loss = domain_loss * self.loss_weight

        generator_trainable_variables = self.models['generator'].trainable_variables
        for i in range(self.n_sources):
            generator_trainable_variables += self.models[f'classifier_{i}'].trainable_variables
        self.optimizers['generator_optimizer'].apply_gradients(zip(
            generator_tape.gradient(generator_loss, generator_trainable_variables),
            generator_trainable_variables
        ))
        discriminator_trainable_variables = self.models['discriminator'].trainable_variables
        self.optimizers['discriminator_optimizer'].apply_gradients(zip(
            discriminator_tape.gradient(discriminator_loss, discriminator_trainable_variables),
            discriminator_trainable_variables
        ))

        self.metrics['domain'].update_state(domain_loss)
        self.metrics['scce'].update_state(classification_loss)
        self.metrics[f'target_acc'].update_state(batch[-1][1], tf.add_n(target_predictions))
        for i in range(self.n_sources):
            self.metrics[f'{self.domains[i]}_acc'].update_state(batch[i][1], source_predictions[i])
        for i in range(self.n_sources + 1):
            self.metrics['domain_acc'].update_state(discriminator_labels[i], discriminator_predictions[i])

    @tf.function
    def validate(self, batch):
        source_features = tuple(self.models['generator'](batch[i][0], training=False) for i in range(self.n_sources))
        source_predictions = tuple(
            self.models[f'classifier_{i}'](source_features[i], training=False) for i in range(self.n_sources)
        )
        for i in range(self.n_sources):
            self.metrics[f'{self.domains[i]}_val_acc'].update_state(batch[i][1], source_predictions[i])

    def _init_models(
        self, build_discriminator_lambda, build_generator_lambda, build_classifier_lambda, n_frozen_layers
    ):
        models = {
            'generator': build_generator_lambda(),
            'discriminator': build_discriminator_lambda()
        }
        backbone = models['generator'].layers[0]
        for layer in backbone.layers[:n_frozen_layers]:
            layer.trainable = False
        for i in range(self.n_sources):
            models[f'classifier_{i}'] = build_classifier_lambda()
        return models

    @staticmethod
    def _init_losses():
        return {
            'classification': ClassificationLoss(),
            'domain': ClassificationLoss()
        }

    def _init_metrics(self):
        metrics = {
            'scce': tf.keras.metrics.Mean(),
            'target_acc': tf.keras.metrics.SparseCategoricalAccuracy(),
            'domain': tf.keras.metrics.Mean(),
            'domain_acc': tf.keras.metrics.SparseCategoricalAccuracy()
        }
        for i in range(self.n_sources):
            metrics[f'{self.domains[i]}_acc'] = tf.keras.metrics.SparseCategoricalAccuracy()
            metrics[f'{self.domains[i]}_val_acc'] = tf.keras.metrics.SparseCategoricalAccuracy()
        return metrics

    @staticmethod
    def _init_optimizers(learning_rate):
        return {
            'generator_optimizer': tf.keras.optimizers.Adam(learning_rate),
            'discriminator_optimizer': tf.keras.optimizers.Adam(learning_rate),
        }


class DomainClassifierTestStep:
    def __init__(self, build_generator_lambda, build_classifier_lambda, domains):
        self.n_sources = len(domains) - 1
        self.iteration = tf.Variable(0, name='iteration')
        self.models = self._init_models(build_generator_lambda, build_classifier_lambda)
        self.metrics = self._init_metrics()

    @tf.function
    def test(self, batch):
        self.iteration.assign_add(1)
        features = self.models['generator'](batch[0], training=False)
        predictions = tuple(
            self.models[f'classifier_{i}'](features, training=False) for i in range(self.n_sources)
        )
        self.metrics[f'acc'].update_state(batch[1], tf.add_n(predictions))

    def _init_models(self, build_generator_lambda, build_classifier_lambda):
        models = {
            'generator': build_generator_lambda()
        }
        for i in range(self.n_sources):
            models[f'classifier_{i}'] = build_classifier_lambda()
        return models

    @staticmethod
    def _init_metrics():
        return {
            'acc': tf.keras.metrics.SparseCategoricalAccuracy()
        }
