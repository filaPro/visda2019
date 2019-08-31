import tensorflow as tf

from .common import ClassificationLoss, run_balanced


class GradientReverse(tf.keras.layers.Layer):
    def __init__(self, multiplier):
        super().__init__()
        self.multiplier = multiplier

    def call(self, x):
        return self._reverse_gradient(x, self.multiplier)

    @staticmethod
    @tf.custom_gradient
    def _reverse_gradient(x, multiplier):
        return x, lambda d: -d * multiplier


class DannTrainStep:
    def __init__(
        self, build_backbone_lambda, build_top_lambda, build_discriminator_lambda, domains, learning_rate, loss_weight,
        batch_size
    ):
        self.n_sources = len(domains) - 1
        self.domains = domains
        self.loss_weight = loss_weight
        self.batch_size = batch_size
        self.iteration = self.iteration = tf.Variable(
            0, name='iteration', dtype=tf.int64, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA
        )
        self.models = self._init_models(
            build_backbone_lambda=build_backbone_lambda,
            build_top_lambda=build_top_lambda,
            build_discriminator_lambda=build_discriminator_lambda
        )
        self.losses = self._init_losses()
        self.metrics = self._init_metrics()
        self.optimizers = self._init_optimizers(learning_rate)

    def train(self, batch):
        self.iteration.assign_add(1)

        with tf.GradientTape() as tape:
            features = run_balanced(
                models=(self.models['backbone'],),
                tensors=tuple(batch[i]['image'] for i in range(self.n_sources + 1))
            )
            source_predictions = tuple(
                self.models['top'](features[i], training=True) for i in range(self.n_sources)
            )
            discriminator_predictions = tuple(
                self.models['discriminator'](features[i], training=True) for i in range(self.n_sources + 1)
            )
            discriminator_labels = tuple(
                tf.ones_like(discriminator_predictions[i][:, 0], dtype=tf.uint8) * i
                for i in range(self.n_sources + 1)
            )
            classification_loss = self.losses['classification'](
                tuple(batch[i]['label'] for i in range(self.n_sources)), source_predictions
            )
            classification_loss /= self.batch_size
            domain_loss = self.losses['domain'](discriminator_labels, discriminator_predictions)
            domain_loss /= self.batch_size
            loss = classification_loss + domain_loss * self.loss_weight

        trainable_variables = self.models['backbone'].trainable_variables + \
            self.models['top'].trainable_variables + \
            self.models['discriminator'].trainable_variables
        self.optimizers['optimizer'].apply_gradients(zip(tape.gradient(loss, trainable_variables), trainable_variables))

        target_predictions = self.models['top'](features[-1], training=False)
        self.metrics['scce'].update_state(classification_loss)
        self.metrics['domain'].update_state(domain_loss)
        self.metrics['target_acc'].update_state(batch[-1]['label'], target_predictions)
        for i in range(self.n_sources):
            self.metrics[f'{self.domains[i]}_acc'].update_state(batch[i]['label'], source_predictions[i])
        for i in range(self.n_sources + 1):
            self.metrics['domain_acc'].update_state(discriminator_labels[i], discriminator_predictions[i])

    @staticmethod
    def _init_models(build_backbone_lambda, build_top_lambda, build_discriminator_lambda):
        return {
            'backbone': build_backbone_lambda(),
            'top': build_top_lambda(),
            'discriminator': build_discriminator_lambda()
        }

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
        return metrics

    @staticmethod
    def _init_optimizers(learning_rate):
        return {
            'optimizer': tf.keras.optimizers.Adam(learning_rate)
        }
