import tensorflow as tf

from .common import ClassificationLoss


class GradientReverse(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return self._reverse_gradient(x)

    @staticmethod
    @tf.custom_gradient
    def _reverse_gradient(x):
        return x, lambda d: -d


class DannTrainStep:
    def __init__(
        self, build_backbone_lambda, build_bottom_lambda, build_top_lambda, build_discriminator_lambda, domains,
        freeze_backbone_flag, backbone_training_flag, learning_rate, loss_weight
    ):
        self.n_sources = len(domains) - 1
        self.domains = domains
        self.backbone_training_flag = backbone_training_flag
        self.loss_weight = loss_weight
        self.iteration = tf.Variable(0, name='iteration')
        self.models = self._init_models(
            build_backbone_lambda=build_backbone_lambda,
            build_bottom_lambda=build_bottom_lambda,
            build_top_lambda=build_top_lambda,
            build_discriminator_lambda=build_discriminator_lambda,
            freeze_backbone_flag=freeze_backbone_flag
        )
        self.losses = self._init_losses()
        self.metrics = self._init_metrics()
        self.optimizers = self._init_optimizers(learning_rate)

    @tf.function
    def train(self, batch):
        self.iteration.assign_add(1)

        with tf.GradientTape() as tape:
            source_bottom_features = tuple(
                self.models['backbone'](batch[i][0], training=self.backbone_training_flag)
                for i in range(self.n_sources)
            )
            target_bottom_features = self.models['backbone'](batch[-1][0], training=self.backbone_training_flag)
            source_top_features = tuple(
                self.models['bottom'](source_bottom_features[i], training=True)
                for i in range(self.n_sources)
            )
            target_top_features = self.models['bottom'](target_bottom_features, training=True)
            source_predictions = tuple(
                self.models['top'](source_top_features[i], training=True)
                for i in range(self.n_sources)
            )
            source_discriminator_predictions = tuple(
                self.models['discriminator'](source_top_features[i], training=True) for i in range(self.n_sources)
            )
            target_discriminator_predictions = self.models['discriminator'](target_top_features, training=True)
            discriminator_predictions = source_discriminator_predictions + (target_discriminator_predictions,)
            discriminator_labels = tuple(
                tf.ones((discriminator_predictions[i].shape[0],), dtype=tf.uint8) * i for i in range(self.n_sources + 1)
            )
            classification_loss = self.losses['classification'](
                tuple(zip(*batch[:self.n_sources]))[1], source_predictions
            )
            domain_loss = self.losses['domain'](discriminator_labels, discriminator_predictions)
            loss = classification_loss + domain_loss * self.loss_weight

        trainable_variables = self.models['backbone'].trainable_variables + \
            self.models['bottom'].trainable_variables + self.models['top'].trainable_variables + \
            self.models['discriminator'].trainable_variables
        self.optimizers['optimizer'].apply_gradients(zip(tape.gradient(loss, trainable_variables), trainable_variables))

        target_predictions = self.models['top'](target_top_features, training=False)
        self.metrics['scce'].update_state(classification_loss)
        self.metrics['domain'].update_state(domain_loss)
        self.metrics['target_acc'].update_state(batch[-1][1], target_predictions)
        for i in range(self.n_sources):
            self.metrics[f'{self.domains[i]}_acc'].update_state(batch[i][1], source_predictions[i])
        for i in range(self.n_sources + 1):
            self.metrics['domain_acc'].update_state(discriminator_labels[i], discriminator_predictions[i])

    @staticmethod
    def _init_models(
        build_backbone_lambda, build_bottom_lambda, build_top_lambda, build_discriminator_lambda, freeze_backbone_flag
    ):
        models = {
            'backbone': build_backbone_lambda(),
            'bottom': build_bottom_lambda(),
            'top': build_top_lambda(),
            'discriminator': build_discriminator_lambda()
        }
        if freeze_backbone_flag:
            for layer in models['backbone'].layers:
                layer.trainable = False
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
        return metrics

    @staticmethod
    def _init_optimizers(learning_rate):
        return {
            'optimizer': tf.keras.optimizers.Adam(learning_rate)
        }
