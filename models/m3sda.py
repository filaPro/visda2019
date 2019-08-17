import tensorflow as tf

from .common import ClassificationLoss


class MomentLoss:
    def __init__(self, n_moments):
        self.n_moments = n_moments

    def __call__(self, sources, target):
        loss = .0
        for i in range(1, self.n_moments + 1):
            loss += self._moment_loss(sources, target, i)
        return loss

    def _moment_loss(self, sources, target, moment):
        n_sources = len(sources)
        source_target_loss = .0
        for source in sources:
            source_target_loss += tf.norm(tf.subtract(
                self._moment(source, moment),
                self._moment(target, moment)
            ))
        source_target_loss /= n_sources
        source_source_loss = .0
        for i in range(n_sources - 1):
            for j in range(i + 1, n_sources):
                source_source_loss += tf.norm(tf.subtract(
                    self._moment(sources[i], moment),
                    self._moment(sources[j], moment)
                ))
        source_source_loss /= n_sources * (n_sources - 1) / 2
        return source_target_loss + source_source_loss

    @staticmethod
    # TODO: check axis of reduce_mean
    # TODO: central moments in original implementation
    def _moment(tensor, moment):
        return tf.reduce_mean(tf.pow(tensor, moment), axis=0)


class DiscrepancyLoss:
    def __init__(self):
        self.scorer = tf.keras.losses.MeanAbsoluteError()

    def __call__(self, labels, predictions):
        loss = .0
        for label, prediction in zip(labels, predictions):
            loss += self.scorer(label, prediction)
        loss /= len(labels)
        return loss


class M3sdaTrainStep:
    def __init__(
        self, build_backbone_lambda, build_bottom_lambda, build_top_lambda, domains, n_moments, freeze_backbone_flag,
        backbone_training_flag, learning_rate, loss_weight
    ):
        self.n_sources = len(domains) - 1
        self.domains = domains
        self.loss_weight = loss_weight
        self.backbone_training_flag = backbone_training_flag
        self.iteration = tf.Variable(0, name='iteration')
        self.models = self._init_models(
            build_backbone_lambda=build_backbone_lambda,
            build_bottom_lambda=build_bottom_lambda,
            build_top_lambda=build_top_lambda,
            freeze_backbone_flag=freeze_backbone_flag
        )
        self.losses = self._init_losses(n_moments)
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
                self.models['bottom'](source_bottom_features[i], training=True) for i in range(self.n_sources)
            )
            target_top_features = self.models['bottom'](target_bottom_features, training=True)
            source_predictions = tuple(
                self.models[f'top_{i}'](source_top_features[i], training=True) for i in range(self.n_sources)
            )
            classification_loss = self.losses['classification'](
                tuple(zip(*batch[:self.n_sources]))[1], source_predictions
            )
            moment_loss = self.losses['moment'](source_top_features, target_top_features)
            loss = classification_loss + moment_loss * self.loss_weight

        trainable_variables = self.models['backbone'].trainable_variables + self.models['bottom'].trainable_variables
        for i in range(self.n_sources):
            trainable_variables += self.models[f'top_{i}'].trainable_variables
        self.optimizers['optimizer'].apply_gradients(zip(tape.gradient(loss, trainable_variables), trainable_variables))

        target_predictions = tuple(
            self.models[f'top_{i}'](target_top_features, training=False) for i in range(self.n_sources)
        )
        self.metrics['moment'].update_state(moment_loss)
        self.metrics['scce'].update_state(classification_loss)
        self.metrics[f'target_acc'].update_state(batch[-1][1], tf.add_n(target_predictions))
        for i in range(self.n_sources):
            self.metrics[f'{self.domains[i]}_acc'].update_state(batch[i][1], source_predictions[i])

    def _init_models(self, build_backbone_lambda, build_bottom_lambda, build_top_lambda, freeze_backbone_flag):
        models = {
            'backbone': build_backbone_lambda(),
            'bottom': build_bottom_lambda()
        }
        if freeze_backbone_flag:
            for layer in models['backbone'].layers:
                layer.trainable = False
        for i in range(self.n_sources):
            models[f'top_{i}'] = build_top_lambda()
        return models

    @staticmethod
    def _init_losses(n_moments):
        return {
            'classification': ClassificationLoss(),
            'moment': MomentLoss(n_moments)
        }

    def _init_metrics(self):
        metrics = {
            'moment': tf.keras.metrics.Mean(),
            'scce': tf.keras.metrics.Mean(),
            'target_acc': tf.keras.metrics.SparseCategoricalAccuracy(),
        }
        for i in range(self.n_sources):
            metrics[f'{self.domains[i]}_acc'] = tf.keras.metrics.SparseCategoricalAccuracy()
        return metrics

    @staticmethod
    def _init_optimizers(learning_rate):
        return {
            'optimizer': tf.keras.optimizers.Adam(learning_rate)
        }


class M3sdaTestStep:
    def __init__(self, build_backbone_lambda, build_bottom_lambda, build_top_lambda, domains):
        self.n_sources = len(domains) - 1
        self.iteration = tf.Variable(0, name='iteration')
        self.models = self._init_models(build_backbone_lambda, build_bottom_lambda, build_top_lambda)
        self.metrics = self._init_metrics()

    @tf.function
    def test(self, batch):
        self.iteration.assign_add(1)
        bottom_features = self.models['backbone'](batch[0], training=False)
        top_features = self.models['bottom'](bottom_features, training=False)
        predictions = tuple(
            self.models[f'top_{i}'](top_features, training=False) for i in range(self.n_sources)
        )
        self.metrics[f'acc'].update_state(batch[1], tf.add_n(predictions))

    def _init_models(self, build_backbone_lambda, build_bottom_lambda, build_top_lambda):
        models = {
            'backbone': build_backbone_lambda(),
            'bottom': build_bottom_lambda()
        }
        for i in range(self.n_sources):
            models[f'top_{i}'] = build_top_lambda()
        return models

    @staticmethod
    def _init_metrics():
        return {
            'acc': tf.keras.metrics.SparseCategoricalAccuracy()
        }
