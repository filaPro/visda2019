import tensorflow as tf


def build_generator(image_size):
    return tf.keras.Sequential([
        tf.keras.applications.MobileNetV2(
            input_shape=(image_size, image_size, 3),
            include_top=False,
            weights='imagenet'
        ),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4096, activation='relu'),
    ])


def build_classifer(n_classes):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(n_classes, input_shape=(4096,), activation='softmax')
    ])


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


class ClassificationLoss:
    def __init__(self):
        self.scorer = tf.keras.losses.SparseCategoricalCrossentropy()

    def __call__(self, labels, predictions):
        loss = .0
        for label, prediction in zip(labels, predictions):
            loss += self.scorer(label, prediction)
        loss /= len(labels)
        return loss


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
    def __init__(self, n_classes, domains, image_size, n_moments, n_frozen_layers, learning_rate, loss_weight):
        self.n_sources = len(domains) - 1
        self.domains = domains
        self.loss_weight = loss_weight
        self.iteration = tf.Variable(0, name='iteration')
        self.models = self._init_models(image_size, n_frozen_layers, n_classes)
        self.losses = self._init_losses(n_moments)
        self.metrics = self._init_metrics()
        self.optimizers = self._init_optimizers(learning_rate)

    @tf.function
    def __call__(self, batch):
        self.iteration.assign_add(1)

        with tf.GradientTape() as tape:
            sources_features = tuple(self.models['generator'](batch[i][0]) for i in range(self.n_sources))
            target_features = self.models['generator'](batch[-1][0])
            source_predictions = tuple(
                self.models[f'classifier_{i}'](sources_features[i]) for i in range(self.n_sources)
            )
            target_predictions = tuple(
                self.models[f'classifier_{i}'](target_features) for i in range(self.n_sources)
            )
            classification_loss = self.losses['classification'](
                tuple(zip(*batch[:self.n_sources]))[1], source_predictions
            )
            moment_loss = self.losses['moment'](sources_features, target_features)
            loss = classification_loss + moment_loss * self.loss_weight

        trainable_variables = self.models['generator'].trainable_variables
        for i in range(self.n_sources):
            trainable_variables += self.models[f'classifier_{i}'].trainable_variables
        self.optimizers['optimizer'].apply_gradients(zip(tape.gradient(loss, trainable_variables), trainable_variables))

        self.metrics['moment'].update_state(moment_loss)
        self.metrics['classification'].update_state(classification_loss)
        self.metrics[f'target_accuracy'].update_state(batch[-1][1], tf.add_n(target_predictions))
        for i in range(self.n_sources):
            self.metrics[f'{self.domains[i]}_accuracy'].update_state(batch[i][1], source_predictions[i])

    def _init_models(self, image_size, n_frozen_layers, n_classes):
        models = {
            'generator': build_generator(image_size)
        }
        backbone = models['generator'].layers[0]
        for layer in backbone.layers[:n_frozen_layers]:
            layer.trainable = False
        for i in range(self.n_sources):
            models[f'classifier_{i}'] = build_classifer(n_classes)
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


class M3sdaTestStep:
    def __init__(self, n_classes, domains, image_size):
        self.n_sources = len(domains) - 1
        self.iteration = tf.Variable(0, name='iteration')
        self.models = self._init_models(n_classes, image_size)
        self.metrics = self._init_metrics()

    @tf.function
    def __call__(self, batch):
        self.iteration.assign_add(1)
        features = self.models['generator'](batch)
        predictions = tuple(
            self.models[f'classifier_{i}'](features) for i in range(self.n_sources)
        )
        self.metrics[f'accuracy'].update_state(batch[-1][1], tf.add_n(predictions))

    def _init_models(self, image_size, n_classes):
        models = {
            'generator': build_generator(image_size)
        }
        for i in range(self.n_sources):
            models[f'classifier_{i}'] = build_classifer(n_classes)
        return models

    @staticmethod
    def _init_metrics():
        return {
            'accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        }
