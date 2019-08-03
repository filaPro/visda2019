import itertools
import tensorflow as tf


class Generator(tf.keras.Model):
    def __init__(self, image_size):
        super().__init__()
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(image_size, image_size, 3), 
            include_top=False, 
            weights='imagenet'
        )
        for layer in base_model.layers[:143]:
            layer.trainable = False
        self.model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1024, activation='relu'),
        ])

    def call(self, x):
        return self.model(x)


class Classifier(tf.keras.Model):
    def __init__(self, n_classes):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(n_classes, input_shape=(1024,), activation='softmax')
        ])

    def call(self, x):
        return self.model(x)


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


class M3sdaModel:
    def __init__(self, beta, n_classes, domains, image_size, n_moments):
        self.beta = beta
        self.domains = domains
        self.n_sources = len(domains) - 1
        self.iteration = tf.Variable(0)

        self.generator = Generator(image_size)
        self.source_classifiers = tuple(Classifier(n_classes) for _ in range(self.n_sources))
        self.target_classifiers = tuple(Classifier(n_classes) for _ in range(self.n_sources))
        self.optimizers = tuple(tf.keras.optimizers.Adam() for _ in range(1))
        self.trainable_variables = (
            self.generator.trainable_variables + list(itertools.chain(*(
                c.trainable_variables for c in self.source_classifiers
            ))),
            list(itertools.chain(*(c.trainable_variables for c in self.target_classifiers))),
            self.generator.trainable_variables
        )

        self.metrics = {
            'moment': tf.keras.metrics.Mean(),
            'classification': tf.keras.metrics.Mean(),
            'discrepancy': tf.keras.metrics.Mean(),
            'target_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        }
        for i in range(self.n_sources):
            self.metrics[f'source_accuracy_{domains[i]}'] = tf.keras.metrics.SparseCategoricalAccuracy()

        self.classification_scorer = ClassificationLoss()
        self.moment_scorer = MomentLoss(n_moments)
        self.discrepancy_scorer = DiscrepancyLoss()

    @tf.function
    # TODO: make generator and classifier calls sensitive to trainable
    # TODO: balance losses
    def train_step(self, batch):
        self.iteration.assign_add(1)

        with tf.GradientTape(persistent=True) as tape:
            sources_features = tuple(self.generator(batch[i][0]) for i in range(self.n_sources))
            target_features = self.generator(batch[-1][0])
            source_predictions = tuple(
                self.source_classifiers[i](sources_features[i]) for i in range(self.n_sources)
            )
            target_predictions = tuple(
                self.target_classifiers[i](target_features) for i in range(self.n_sources)
            )
            source_target_predictions = tuple(
                self.source_classifiers[i](target_features) for i in range(self.n_sources)
            )
            classification_loss = self.classification_scorer(
                tuple(zip(*batch[:self.n_sources]))[1], source_predictions
            )
            moment_loss = self.moment_scorer(sources_features, target_features)
            discrepancy_loss = self.discrepancy_scorer(source_target_predictions, target_predictions)
            losses = (
                classification_loss + moment_loss * .001,
                classification_loss - discrepancy_loss,
                discrepancy_loss
            )

        for i in range(1 if not self.beta else 3):
            self.optimizers[i].apply_gradients(zip(
                tape.gradient(losses[i], self.trainable_variables[i]),
                self.trainable_variables[i]
            ))

        self.metrics['moment'].update_state(moment_loss)
        self.metrics['classification'].update_state(classification_loss)
        self.metrics['discrepancy'].update_state(discrepancy_loss)
        self.metrics['target_accuracy'].update_state(
            batch[-1][1],
            tf.add_n(source_target_predictions if not self.beta else target_predictions)
        )
        for i in range(self.n_sources):
            self.metrics[f'source_accuracy_{self.domains[i]}'].update_state(batch[i][1], source_predictions[i])
