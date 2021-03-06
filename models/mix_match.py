import tensorflow as tf
import tensorflow_probability as tfp

from . import run_balanced


class MixMatchTrainStep:
    def __init__(
        self, build_backbone_lambda, build_top_lambda, learning_rate,
        loss_weight, temperature, alpha, global_batch_size, local_batch_size
    ):
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.alpha = alpha
        self.global_batch_size = global_batch_size
        self.local_batch_size = local_batch_size
        self.iteration = tf.Variable(
            0, name='iteration', dtype=tf.int64, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA
        )
        self.models = self._init_models(build_backbone_lambda, build_top_lambda)
        self.losses = self._init_losses()
        self.metrics = self._init_metrics()
        self.optimizers = self._init_optimizers(learning_rate)

    def train(self, batch):
        self.iteration.assign_add(1)

        mixed_source_images, mixed_source_labels, mixed_first_target_images, mixed_first_target_labels, \
            mixed_second_target_images, mixed_second_target_labels = self._mix_match(
                batch[0]['image'], batch[0]['label'], batch[-1]['image'][0], batch[-1]['image'][1]
            )

        with tf.GradientTape() as tape:
            source_predictions, first_target_predictions, second_target_predictions = run_balanced(
                models=(self.models['backbone'], self.models['top']),
                tensors=(mixed_source_images, mixed_first_target_images, mixed_second_target_images)
            )
            source_loss = self.losses['source'](mixed_source_labels, source_predictions)
            first_target_loss = self.losses['target'](mixed_first_target_labels, first_target_predictions)
            second_target_loss = self.losses['target'](mixed_second_target_labels, second_target_predictions)
            target_loss = (first_target_loss + second_target_loss) * .5
            loss = (source_loss + target_loss * self.loss_weight) / self.global_batch_size

        trainable_variables = self.models['backbone'].trainable_variables + self.models['top'].trainable_variables
        self.optimizers['optimizer'].apply_gradients(zip(
            tape.gradient(loss, trainable_variables), trainable_variables
        ))

        target_features = self.models['backbone'](batch[-1]['image'][1], training=False)
        target_predictions = self.models['top'](target_features, training=False)
        self.metrics['source_loss'].update_state(source_loss / self.local_batch_size)
        self.metrics['target_loss'].update_state(target_loss / self.local_batch_size)
        self.metrics['target_acc'].update_state(batch[-1]['label'], target_predictions)

    def _mix_match(self, source_images, source_labels, first_target_images, second_target_images):
        first_target_features = self.models['backbone'](first_target_images, training=True)
        second_target_features = self.models['backbone'](second_target_images, training=True)
        first_target_predictions = self.models['top'](first_target_features, training=True)
        second_target_predictions = self.models['top'](second_target_features, training=True)
        target_predictions = (first_target_predictions + second_target_predictions) / 2.
        target_predictions = self._sharpen(target_predictions, self.temperature)

        combined_images = tf.concat((source_images, first_target_images, second_target_images), axis=0)
        categorical_source_labels = tf.one_hot(source_labels, tf.shape(target_predictions)[1])
        combined_labels = tf.concat((categorical_source_labels, target_predictions, target_predictions), axis=0)
        indexes = tf.range(self.local_batch_size * 3)
        shuffled_indexes = tf.random.shuffle(indexes)
        shuffled_images = tf.gather(combined_images, shuffled_indexes)
        shuffled_labels = tf.gather(combined_labels, shuffled_indexes)

        mixed_images, mixed_labels = self._mix_up(
            first_images=combined_images,
            first_labels=combined_labels,
            second_images=shuffled_images,
            second_labels=shuffled_labels,
            alpha=self.alpha,
            batch_size=self.local_batch_size * 3
        )
        splitted_images = tf.split(mixed_images, 3)
        splitted_labels = tf.split(mixed_labels, 3)
        return (
            splitted_images[0], splitted_labels[0],
            splitted_images[1], splitted_labels[1],
            splitted_images[2], splitted_labels[2]
        )

    @staticmethod
    def _sharpen(x, temperature):
        powered = tf.pow(x, 1. / temperature)
        return powered / tf.reduce_sum(powered, axis=1, keepdims=True)

    @staticmethod
    def _mix_up(first_images, first_labels, second_images, second_labels, alpha, batch_size):
        distribution = tfp.distributions.Beta(alpha, alpha)
        unbounded_decay = distribution.sample(batch_size)
        decay = tf.maximum(unbounded_decay, 1. - unbounded_decay)
        image_decay = tf.reshape(decay, (-1, 1, 1, 1))
        images = first_images * image_decay + second_images * (1. - image_decay)
        label_decay = tf.reshape(decay, (-1, 1))
        labels = first_labels * label_decay + second_labels * (1. - label_decay)
        return images, labels

    @staticmethod
    def _init_models(build_backbone_lambda, build_top_lambda):
        return {
            'backbone': build_backbone_lambda(),
            'top': build_top_lambda()
        }

    @staticmethod
    def _init_losses():
        return {
            'source': tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM),
            'target': tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        }

    @staticmethod
    def _init_metrics():
        return {
            'source_loss': tf.keras.metrics.Mean(),
            'target_loss': tf.keras.metrics.Mean(),
            'target_acc': tf.keras.metrics.SparseCategoricalAccuracy()
        }

    @staticmethod
    def _init_optimizers(learning_rate):
        return {
            'optimizer': tf.keras.optimizers.Adam(learning_rate)
        }
