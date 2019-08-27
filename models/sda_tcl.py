import tensorflow as tf
from functools import partial


class Centers(tf.keras.layers.Layer):
    def __init__(self, n_classes, feature_size):
        super().__init__()
        self.centers = self.add_weight(
            shape=(n_classes, feature_size),
            trainable=True,
            initializer='glorot_uniform'
        )
        self.epsilon = 1e-6

    def __call__(self, points):
        squared_points = tf.reshape(tf.reduce_sum(tf.square(points), 1), (-1, 1))
        squared_centers = tf.reshape(tf.reduce_sum(tf.square(self.centers), 1), (-1, 1))
        return tf.sqrt(
            squared_points - 2 * tf.matmul(points, self.centers, False, True) + squared_centers + self.epsilon
        )


def build_centers(n_classes, feature_size):
    return tf.keras.models.Sequential([Centers(n_classes, feature_size)])


class SdaTclTrainStep:
    def __init__(
        self, build_backbone_lambda, build_bottom_lambda, build_discriminator_lambda, backbone_training_flag,
        backbone_learning_rate, learning_rate, source_loss_weight, target_loss_weight, discriminator_loss_weight,
        positive_margin, negative_margin, n_classes, feature_size, batch_size
    ):
        self.backbone_training_flag = backbone_training_flag
        self.source_loss_weight = source_loss_weight
        self.target_loss_weight = target_loss_weight
        self.discriminator_loss_weight = discriminator_loss_weight
        self.positive_margin = positive_margin
        self.negative_margin = negative_margin
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.iteration = tf.Variable(
            0, name='iteration', dtype=tf.int64, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA
        )
        build_centers_lambda = partial(build_centers, n_classes=n_classes, feature_size=feature_size)
        self.models = self._init_models(
            build_backbone_lambda, build_bottom_lambda, build_discriminator_lambda, build_centers_lambda
        )
        self.losses = self._init_losses()
        self.metrics = self._init_metrics()
        self.optimizers = self._init_optimizers(backbone_learning_rate, learning_rate)

    def train(self, batch):
        self.iteration.assign_add(1)

        with tf.GradientTape() as tape, tf.GradientTape() as backbone_tape:
            source_bottom_features = self.models['backbone'](
                batch[0]['image'], training=self.backbone_training_flag
            )
            target_bottom_features = self.models['backbone'](
                batch[-1]['image'], training=self.backbone_training_flag
            )
            source_features = self.models['bottom'](source_bottom_features, training=True)
            target_features = self.models['bottom'](target_bottom_features, training=True)
            source_discriminator_predictions = self.models['discriminator'](source_features, training=True)
            target_discriminator_predictions = self.models['discriminator'](target_features, training=True)
            source_distances = self.models['source_centers'](target_features, training=True)
            target_distances = self.models['target_centers'](target_features, training=True)

            positive_mask = tf.one_hot(batch[0]['label'], self.n_classes)
            source_positive_loss = tf.nn.relu(source_distances[tf.cast(positive_mask, tf.bool)] - self.positive_margin)
            source_positive_loss = tf.reduce_sum(source_positive_loss) / self.batch_size

            negative_distance = tf.reduce_min(source_distances + tf.reduce_max(source_distances) * positive_mask, axis=1)


    @staticmethod
    def _init_models(build_backbone_lambda, build_bottom_lambda, build_discriminator_lambda, build_centers_lambda):
        return {
            'backbone': build_backbone_lambda(),
            'bottom': build_bottom_lambda(),
            'discriminator': build_discriminator_lambda(),
            'source_centers': build_centers_lambda(),
            'target_centers': build_centers_lambda()
        }

    @staticmethod
    def _init_losses():
        return {}

    @staticmethod
    def _init_metrics():
        return {
            'source_acc': tf.keras.metrics.Accuracy(),
            'target_acc': tf.keras.metrics.Accuracy()
        }

    @staticmethod
    def _init_optimizers(backbone_learning_rate, learning_rate):
        return {
            'backbone_optimizer': tf.keras.optimizers.Adam(backbone_learning_rate),
            'optimizer': tf.keras.optimizers.Adam(learning_rate)
        }
