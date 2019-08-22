import tensorflow as tf


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


class SdaTclTrainStep:
    def __init__(
        self, build_backbone_lambda, build_bottom_lambda, build_discriminator_lambda, domains, backbone_training_flag,
        backbone_learning_rate, bottom_learning_rate, loss_weight, batch_size
    ):
        self.n_sources = len(domains) - 1
        self.domains = domains
        self.backbone_training_flag = backbone_training_flag
        self.loss_weight = loss_weight
        self.batch_size = batch_size
        self.iteration = tf.Variable(
            0, name='iteration', dtype=tf.int64, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA
        )
        self.models = self._init_models(build_backbone_lambda, build_bottom_lambda)
        self.losses = self._init_losses()
        self.metrics = self._init_metrics()
        self.optimizers = self._init_optimizers(backbone_learning_rate, bottom_learning_rate)

    def train(self, batch):
        pass

    @staticmethod
    def _init_models(build_backbone_lambda, build_bottom_lambda, build_discriminator_lambda):
        models = {
            'backbone': build_backbone_lambda(),
            'bottom': build_bottom_lambda(),
            'discriminator': build_discriminator_lambda()
        }
        return models