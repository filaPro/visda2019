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
    def __init__(self, build_backbone_lambda, build_bottom_lambda):
        pass

    @tf.function
    def train(self, batch):
        pass
