import tensorflow as tf


class Classifier(tf.keras.Model):
    def __init__(self, image_size, n_frozen_layers, n_classes):
        super().__init__()
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(image_size, image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        for layer in base_model.layers[:n_frozen_layers]:
            layer.trainable = False
        self.model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(n_classes, activation='softmax')
        ])

    def call(self, x):
        return self.model(x)


class SourceModel:
    pass