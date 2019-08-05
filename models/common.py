import tensorflow as tf


class ClassificationLoss:
    def __init__(self):
        self.scorer = tf.keras.losses.SparseCategoricalCrossentropy()

    def __call__(self, labels, predictions):
        loss = .0
        for label, prediction in zip(labels, predictions):
            loss += self.scorer(label, prediction)
        loss /= len(labels)
        return loss
