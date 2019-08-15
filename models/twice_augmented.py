import tensorflow as tf

from .common import ClassificationLoss
from preprocessor import Preprocessor


class TwiceAugmentedPreprocessor:
    def __init__(self, first_config, second_config):
        self.first_preprocessor = Preprocessor(first_config)
        self.second_preprocessor = Preprocessor(second_config)

    @tf.function
    def __call__(self, image):
        return self.first_preprocessor(image), self.second_preprocessor(image)


# def make_twice_augmented_dataset(
#     source_paths, source_labels, source_config,
#     target_paths, target_labels, first_target_config, second_target_config,
#     batch_size
# ):
#     source_preprocessor = Preprocessor(source_config)
#     first_target_preprocessor = Preprocessor(first_target_config)
#     second_target_preprocessor = Preprocessor(second_target_config)
#     datasets = []
#     for paths, labels in zip(source_paths, source_labels):
#         datasets.append(make_domain_dataset(paths, labels, source_preprocessor, batch_size, None))
#     datasets.append(make_domain_dataset(target_paths, target_labels, first_target_preprocessor, batch_size, 42))
#     datasets.append(make_domain_dataset(target_paths, target_labels, second_target_preprocessor, batch_size, 42))
#     return tf.data.Dataset.zip(tuple(datasets)).repeat()


class TwiceAugmentedTrainStep:
    def __init__(
        self, build_backbone_lambda, build_top_lambda, domains, freeze_backbone_flag, backbone_training_flag,
        learning_rate, loss_weight
    ):
        self.n_sources = len(domains) - 1
        self.domains = domains
        self.backbone_training_flag = backbone_training_flag
        self.loss_weight = loss_weight
        self.iteration = tf.Variable(0, name='iteration')
        self.models = self._init_models(build_backbone_lambda, build_top_lambda, freeze_backbone_flag)
        self.losses = self._init_losses()
        self.metrics = self._init_metrics()
        self.optimizers = self._init_optimizers(learning_rate)

    @tf.function
    def train(self, batch):
        self.iteration.assign_add(1)

        with tf.GradientTape() as tape:
            source_top_features = tuple(
                self.models['backbone'](batch[i][0], training=self.backbone_training_flag)
                for i in range(self.n_sources)
            )
            source_predictions = tuple(
                self.models['top'](source_top_features[i], training=True)
                for i in range(self.n_sources)
            )
            classification_loss = self.losses['classification'](
                tuple(zip(*batch[:self.n_sources]))[1], source_predictions
            )
            first_target_top_features = self.models['backbone'](batch[-1][0][0], training=self.backbone_training_flag)
            first_target_predictions = self.models['top'](first_target_top_features, training=True)
            second_target_top_features = self.models['backbone'](batch[-1][0][1], training=self.backbone_training_flag)
            second_target_predictions = self.models['top'](second_target_top_features, training=True)
            twice_loss = self.losses['twice'](first_target_predictions, second_target_predictions)
            loss = classification_loss + self.loss_weight * twice_loss

        trainable_variables = self.models['backbone'].trainable_variables + self.models['top'].trainable_variables
        self.optimizers['optimizer'].apply_gradients(zip(tape.gradient(loss, trainable_variables), trainable_variables))

        self.metrics['scce'].update_state(classification_loss)
        self.metrics['twice'].update_state(twice_loss)
        self.metrics['target_acc'].update_state(batch[-1][1], first_target_predictions)
        for i in range(self.n_sources):
            self.metrics[f'{self.domains[i]}_acc'].update_state(batch[i][1], source_predictions[i])

    @staticmethod
    def _init_models(build_backbone_lambda, build_top_lambda, freeze_backbone_flag):
        models = {
            'backbone': build_backbone_lambda(),
            'top': build_top_lambda()
        }
        if freeze_backbone_flag:
            for layer in models['backbone'].layers:
                layer.trainable = False
        return models

    @staticmethod
    def _init_losses():
        return {
            'classification': ClassificationLoss(),
            'twice': tf.keras.losses.KLDivergence()
        }

    def _init_metrics(self):
        metrics = {
            'scce': tf.keras.metrics.Mean(),
            'twice': tf.keras.metrics.Mean(),
            'target_acc': tf.keras.metrics.SparseCategoricalAccuracy()
        }
        for i in range(self.n_sources):
            metrics[f'{self.domains[i]}_acc'] = tf.keras.metrics.SparseCategoricalAccuracy()
        return metrics

    @staticmethod
    def _init_optimizers(learning_rate):
        return {
            'optimizer': tf.keras.optimizers.Adam(learning_rate)
        }
