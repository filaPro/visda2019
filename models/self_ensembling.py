import tensorflow as tf

from .common import ClassificationLoss
from preprocessor import Preprocessor


class SelfEnsemblingPreprocessor:
    def __init__(self, first_config, second_config):
        self.first_preprocessor = Preprocessor(first_config)
        self.second_preprocessor = Preprocessor(second_config)

    @tf.function
    def __call__(self, image):
        return self.first_preprocessor(image), self.second_preprocessor(image)


class ExponentialMovingAveraging:
    def __init__(self, decay):
        self.decay = decay

    def __call__(self, old, new):
        for old_variable, new_variable in zip(old, new):
            old_variable.assign(self.decay * old_variable + (1 - self.decay) * new_variable)


class SelfEnsemblingTrainStep:
    def __init__(
        self, build_backbone_lambda, build_top_lambda, domains, backbone_training_flag,
        backbone_learning_rate, top_learning_rate, loss_weight, decay, batch_size
    ):
        self.n_sources = len(domains) - 1
        self.domains = domains
        self.backbone_training_flag = backbone_training_flag
        self.loss_weight = loss_weight
        self.batch_size = batch_size
        self.iteration = tf.Variable(
            0, name='iteration', dtype=tf.int64, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA
        )
        self.ema = ExponentialMovingAveraging(decay)
        self.models = self._init_models(build_backbone_lambda, build_top_lambda)
        self.losses = self._init_losses()
        self.metrics = self._init_metrics()
        self.optimizers = self._init_optimizers(backbone_learning_rate, top_learning_rate)

    def train(self, batch):
        self.iteration.assign_add(1)

        teacher_target_top_features = self.models['teacher_backbone'](
            batch[-1]['image'][1], training=self.backbone_training_flag
        )
        teacher_target_predictions = self.models['teacher_top'](teacher_target_top_features, training=True)
        with tf.GradientTape() as tape, tf.GradientTape() as tape_backbone:
            source_top_features = tuple(
                self.models['student_backbone'](batch[i]['image'], training=self.backbone_training_flag)
                for i in range(self.n_sources)
            )
            source_predictions = tuple(
                self.models['student_top'](source_top_features[i], training=True)
                for i in range(self.n_sources)
            )
            classification_loss = self.losses['classification'](
                tuple(batch[i]['label'] for i in range(self.n_sources)), source_predictions
            )
            classification_loss /= self.batch_size
            student_target_top_features = self.models['student_backbone'](
                batch[-1]['image'][0], training=self.backbone_training_flag
            )
            student_target_predictions = self.models['student_top'](student_target_top_features, training=True)
            adaptation_loss = self.losses['adaptation'](student_target_predictions, teacher_target_predictions)
            adaptation_loss /= self.batch_size
            loss = classification_loss + self.loss_weight * adaptation_loss

        student_trainable_variables = self.models['student_backbone'].trainable_variables + \
            self.models['student_top'].trainable_variables
        teacher_trainable_variables = self.models['teacher_backbone'].trainable_variables + \
            self.models['teacher_top'].trainable_variables
        self.optimizers['top_optimizer'].apply_gradients(zip(
            tape.gradient(loss, self.models['student_top'].trainable_variables),
            self.models['student_top'].trainable_variables
        ))
        self.optimizers['backbone_optimizer'].apply_gradients(zip(
            tape_backbone.gradient(loss, self.models['student_backbone'].trainable_variables),
            self.models['student_backbone'].trainable_variables)
        )
        self.ema(teacher_trainable_variables, student_trainable_variables)

        self.metrics['scce'].update_state(classification_loss)
        self.metrics['adaptation'].update_state(adaptation_loss)
        self.metrics['student_target_acc'].update_state(batch[-1]['label'], student_target_predictions)
        self.metrics['teacher_target_acc'].update_state(batch[-1]['label'], teacher_target_predictions)
        for i in range(self.n_sources):
            self.metrics[f'{self.domains[i]}_acc'].update_state(batch[i]['label'], source_predictions[i])

    @staticmethod
    def _init_models(build_backbone_lambda, build_top_lambda):
        models = {
            'student_backbone': build_backbone_lambda(),
            'student_top': build_top_lambda(),
            'teacher_backbone': build_backbone_lambda(),
            'teacher_top': build_top_lambda()
        }
        return models

    @staticmethod
    def _init_losses():
        return {
            'classification': ClassificationLoss(),
            'adaptation': tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
            # TODO: Why KLD is not better?
            # 'adaptation': tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM)
        }

    def _init_metrics(self):
        metrics = {
            'scce': tf.keras.metrics.Mean(),
            'adaptation': tf.keras.metrics.Mean(),
            'student_target_acc': tf.keras.metrics.SparseCategoricalAccuracy(),
            'teacher_target_acc': tf.keras.metrics.SparseCategoricalAccuracy()
        }
        for i in range(self.n_sources):
            metrics[f'{self.domains[i]}_acc'] = tf.keras.metrics.SparseCategoricalAccuracy()
        return metrics

    @staticmethod
    def _init_optimizers(backbone_learning_rate, top_learning_rate):
        return {
            'backbone_optimizer': tf.keras.optimizers.Adam(backbone_learning_rate),
            'top_optimizer': tf.keras.optimizers.Adam(top_learning_rate)
        }


class SelfEnsemblingTestStep:
    def __init__(self, build_backbone_lambda, build_top_lambda):
        self.iteration = tf.Variable(0, name='iteration')
        self.models = self._init_models(build_backbone_lambda, build_top_lambda)
        self.metrics = self._init_metrics()

    @tf.function
    def test(self, batch):
        self.iteration.assign_add(1)
        top_features = self.models['teacher_backbone'](batch['image'], training=False)
        predictions = self.models['teacher_top'](top_features, training=False)
        self.metrics['acc'].update_state(batch['label'], predictions)

    @staticmethod
    def _init_models(build_backbone_lambda, build_model_lambda):
        return {
            'teacher_backbone': build_backbone_lambda(),
            'teacher_top': build_model_lambda()
        }

    @staticmethod
    def _init_metrics():
        return {
            'acc': tf.keras.metrics.SparseCategoricalAccuracy()
        }
