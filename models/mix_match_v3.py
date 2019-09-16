import math
import h5py
import string
import tensorflow as tf
import tensorflow_probability as tfp
from collections import namedtuple

BASE_WEIGHTS_PATH = 'https://github.com/Callidior/keras-applications/releases/download/efficientnet/'
WEIGHTS_HASHES = (
    'c1421ad80a9fc67c2cc4000f666aa50789ce39eedb4e06d531b0c593890ccff3',
    '75de265d03ac52fa74f2f510455ba64f9c7c5fd96dc923cd4bfefa3d680c4b68',
    '433b60584fafba1ea3de07443b74cfd32ce004a012020b07ef69e22ba8669333',
    'c5d42eb6cfae8567b418ad3845cfd63aa48b87f1bd5df8658a49375a9f3135c7',
    '7942c1407ff1feb34113995864970cd4d9d91ea64877e8d9c38b6c1e0767c411',
    '9d197bc2bfe29165c10a2af8c2ebc67507f5d70456f09e584c71b822941b1952',
    '1d0923bb038f2f8060faaf0a0449db4b96549a881747b7c7678724ac79f427ed',
    '60b56ff3a8daccc8d96edfd40b204c113e51748da657afd58034d54d3cec2bac'
)
BlockArgs = namedtuple('BlockArgs', (
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters', 'expand_ratio', 'id_skip', 'strides', 'se_ratio'
))
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)
DEFAULT_BLOCKS_ARGS = (
    BlockArgs(
        kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
        expand_ratio=1, id_skip=True, strides=(1, 1), se_ratio=0.25
    ),
    BlockArgs(
        kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
        expand_ratio=6, id_skip=True, strides=(2, 2), se_ratio=0.25
    ),
    BlockArgs(
        kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
        expand_ratio=6, id_skip=True, strides=(2, 2), se_ratio=0.25
    ),
    BlockArgs(
        kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
        expand_ratio=6, id_skip=True, strides=(2, 2), se_ratio=0.25
    ),
    BlockArgs(
        kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
        expand_ratio=6, id_skip=True, strides=(1, 1), se_ratio=0.25
    ),
    BlockArgs(
        kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
        expand_ratio=6, id_skip=True, strides=(2, 2), se_ratio=0.25
    ),
    BlockArgs(
        kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
        expand_ratio=6, id_skip=True, strides=(1, 1), se_ratio=0.25
    )
)


def swish(x):
    return x * tf.nn.sigmoid(x)


class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = tf.shape(inputs)
        noise_shape = tuple(
            symbolic_shape[axis] if shape is None else shape for axis, shape in enumerate(self.noise_shape)
        )
        return noise_shape


def round_filters(filters, width_coefficient, depth_divisor):
    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    return int(math.ceil(depth_coefficient * repeats))


class Singleton:
    def __init__(self):
        self.layers = {}

    def get(self, layer_class, **kwargs):
        name = kwargs['name']
        if name not in self.layers:
            self.layers[name] = layer_class(**kwargs)
        return self.layers[name]


def mb_conv_block(singleton, domain, inputs, block_args, activation, drop_rate=None, prefix=''):
    has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)

    # Expansion phase
    filters = block_args.input_filters * block_args.expand_ratio
    if block_args.expand_ratio != 1:
        x = singleton.get(
            tf.keras.layers.Conv2D,
            filters=filters, kernel_size=1, padding='same', use_bias=False, name=f'{prefix}expand_conv'
        )(inputs)
        x = singleton.get(tf.keras.layers.BatchNormalization, name=f'{prefix}expand_bn_{domain}')(x)
        x = singleton.get(tf.keras.layers.Activation, activation=activation, name=f'{prefix}expand_activation')(x)
    else:
        x = inputs

    # Depthwise Convolution
    x = singleton.get(
        tf.keras.layers.DepthwiseConv2D, kernel_size=block_args.kernel_size,
        strides=block_args.strides, padding='same', use_bias=False, name=f'{prefix}dwconv'
    )(x)
    x = singleton.get(tf.keras.layers.BatchNormalization, name=f'{prefix}bn_{domain}')(x)
    x = singleton.get(tf.keras.layers.Activation, activation=activation, name=f'{prefix}activation')(x)

    # Squeeze and Excitation phase
    if has_se:
        num_reduced_filters = max(1, int(block_args.input_filters * block_args.se_ratio))
        se_tensor = singleton.get(tf.keras.layers.GlobalAveragePooling2D, name=f'{prefix}se_squeeze')(x)
        se_tensor = singleton.get(
            tf.keras.layers.Reshape, target_shape=(1, 1, filters), name=f'{prefix}se_reshape'
        )(se_tensor)
        se_tensor = singleton.get(
            tf.keras.layers.Conv2D, filters=num_reduced_filters,
            kernel_size=1, activation=activation, padding='same', use_bias=True, name=f'{prefix}se_reduce'
        )(se_tensor)
        se_tensor = singleton.get(
            tf.keras.layers.Conv2D, filters=filters,
            kernel_size=1, activation='sigmoid', padding='same', use_bias=True, name=f'{prefix}se_expand'
        )(se_tensor)
        x = singleton.get(tf.keras.layers.Multiply, name=f'{prefix}se_excite')([x, se_tensor])

    # Output phase
    x = singleton.get(
        tf.keras.layers.Conv2D, filters=block_args.output_filters,
        kernel_size=1, padding='same', use_bias=False, name=f'{prefix}project_conv'
    )(x)
    x = singleton.get(tf.keras.layers.BatchNormalization, name=f'{prefix}project_bn_{domain}')(x)
    if block_args.id_skip and all(
        s == 1 for s in block_args.strides
    ) and block_args.input_filters == block_args.output_filters:
        if drop_rate and (drop_rate > 0):
            x = singleton.get(FixedDropout, rate=drop_rate, noise_shape=(None, 1, 1, 1), name=f'{prefix}drop')(x)
        x = singleton.get(tf.keras.layers.Add, name=f'{prefix}add')([x, inputs])

    return x


def efficient_net(
    singleton, domain, width_coefficient, depth_coefficient, input_shape, drop_connect_rate=.2,
    depth_divisor=8, blocks_args=DEFAULT_BLOCKS_ARGS
):
    inputs = singleton.get(tf.keras.layers.Input, shape=input_shape, name='input')
    activation = swish

    # Build stem
    x = inputs
    x = singleton.get(
        tf.keras.layers.Conv2D, filters=round_filters(32, width_coefficient, depth_divisor),
        kernel_size=3, strides=(2, 2), padding='same', use_bias=False, name='stem_conv'
    )(x)
    x = singleton.get(tf.keras.layers.BatchNormalization, name=f'stem_bn_{domain}')(x)
    x = singleton.get(tf.keras.layers.Activation, activation=activation, name='stem_activation')(x)

    # Build blocks
    num_blocks_total = sum(block_args.num_repeat for block_args in blocks_args)
    block_num = 0
    for idx, block_args in enumerate(blocks_args):
        # Update block input and output filters based on depth multiplier.
        block_args = block_args._replace(
            input_filters=round_filters(block_args.input_filters, width_coefficient, depth_divisor),
            output_filters=round_filters(block_args.output_filters, width_coefficient, depth_divisor),
            num_repeat=round_repeats(block_args.num_repeat, depth_coefficient))

        # The first block needs to take care of stride and filter size increase.
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        x = mb_conv_block(
            singleton=singleton, inputs=x, domain=domain, block_args=block_args, activation=activation,
            drop_rate=drop_rate, prefix=f'block{idx + 1}a_')
        block_num += 1
        if block_args.num_repeat > 1:
            # pylint: disable=protected-access
            block_args = block_args._replace(input_filters=block_args.output_filters, strides=[1, 1])
            # pylint: enable=protected-access
            for bidx in range(block_args.num_repeat - 1):
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                block_prefix = f'block{idx + 1}{string.ascii_lowercase[bidx + 1]}_'
                x = mb_conv_block(
                    singleton=singleton, inputs=x, domain=domain, block_args=block_args,
                    activation=activation, drop_rate=drop_rate, prefix=block_prefix
                )
                block_num += 1

    # Build top
    x = singleton.get(
        tf.keras.layers.Conv2D, filters=round_filters(1280, width_coefficient, depth_divisor),
        kernel_size=1, padding='same', use_bias=False, name='top_conv'
    )(x)
    x = singleton.get(tf.keras.layers.BatchNormalization, name=f'top_bn_{domain}')(x)
    x = singleton.get(tf.keras.layers.Activation, activation=activation, name='top_activation')(x)

    # Create model.
    return tf.keras.models.Model(inputs, x)


class EfficientNet:
    def __init__(self, input_shape, model_index, width_coefficient, depth_coefficient):
        self.model_index = model_index
        self.singleton = Singleton()
        self.input_shape = input_shape
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient

    def __call__(self, domain):
        return efficient_net(
            singleton=self.singleton, domain=domain, width_coefficient=self.width_coefficient,
            depth_coefficient=self.depth_coefficient, input_shape=self.input_shape
        )

    def load_weights(self):
        model_name = f'efficientnet-b{self.model_index}'
        file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
        file_hash = WEIGHTS_HASHES[self.model_index]
        weight_path = tf.keras.utils.get_file(
            file_name, BASE_WEIGHTS_PATH + file_name, cache_subdir='models', file_hash=file_hash
        )
        file = h5py.File(weight_path, 'r')
        for name, layer in self.singleton.layers.items():
            if name == 'input' or len(layer.weights) == 0:
                continue
            if 'bn_' in name:
                name = name[:-2]

            weights = []
            index_name = list(file['model_weights'][name].keys())
            assert len(index_name) == 1
            for variable in layer.weights:
                weights.append(file['model_weights'][name][index_name[0]][variable.name.split('/')[-1]])
            layer.set_weights(weights)


def efficient_net_b0(input_shape):
    return EfficientNet(
        input_shape=input_shape, model_index=0, width_coefficient=1., depth_coefficient=1.
    )


def efficient_net_b4(input_shape):
    return EfficientNet(
        input_shape=input_shape, model_index=4, width_coefficient=1.4, depth_coefficient=1.8
    )


def efficient_net_b5(input_shape):
    return EfficientNet(
        input_shape=input_shape, model_index=5, width_coefficient=1.6, depth_coefficient=2.2
    )


class MSELoss:
    def __init__(self):
        self.scorer = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

    def __call__(self, labels, predictions):
        loss = .0
        for label, prediction in zip(labels, predictions):
            loss += self.scorer(label, prediction)
        loss /= len(labels)
        return loss


class ClassificationLoss:
    def __init__(self):
        self.scorer = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)

    def __call__(self, labels, predictions):
        loss = .0
        for label, prediction in zip(labels, predictions):
            loss += self.scorer(label, prediction)
        loss /= len(labels)
        return loss


class MixMatchV3TrainStep:
    def __init__(
        self, build_backbone_lambda, build_top_lambda, learning_rate, source_domains,
        loss_weight, temperature, alpha, global_batch_size, local_batch_size
    ):
        self.source_domains = source_domains
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.alpha = alpha
        self.global_batch_size = global_batch_size
        self.local_batch_size = local_batch_size
        self.iteration = tf.Variable(
            0, name='iteration', dtype=tf.int64, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA
        )
        self.models = self._init_models(build_backbone_lambda, build_top_lambda, source_domains)
        self.losses = self._init_losses()
        self.metrics = self._init_metrics()
        self.optimizers = self._init_optimizers(learning_rate)

    def train(self, batch):
        self.iteration.assign_add(1)

        mixed_source_images, mixed_source_labels, mixed_target_images, mixed_target_labels = self._mix_match(
            source_images=tuple(batch[i]['image'] for i in range(len(self.source_domains))),
            source_labels=tuple(batch[i]['label'] for i in range(len(self.source_domains))),
            target_images=tuple(batch[-1]['image'][i] for i in range(len(batch[-1]['image'])))
        )

        with tf.GradientTape() as tape:
            source_predictions = self._run_models(mixed_source_images, self.source_domains, training=True)
            target_predictions = self._run_models(mixed_target_images, (0,) * len(mixed_target_images), training=True)
            source_loss = self.losses['source'](mixed_source_labels, source_predictions)
            target_loss = self.losses['target'](mixed_target_labels, target_predictions)
            loss = (source_loss + target_loss * self.loss_weight) / self.global_batch_size

        trainable_variables = self.models['backbone_0'].trainable_variables + self.models['top'].trainable_variables
        self.optimizers['optimizer'].apply_gradients(zip(
            tape.gradient(loss, trainable_variables), trainable_variables
        ))
        target_predictions = self._run_models((batch[-1]['image'][0],), (0,), training=False)[0]
        self.metrics['source_loss'].update_state(source_loss / self.local_batch_size)
        self.metrics['target_loss'].update_state(target_loss / self.local_batch_size)
        self.metrics['target_acc'].update_state(batch[-1]['label'], target_predictions)

    def _mix_match(self, source_images, source_labels, target_images):
        target_predictions = self._run_models(target_images, (0,) * len(target_images), training=True)
        target_predictions = tf.add_n(target_predictions)
        target_predictions /= len(target_images)
        target_predictions = self._sharpen(target_predictions, self.temperature)
        target_predictions = tf.concat(tuple(target_predictions for _ in range(len(target_images))), axis=0)

        n_batches = len(source_images) + len(target_images)
        combined_images = tf.concat(source_images + target_images, axis=0)
        combined_source_labels = tf.concat(source_labels, axis=0)
        categorical_source_labels = tf.one_hot(combined_source_labels, tf.shape(target_predictions)[1])
        combined_labels = tf.concat((categorical_source_labels, target_predictions), axis=0)
        indexes = tf.range(self.local_batch_size * n_batches)
        shuffled_indexes = tf.random.shuffle(indexes)
        shuffled_images = tf.gather(combined_images, shuffled_indexes)
        shuffled_labels = tf.gather(combined_labels, shuffled_indexes)

        mixed_images, mixed_labels = self._mix_up(
            first_images=combined_images,
            first_labels=combined_labels,
            second_images=shuffled_images,
            second_labels=shuffled_labels,
            alpha=self.alpha,
            batch_size=self.local_batch_size * n_batches
        )
        splitted_images = tf.split(mixed_images, n_batches)
        splitted_labels = tf.split(mixed_labels, n_batches)
        return (
            splitted_images[:len(source_images)],
            splitted_labels[:len(source_images)],
            splitted_images[len(source_images):],
            splitted_labels[len(source_images):]
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

    def _run_models(self, images, domains, training):
        features = tuple(
            self.models[f'backbone_{domain}'](image, training=training) for image, domain in zip(images, domains)
        )
        return tuple(self.models['top'](feature, training=training) for feature in features)

    @staticmethod
    def _init_models(build_backbone_lambda, build_top_lambda, source_domains):
        backbone = build_backbone_lambda()
        models = {'top': build_top_lambda()}
        for i in range(max(source_domains) + 1):
            models[f'backbone_{i}'] = backbone(i)
        backbone.load_weights()
        return models

    @staticmethod
    def _init_losses():
        return {
            'source': ClassificationLoss(),
            'target': MSELoss()
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

