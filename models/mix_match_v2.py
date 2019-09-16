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


class DomainSpecificBatchNormalization(tf.keras.layers.Layer):
    def __init__(self, n_domains, name):
        for i in range(n_domains):
            layer = tf.keras.layers.BatchNormalization()
            self.__setattr__(f'layer_{i}', layer)
        super().__init__(name=name)

    def call(self, inputs, training=None):
        def f0():
            return self.layer_0(inputs[0])

        def f1():
            return self.layer_1(inputs[0])

        def f2():
            return self.layer_2(inputs[0])

        def f3():
            return self.layer_3(inputs[0])

        return tf.switch_case(inputs[1], branch_fns={0: f0, 1: f1, 2: f2, 3: f3})
        return self.__getattribute__(f'layer_{0}')(inputs[0])  # inputs[1] # TODO: <-


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


def mb_conv_block(inputs, n_domains, block_args, activation, drop_rate=None, prefix=''):
    has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)

    # Expansion phase
    filters = block_args.input_filters * block_args.expand_ratio
    if block_args.expand_ratio != 1:
        x = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False, name=prefix + 'expand_conv')(inputs[0])
        # x = tf.keras.layers.BatchNormalization(name=prefix + 'expand_bn')(x)
        x = DomainSpecificBatchNormalization(n_domains=n_domains, name=prefix + 'expand_bn')((x, inputs[1]))
        x = tf.keras.layers.Activation(activation, name=prefix + 'expand_activation')(x)
    else:
        x = inputs[0]

    # Depthwise Convolution
    x = tf.keras.layers.DepthwiseConv2D(
        block_args.kernel_size, strides=block_args.strides, padding='same', use_bias=False, name=prefix + 'dwconv'
    )(x)
    # x = tf.keras.layers.BatchNormalization(name=prefix + 'bn')(x)
    x = DomainSpecificBatchNormalization(n_domains=n_domains, name=prefix + 'bnn')((x, inputs[1]))
    x = tf.keras.layers.Activation(activation, name=prefix + 'activation')(x)

    # Squeeze and Excitation phase
    if has_se:
        num_reduced_filters = max(1, int(block_args.input_filters * block_args.se_ratio))
        se_tensor = tf.keras.layers.GlobalAveragePooling2D(name=prefix + 'se_squeeze')(x)
        se_tensor = tf.keras.layers.Reshape((1, 1, filters), name=prefix + 'se_reshape')(se_tensor)
        se_tensor = tf.keras.layers.Conv2D(
            num_reduced_filters, 1, activation=activation, padding='same', use_bias=True, name=prefix + 'se_reduce'
        )(se_tensor)
        se_tensor = tf.keras.layers.Conv2D(
            filters, 1, activation='sigmoid', padding='same', use_bias=True, name=prefix + 'se_expand'
        )(se_tensor)
        x = tf.keras.layers.multiply([x, se_tensor], name=prefix + 'se_excite')

    # Output phase
    x = tf.keras.layers.Conv2D(
        block_args.output_filters, 1, padding='same', use_bias=False, name=prefix + 'project_conv'
    )(x)
    # x = tf.keras.layers.BatchNormalization(name=prefix + 'project_bn')(x)
    x = DomainSpecificBatchNormalization(n_domains=n_domains, name=prefix + 'project_bn')((x, inputs[1]))
    if block_args.id_skip and all(
        s == 1 for s in block_args.strides
    ) and block_args.input_filters == block_args.output_filters:
        if drop_rate and (drop_rate > 0):
            x = FixedDropout(drop_rate, noise_shape=(None, 1, 1, 1), name=prefix + 'drop')(x)
        x = tf.keras.layers.add([x, inputs[0]], name=prefix + 'add')

    return x


def efficient_net(
    n_domains, width_coefficient, depth_coefficient, drop_connect_rate,
    depth_divisor=8, blocks_args=DEFAULT_BLOCKS_ARGS, model_index=0, weights='imagenet', input_shape=None
):
    image_input = tf.keras.layers.Input(shape=input_shape)
    domain_input = tf.keras.layers.Input(batch_shape=(1,), dtype=tf.int32)  # TODO: ?
    # domain = domain_input
    # domain = tf.strings.as_string(tf.keras.layers.Reshape(())(domain_input))  # tf.strings.as_string(tf.keras.layers.Reshape(()))
    # domain = domain_input[0]
    activation = swish

    # Build stem
    x = image_input
    x = tf.keras.layers.Conv2D(
        round_filters(32, width_coefficient, depth_divisor), 3, strides=(2, 2), padding='same', use_bias=False,
        name='stem_conv'
    )(x)
    # x = tf.keras.layers.BatchNormalization(name='stem_bn')(x)
    x = DomainSpecificBatchNormalization(n_domains=n_domains, name='stem_bn')((x, domain_input))
    x = tf.keras.layers.Activation(activation, name='stem_activation')(x)

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
        x = mb_conv_block(inputs=(x, domain_input), n_domains=n_domains, block_args=block_args, activation=activation, drop_rate=drop_rate, prefix='block{}a_'.format(idx + 1))
        block_num += 1
        if block_args.num_repeat > 1:
            # pylint: disable=protected-access
            block_args = block_args._replace(input_filters=block_args.output_filters, strides=[1, 1])
            # pylint: enable=protected-access
            for bidx in range(block_args.num_repeat - 1):
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                block_prefix = 'block{}{}_'.format(idx + 1, string.ascii_lowercase[bidx + 1])
                x = mb_conv_block(
                    inputs=(x, domain_input), n_domains=n_domains, block_args=block_args,
                    activation=activation, drop_rate=drop_rate, prefix=block_prefix
                )
                block_num += 1

    # Build top
    x = tf.keras.layers.Conv2D(
        round_filters(1280, width_coefficient, depth_divisor), 1, padding='same', use_bias=False, name='top_conv'
    )(x)
    # x = tf.keras.layers.BatchNormalization(name='top_bn')(x)
    x = DomainSpecificBatchNormalization(n_domains=n_domains, name='top_bn')((x, domain_input))
    x = tf.keras.layers.Activation(activation, name='top_activation')(x)

    # Create model.
    model_name = f'efficientnet-b{model_index}'
    model = tf.keras.models.Model((image_input, domain_input), x, name=model_name)

    # Load weights.
    if weights == 'imagenet':
        file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
        file_hash = WEIGHTS_HASHES[model_index]
        weights_path = tf.keras.utils.get_file(
            file_name, BASE_WEIGHTS_PATH + file_name, cache_subdir='models', file_hash=file_hash
        )
        # model.load_weights(weights_path) # TODO:
        # file = h5py.File(weights_path, 'r')
        # print(file.keys)

    elif weights is not None:
        model.load_weights(weights)

    return model


def efficient_net_b0(n_domains, input_shape, weights='imagenet'):
    return efficient_net(n_domains, 1.0, 1.0, 0.2, model_index=0, weights=weights, input_shape=input_shape)


def efficient_net_b5(n_domains, input_shape, weights='imagenet'):
    return efficient_net(n_domains, 1.6, 2.2, 0.4, model_index=5, weights=weights, input_shape=input_shape)


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


class MixMatchV2TrainStep:
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
        self.models = self._init_models(build_backbone_lambda, build_top_lambda)
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

        trainable_variables = self.models['backbone'].trainable_variables + self.models['top'].trainable_variables
        self.optimizers['optimizer'].apply_gradients(zip(
            tape.gradient(loss, trainable_variables), trainable_variables
        ))
        target_predictions = self._run_models((batch[-1]['image'][0],), (0,), training=False)
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
            self.models['backbone']((image, domain), training=training) for image, domain in zip(images, domains)
        )
        return tuple(self.models['top'](feature, training=training) for feature in features)

    @staticmethod
    def _init_models(build_backbone_lambda, build_top_lambda):
        return {
            'backbone': build_backbone_lambda(),
            'top': build_top_lambda()
        }

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

