import tensorflow as tf
import efficientnet.tfkeras as efficient_net

from preprocessor import Preprocessor


def get_backbone(name):
    if name == 'vgg19':
        return tf.keras.applications.VGG19, 512, 'caffe'
    elif name == 'mobile_net_v2':
        return tf.keras.applications.MobileNetV2, 1280, 'tf'
    elif name == 'efficient_net_b0':
        return efficient_net.EfficientNetB0, 1280, 'torch'
    elif name == 'efficient_net_b1':
        return efficient_net.EfficientNetB1, 1280, 'torch'
    elif name == 'efficient_net_b2':
        return efficient_net.EfficientNetB2, 1408, 'torch'
    elif name == 'efficient_net_b3':
        return efficient_net.EfficientNetB3, 1536, 'torch'
    elif name == 'efficient_net_b4':
        return efficient_net.EfficientNetB4, 1792, 'torch'
    elif name == 'efficient_net_b5':
        return efficient_net.EfficientNetB5, 2048, 'torch'
    elif name == 'efficient_net_b6':
        return efficient_net.EfficientNetB6, 2304, 'torch'
    elif name == 'efficient_net_b7':
        return efficient_net.EfficientNetB7, 2560, 'torch'
    else:
        raise ValueError(f'Invalid name: {name}')


def build_backbone(name, size):
    return get_backbone(name)[0](
        input_shape=(size, size, 3),
        include_top=False,
        weights='imagenet'
    )


def get_n_backbone_channels(name):
    return get_backbone(name)[1]


def get_backbone_normalization(name):
    return get_backbone(name)[2]


def build_top(n_channels, n_classes):
    return tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(input_shape=(7, 7, n_channels)),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])


class ClassificationLoss:
    def __init__(self):
        self.scorer = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)

    def __call__(self, labels, predictions):
        loss = .0
        for label, prediction in zip(labels, predictions):
            loss += self.scorer(label, prediction)
        loss /= len(labels)
        return loss


def run_balanced(models, tensors):
    n_tensors = len(tensors)
    splitted_tensors = tuple(tf.split(tensors[i], n_tensors) for i in range(n_tensors))
    results = []
    for i in range(n_tensors):
        combined_tensors = tf.concat(tuple(splitted_tensors[j][i] for j in range(n_tensors)), axis=0)
        for model in models:
            combined_tensors = model(combined_tensors, training=True)
        results.append(combined_tensors)
    splitted_results = tuple(tf.split(results[i], n_tensors) for i in range(n_tensors))
    combined_results = []
    for i in range(n_tensors):
        combined_results.append(tf.concat(tuple(splitted_results[j][i] for j in range(n_tensors)), axis=0))
    return tuple(combined_results)


class SelfEnsemblingPreprocessor:
    def __init__(self, configs):
        self.preprocessors = tuple(Preprocessor(config) for config in configs)

    def __call__(self, image):
        return tuple(preprocessor(image) for preprocessor in self.preprocessors)
