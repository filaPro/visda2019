import tensorflow as tf
import efficientnet.tfkeras as efficient_net


def get_backbone(name):
    if name == 'vgg19':
        backbone = tf.keras.applications.VGG19
    elif name == 'mobilenet_v2':
        backbone = tf.keras.applications.MobileNetV2
    elif name == 'efficient_net_b0':
        backbone = efficient_net.EfficientNetB0
    elif name == 'efficient_net_b1':
        backbone = efficient_net.EfficientNetB1
    elif name == 'efficient_net_b2':
        backbone = efficient_net.EfficientNetB2
    elif name == 'efficient_net_b3':
        backbone = efficient_net.EfficientNetB3
    elif name == 'efficient_net_b4':
        backbone = efficient_net.EfficientNetB4
    elif name == 'efficient_net_b5':
        backbone = efficient_net.EfficientNetB5
    elif name == 'efficient_net_b6':
        backbone = efficient_net.EfficientNetB6
    elif name == 'efficient_net_b7':
        backbone = efficient_net.EfficientNetB7
    else:
        raise ValueError(f'Invalid name: {name}')
    return backbone


def build_backbone(name, size):
    return get_backbone(name)(
        input_shape=(size, size, 3),
        include_top=False,
        weights='imagenet'
    )


class ClassificationLoss:
    def __init__(self):
        self.scorer = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)

    def __call__(self, labels, predictions):
        loss = .0
        for label, prediction in zip(labels, predictions):
            loss += self.scorer(label, prediction)
        loss /= len(labels)
        return loss
