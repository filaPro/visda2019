import tensorflow as tf
import efficientnet.tfkeras as efficient_net


def get_backbone(name):
    if name == 'vgg19':
        backbone = tf.keras.applications.VGG19
    elif name == 'mobile_net_v2':
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
