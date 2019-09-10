import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


class Preprocessor:
    def __init__(self, config):
        """
        :param config: [{'method': str, 'key_0': value_0, ...}]
            method: random_brightness
                delta: .2, range(.0, 1.)
            method: random_contrast
                delta: .5, range(.0, 1.)
            method: random_rotate
                angle: 10., range: (.0, 180.)
            method: random_translate
                shift: .1, range: (.0, 1.)
        """
        self.config = config

    def __call__(self, image):
        for item in self.config:
            if item['method'] == 'keras':
                if item['mode'] == 'tf':
                    image /= 127.5
                    image -= 1.
                elif item['mode'] == 'caffe':
                    image = image[..., ::-1]
                    mean = tf.constant((103.939, 116.779, 123.68))
                    image -= mean
                elif item['mode'] == 'torch':
                    image /= 255.
                    mean = tf.constant((.485, .456, .406))
                    std = tf.constant((.229, .224, .225))
                    image -= mean
                    image /= std
                else:
                    raise ValueError(f'Invalid mode: {item["mode"]} for method: keras')
            elif item['method'] == 'resize':
                image = tf.image.resize(image, size=(item['height'], item['width']))
            elif item['method'] == 'resize_min':
                shape = tf.cast(tf.shape(image), tf.float32)
                ratio = item['size'] / tf.minimum(shape[0], shape[1])
                image = tf.image.resize(image, size=(ratio * shape[0], ratio * shape[1]))
            elif item['method'] == 'resize_max':
                shape = tf.cast(tf.shape(image), tf.float32)
                ratio = item['size'] / tf.maximum(shape[0], shape[1])
                image = tf.image.resize(image, size=(ratio * shape[0], ratio * shape[1]))
            elif item['method'] == 'central_crop':
                shape = tf.shape(image)
                image = tf.image.crop_to_bounding_box(
                    image,
                    offset_height=(shape[0] - item['height']) // 2,
                    offset_width=(shape[1] - item['width']) // 2,
                    target_height=item['height'],
                    target_width=item['width']
                )
            elif item['method'] == 'random_crop':
                image = tf.image.random_crop(image, size=(item['height'], item['width'], item['n_channels']))
            elif item['method'] == 'random_size_crop':
                height = tf.random.uniform([], item['min_height'], item['max_height'], dtype=tf.int64)
                width = tf.random.uniform([], item['min_width'], item['max_width'], dtype=tf.int64)
                image = tf.image.random_crop(image, size=(height, width, item['n_channels']))
            elif item['method'] == 'flip_left_right':
                image = tf.image.flip_left_right(image)
            elif item['method'] == 'random_flip_left_right':
                image = tf.image.random_flip_left_right(image)
            elif item['method'] == 'random_brightness':
                delta = item['delta'] * 255.
                image = tf.image.random_brightness(image, delta)
                image = tf.clip_by_value(image, 0., 255.)
            elif item['method'] == 'random_contrast':
                image = tf.image.random_contrast(image, 1 - item['delta'], 1 + item['delta'])
                image = tf.clip_by_value(image, 0., 255.)
            elif item['method'] == 'random_rotate':
                radians = item['angle'] / 180. * np.pi
                angle = tf.random.uniform([], -radians, radians)
                image = tfa.image.rotate(image, angle)
            elif item['method'] == 'random_mean_filter':
                if tf.random.uniform([]) > .5:
                    size = item['size']
                    image = tfa.image.mean_filter2d(image, filter_shape=(size, size))
            elif item['method'] == 'reshape':
                image = tf.reshape(image, (item['height'], item['width'], item['n_channels']))
            else:
                raise ValueError(f'Invalid parameter name: {item["method"]}')
        return image
