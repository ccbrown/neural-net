#!/usr/bin/env python
import cv2
import numpy as np
import struct
try:
    import keras
except ImportError:
    from tensorflow import keras


class Residual(keras.layers.Layer):
    def __init__(self, layers, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.layers = layers

    def call(self, x):
        return keras.layers.Add()([x, keras.Sequential(self.layers)(x)])

    def compute_output_shape(self, input_shape):
        return input_shape


class DarknetConvolutional(keras.Sequential):
    def __init__(self, filters, size, strides, activation=None, batch_normalize=False, **kwargs):
        super(DarknetConvolutional, self).__init__([
            keras.layers.Conv2D(filters, size, strides=strides, padding='same', use_bias=not batch_normalize, **kwargs),
        ])
        self.filters = filters
        self.batch_normalize = batch_normalize
        self.size = size
        if batch_normalize:
            self.add(keras.layers.BatchNormalization(epsilon=1e-5))
        if activation == 'leaky':
            self.add(keras.layers.LeakyReLU(alpha=0.1))

    def load_darknet_weights(self, f):
        weights = {}
        weights['beta:0' if self.batch_normalize else 'bias:0'] = np.fromfile(f, dtype=np.float32, count=self.filters)
        if self.batch_normalize:
            weights['gamma:0'] = np.fromfile(f, dtype=np.float32, count=self.filters)
            weights['moving_mean:0'] = np.fromfile(f, dtype=np.float32, count=self.filters)
            weights['moving_variance:0'] = np.fromfile(f, dtype=np.float32, count=self.filters)
        kernel_weights_shape = (self.filters, self.input_shape[3], self.size, self.size)
        weights['kernel:0'] = np.fromfile(f, dtype=np.float32, count=np.prod(kernel_weights_shape)).reshape(kernel_weights_shape).transpose([2, 3, 1, 0])
        self.set_weights([weights[w.name.split('/')[-1]] for w in self.weights])


class Darknet53Residual(Residual):
    def __init__(self, filters1, filters2, **kwargs):
        super(Darknet53Residual, self).__init__([
            DarknetConvolutional(filters1, 1, 1, batch_normalize=True, activation='leaky', **kwargs),
            DarknetConvolutional(filters2, 3, 1, batch_normalize=True, activation='leaky'),
        ])


# https://github.com/pjreddie/darknet/blob/master/cfg/darknet53.cfg
class Darknet53(keras.Sequential):
    def __init__(self, **kwargs):
        super(Darknet53, self).__init__([
            DarknetConvolutional(32, 3, 1, batch_normalize=True, activation='leaky', **kwargs),
            DarknetConvolutional(64, 3, 2, batch_normalize=True, activation='leaky'),
            Darknet53Residual(32, 64),
            DarknetConvolutional(128, 3, 2, batch_normalize=True, activation='leaky'),
        ] + [Darknet53Residual(64, 128) for i in range(2)] + [
            DarknetConvolutional(256, 3, 2, batch_normalize=True, activation='leaky'),
        ] + [Darknet53Residual(128, 256) for i in range(8)] + [
            DarknetConvolutional(512, 3, 2, batch_normalize=True, activation='leaky'),
        ] + [Darknet53Residual(256, 512) for i in range(8)] + [
            DarknetConvolutional(1024, 3, 2, batch_normalize=True, activation='leaky'),
        ] + [Darknet53Residual(512, 1024) for i in range(4)] + [
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Reshape((1, 1, 1024)),
            DarknetConvolutional(1000, 1, 1),
            keras.layers.Reshape((1000,)),
            keras.layers.Softmax(),
        ])


def load_weights(l, f):
    if hasattr(l, 'load_darknet_weights'):
        l.load_darknet_weights(f)
    elif hasattr(l, 'layers'):
        for sl in l.layers:
            load_weights(sl, f)


model = Darknet53(input_shape=(256, 256, 3))
model.summary()

with open('darknet53.weights', 'rb') as f:
    major, minor, revision = struct.unpack('<iii', f.read(12))
    if major != 0 or minor != 2:
        raise RuntimeError('unsupported weights file version. got {}.{}.{}, expected 0.2'.format(major, minor, revision))
    f.read(8)
    load_weights(model, f)

img = cv2.cvtColor(cv2.imread('dog.jpg'), cv2.COLOR_BGR2RGB)
prediction = model.predict([[np.array(img) / 255.0]])[0]
with open('labels.txt', 'r') as f:
    labels = f.read().splitlines()
for i in np.flip(np.argsort(prediction))[:5]:
    print('{}: {:.2f}%'.format(labels[i], prediction[i] * 100.0))
