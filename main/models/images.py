#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input


def load_image(img_path):
    # https://www.tensorflow.org/tutorials/text/image_captioning
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    # input shape of MobileNet is (224, 224, 3)
    img = tf.image.resize(img, (224, 224))
    img = preprocess_input(img)
    return img


def create_mobilenet_encoder():
    mobilenet = MobileNet(include_top=False, weights='imagenet')
    mobilenet_input = mobilenet.input
    # shape of the last layer in MobileNet is (1, 1, 1024)
    x = mobilenet.layers[-4].output
    # average values in each channel to make output 1D
    out = tf.keras.layers.GlobalAveragePooling2D()(x)

    model = tf.keras.Model(
        inputs=mobilenet_input,
        outputs=out
    )
    return model


class ImageEncoder(object):
    """
    Encode image data for RNN.
    This will generate vectors with 128 length.

    Example:
        >>> imgs = ['data/img1.jpg', 'data/img2.jpg']
        >>> encoder = ImageEncoder()
        >>> encoder.train(dataset)
        >>> features = encoder.get_imgs_features(imgs)
        >>> print(features.shape)
        (2, 128)
    """

    def __init__(self):
        self.model = create_mobilenet_encoder()

        # no need to train
        for layer in self.model.layers:
            layer.trainable = False

        self._kwgs = {}

    def get_image_features(self, image_path):
        """Calculate features from a single image.
        Args:
            image_path: str
                a path to an image data
        Return:
            numpy.ndarray with shape (128,)
        """
        img = np.expand_dims(load_image(image_path), axis=0)
        return self.model(img, training=False)[0]

    def get_imgs_features(self, image_paths):
        """Calculate features from images.
        Args:
            image_paths: a list or tuple of str
                path to image data
        Return:
            numpy.ndarray with shape (len(image_paths), 128)
        """
        images = np.zeros((len(image_paths), 224, 224, 3))
        for i, img_path in enumerate(image_paths):
            images[i, :, :, :] = load_image(img_path)
        return self.model(images, training=False)
