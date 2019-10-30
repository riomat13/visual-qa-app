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
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, 'relu')(x)

    # output has the same hidden unit size as decoder input
    out = tf.keras.layers.Dense(128, 'tanh')(x)

    model = tf.keras.Model(
        inputs=mobilenet_input,
        outputs=out
    )
    return model


class ImageEncoder(object):
    def __init__(self):
        self.model = create_mobilenet_encoder()

    def get_image_features(self, image_path):
        img = np.expand_dims(load_image(image_path), axis=0)
        return self.model(img)[0]

    def get_imgs_features(self, image_paths):
        images = np.zeros((len(image_paths), 224, 224, 3))
        for i, img_path in enumerate(image_paths):
            images[i, :, :, :] = load_image(img_path)
        return self.model(images)
