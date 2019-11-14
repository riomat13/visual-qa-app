#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.applications import MobileNet


def _get_mobilenet_encoder():
    model = None

    def model_generator():
        """MobileNet base encoder model

        Given image shape has to be (None, 224, 224, 3)

        Usage:
            >>> imgs.shape
            (32, 224, 224, 3)
            >>> model = get_mobilenet_encoder()
            >>> encoded = model(imgs)
            >>> encoded.shape
            TensorShape([32, 7, 7, 1024])
        """
        nonlocal model

        if model is None:
            mobilenet = MobileNet(include_top=False, weights='imagenet')
            mobilenet_input = mobilenet.input
            # shape of the last layer in MobileNet: (7, 7, 1024)
            out = mobilenet.layers[-1].output

            model = tf.keras.Model(
                inputs=mobilenet_input,
                outputs=out
            )
        return model
    return model_generator


# generate mobilenet base model function
get_mobilenet_encoder = _get_mobilenet_encoder()


class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        # FC for input feature
        self.dense_features = tf.keras.layers.Dense(units)
        # FC for hidden states from encoder
        self.dense_states = tf.keras.layers.Dense(units)

        self.dense_out = tf.keras.layers.Dense(1)

    def call(self, features, states):
        """
        Args:
            features: encoded features from RNN
                shape = (batch_size, sequence_length, embedding_dim)
            states:   hidden states
                shape = (batch_size, hidden_size)
        Returns:
            context: context tensor
                shape = (batch_size, hidden_size)
            attention_weights: weights used for attention
                shape = (batch_size, sequence_length, 1)
        """
        features = self.dense_features(features)

        # weights to update feature importance
        states = self.dense_states(states)
        states = tf.keras.layers.RepeatVector(features.shape[1])(states)

        # calculate attention weights
        # (batch_size, sequence_length, units)
        score = tf.nn.tanh(features + states)
        score = self.dense_out(score)
        attention_weights = tf.nn.softmax(score, axis=1)

        # update the feature weighted by importance
        # context shape = (batch_size, units)
        context = attention_weights * features
        context = tf.reduce_sum(context, axis=1)

        return context, attention_weights


class Encoder(tf.keras.Model):
    """Encoding image and question to answer the question."""
    def __init__(self, units=128):
        super(Encoder, self).__init__()
        self.dense_img = tf.keras.layers.Dense(units)
        self.dense_sent = tf.keras.layers.Dense(units)

    def call(self, img_features, sent_features):
        """Calculate encoded feature by images and questions.
        Args:
            img_features : 2D tensor object represents image feature
            sent_features: 2D tensor object represents questions embedding
        Return:
            2D tensor
                shape => (batch_size, units)
        """
        img_features = self.dense_img(img_features)
        sent_features = self.dense_sent(sent_features)
        state = tf.nn.tanh(img_features + sent_features)
        return state


class Decoder(tf.keras.Model):
    """Decoding by passing previously gererated word."""
    def __init__(self,
                 units,
                 vocab_size,
                 embedding_layer=None):
        super(Decoder, self).__init__()
        self.units = units

        self.embedding = embedding_layer
        if self.embedding is None:
            # if not provided, re-define with fixed dimmension
            self.embedding = tf.keras.layers.Embedding(vocab_size+1, 256)

        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.dense1 = tf.keras.layers.Dense(units)
        self.output_layer = tf.keras.layers.Dense(vocab_size)
        self.attention = Attention(units)

    def call(self, x, features, encoded):
        # context shape = (batch_size, units)
        context, attention_weights = self.attention(features, encoded)
        x = self.embedding(x)

        # concatenate sequence and attention weighted context
        # shape => (batch_size, 1, embedding_dim + attention_units)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        x, state = self.gru(x)

        # output shape => (batch_size, 1, units)
        x = self.dense1(x)
        # reshape to (batch_size * 1, units)
        x = tf.reshape(x, (-1, x.shape[-1]))

        x = self.output_layer(x)
        return x, state, attention_weights
