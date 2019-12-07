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
            out = tf.reshape(out, (-1, 49, 1024))

            model = tf.keras.Model(
                inputs=mobilenet_input,
                outputs=out
            )
        return model
    return model_generator


# generate mobilenet base model function
get_mobilenet_encoder = _get_mobilenet_encoder()


class _AdditiveAttention(tf.keras.Model):
    def __init__(self, units):
        super(_AdditiveAttention, self).__init__()
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


class _DotAttention(tf.keras.Model):
    def __init__(self, seq_length):
        super(_DotAttention, self).__init__()
        self.repeat = tf.keras.layers.RepeatVector(seq_length)

    def call(self, features, states):
        """
        Args:
            features: encoded features from RNN
                shape = (batch_size, sequence_length, units)
            states:   hidden states
                shape = (batch_size, hidden_size)
        Returns:
            context: context tensor
                shape = (batch_size, hidden_size)
            attention_weights: weights used for attention
                shape = (batch_size, sequence_length, 1)
        """
        states = self.repeat(states)

        # calculate attention weights with dot score
        # (batch_size, sequence_length, units)
        score = features * states
        score = tf.reduce_sum(score, axis=-1)
        score = tf.expand_dims(score, axis=-1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # update the feature weighted by importance
        # context shape = (batch_size, units)
        context = attention_weights * features
        context = tf.reduce_sum(context, axis=1)

        return context, attention_weights


class Attention(tf.keras.Model):
    def __init__(self, units, seq_length, mode='dot'):
        super(Attention, self).__init__()
        # TODO: add other score functions
        mode = mode.lower()
        if mode == 'dot':
            self.model = _DotAttention(seq_length)
        elif mode == 'additive':
            self.model = _AdditiveAttention(units)
        else:
            raise ValueError('Choose mode from [`dot`, `additive`]')

    def call(self, features, states):
        return self.model(features, states)


class SimpleQuestionImageEncoder(tf.keras.Model):
    """Encode questions and image by embedding and dense."""
    def __init__(self, units, vocab_size, embedding_dim):
        super(SimpleQuestionImageEncoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size+1, embedding_dim)
        self.dense = tf.keras.layers.Dense(units)

    def call(self, qs, imgs):
        """Encoding Question and Image

        Args:
            qs: question
                shape: (batch_size, seq_length)
            imgs: images
                shape: (batch_size, 49, 1024)

        Return:
            q_embedded: embedded question
                shape: (batch_size, seq_length, embedding_dim)
            img_features: image features
                shape: (batch_size, units)
        """
        q_embedded = self.embedding(qs)
        img_features = self.dense(imgs)
        return q_embedded, img_features


class QuestionImageEncoder(tf.keras.Model):
    """Encode questions and image by embedding and dense."""
    def __init__(self, units, vocab_size, embedding_dim):
        super(QuestionImageEncoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size+1, embedding_dim)
        self.dense = tf.keras.layers.Dense(units)
        self.bi_gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(embedding_dim,
                                return_sequences=True,
                                recurrent_initializer='glorot_uniform'),
            merge_mode='mul',
        )

    def call(self, qs, imgs):
        """Encoding Question and Image

        Args:
            qs: question
                shape: (batch_size, seq_length)
            imgs: images
                shape: (batch_size, 49, 1024)

        Return:
            q_embedded: embedded question
                shape: (batch_size, seq_length, embedding_dim)
            img_features: image features
                shape: (batch_size, units)
        """
        qs_features = self.embedding(qs)
        qs_features = self.bi_gru(qs_features)
        imgs_encoded = self.dense(imgs)
        return qs_features, imgs_encoded
