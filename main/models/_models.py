#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, GRU


class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        # FC for input feature
        self.dense1 = Dense(units)
        # FC for hidden states from encoder
        self.dense2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden_outputs):
        """
        Args:
            features:      encoded features
                shape = (batch_size, hidden_size)
            hidden_outputs: hidden outputs from encoder
                shape = (batch_size, sequence_length, embedding_dim)
        Returns:
            context: context tensor
                shape = (batch_size, hidden_size)
            attention_weights: weights used for attention
                shape = (batch_size, sequence_length, 1)
        """
        # expand dim to add time step axis => (batch_size, 1, hidden_size)
        x = tf.expand_dims(features, 1)

        # weights to update feature importance
        x_encoded = self.dense1(x)
        hidden_encoded = self.dense2(hidden_outputs)
        score_ = tf.nn.tanh(x_encoded + hidden_encoded)
        score = self.V(score_)

        # normalize output score
        attention_weights = tf.nn.softmax(score, axis=1)

        # update the feature weighted by importance
        # context shape = (batch_size, units)
        context = attention_weights * hidden_outputs
        context = tf.reduce_sum(context, axis=1)

        return context, attention_weights


class QuestionTypeClassification(tf.keras.Model):
    """Classify question type."""
    def __init__(self, embedding_dim, units, vocab_size, num_classes=None):
        # fetch number of class to be used as output for classification
        if num_classes is None:
            from main.utils.loader import fetch_question_types
            num_classes = len(fetch_question_types())

        super(QuestionTypeClassification, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(units,
                       return_sequences=False,
                       recurrent_initializer='glorot_uniform')
        self.dense1 = Dense(units, activation='relu')
        self.dense2 = Dense(256, activation='relu')
        self.out_layer = Dense(num_classes, activation='softmax')

    def call(self, sequences):
        x = self.embedding(sequences)
        x = self.gru(x)
        x = self.dense1(x)
        x = self.dense2(x)

        # output shape = (batch_size, num_classes)
        x = self.out_layer(x)
        return x


class Encoder(tf.keras.Model):
    """Encoding image and question to answer the question."""
    def __init__(self, units=128):
        super(Encoder, self).__init__()
        self.dense_img = Dense(units)
        self.dense_sent = Dense(units)

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
            self.embedding = Embedding(vocab_size, 256)

        self.gru = GRU(units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform')
        self.dense1 = Dense(units)
        self.output_layer = Dense(vocab_size)
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
