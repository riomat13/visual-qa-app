#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from main.settings import Config
from .common import Attention


class QuestionTypeClassification(tf.keras.Model):
    """Classify question type."""
    def __init__(self, embedding_dim, units, vocab_size, num_classes=None):
        # fetch number of class to be used as output for classification
        if num_classes is None:
            from main.utils.loader import fetch_question_types
            num_classes = len(fetch_question_types())

        super(QuestionTypeClassification, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size+1, embedding_dim)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=False,
                                       recurrent_initializer='glorot_uniform')
        self.dense = tf.keras.layers.Dense(units)
        self.out_layer = tf.keras.layers.Dense(num_classes)

    def call(self, sequences):
        x = self.embedding(sequences)
        x = self.gru(x)
        x = self.dense(x)

        # output shape = (batch_size, num_classes)
        x = self.out_layer(x)
        return x


class ClassificationModel(tf.keras.Model):
    def __init__(self,
                 units,
                 seq_length,
                 vocab_size,
                 embedding_dim,
                 num_classes):
        super(ClassificationModel, self).__init__()
        # images
        self.attention_img = Attention(units)

        # questions
        self.embedding = tf.keras.layers.Embedding(vocab_size+1, embedding_dim)
        self.gru_q = tf.keras.layers.GRU(units,
                                         return_state=True,
                                         return_sequences=True,
                                         recurrent_initializer='glorot_uniform')
        self.attention_q = Attention(units)

        # classification('yes', 'no' or 'others')
        self.fc1 = tf.keras.layers.Dense(512, name='fc1')
        self.fc2 = tf.keras.layers.Dense(512, name='fc2')
        self.output_layer = tf.keras.layers.Dense(num_classes, name='output_layer')

    def call(self, qs, imgs):
        """Execute network and output the result
        Args:
            qs: Tensor
                shape = (None, sequence_length)
            imgs: Tensor
                shape = (None, 49, 1024)
        Return:
            x: Tensor
                shape = (None, num_classes)
            weights: shape
                shape = (None, sequence_length)
                This is weighted importance of each word in input sequences.
        """
        # encode question sequence
        # shape => (batch_size, sequence_length, units)
        q_encoded = self.embedding(qs)
        q_outputs, q_state = self.gru_q(q_encoded)

        # apply attentions to each question sequence and image
        context_img, _ = self.attention_img(imgs, q_state)
        context_q, weights = self.attention_q(q_outputs, context_img)

        # concatenate question and image features
        x = tf.concat([context_q, context_img, q_state], axis=-1)

        # FC layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output_layer(x)
        return x, weights


class SequenceGeneratorModel(tf.keras.Model):
    """Generate word based on input context."""
    def __init__(self,
                 units,
                 vocab_size,
                 embedding_dim,
                 embedding_layer=None):

        super(SequenceGeneratorModel, self).__init__()

        if embedding_layer is None:
            embedding_layer = tf.keras.layers.Embedding(vocab_size+1,
                                                        embedding_dim)
        self.embedding = embedding_layer
        self.attention = Attention(units)
        self.gru = tf.keras.layers.GRU(256,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        self.fc1 = tf.keras.layers.Dense(1024)
        self.fc2 = tf.keras.layers.Dense(1024)
        self.output_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, x, qs, features, hidden):
        """Generate word based on input word.
        Args:
            x: tensor represents input words
                shape = (None, )
            qs: tensor
                embedded sequence
                shape = (None, sequence_length, embedding_dim)
            features: tensor
                encoded features to be used for prediction
                shape = (None, units)
            hidden: tensor
                previous state
                shape = (None, units)
        Return:
            (next_word, hidden_state, attention_weights)
            next_word: tensor
                generated word
                shape = (None, vocab_size)
            hidden_state: tensor
                last state to use for next step
                shape = (None, units)
            attention_weights: tensor
                weights of input sequence
                shape = (None, seqence_length)
        """
        # make input as sequence with length 1
        # shape => (None, 1, embedding_dim)
        x = tf.expand_dims(x, axis=1)
        x = self.embedding(x)

        # apply attention based on input words to get context vector
        # context shape => (None, units)
        qs = self.embedding(qs)
        context, weights = self.attention(qs, hidden)
        context = tf.expand_dims(context, axis=1)

        x = tf.concat([x, context], axis=-1)
        x, state = self.gru(x)

        x = tf.concat([x, features], axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output_layer(x)
        return x, state, weights


class QuestionImageEncoder(tf.keras.Model):
    """Encode question and image."""
    def __init__(self, units, vocab_size, embedding_dim):
        super(QuestionImageEncoder, self).__init__()
        # questions
        self.embedding = tf.keras.layers.Embedding(vocab_size+1,
                                                   embedding_dim)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.attention_q = Attention(units)

        # images
        self.attention_img = Attention(units)

        self.fc = tf.keras.layers.Dense(units)

    def call(self, qs, imgs):
        # questions
        qs = self.embedding(qs)
        q_encoded, q_state = self.gru(qs)

        # images
        # encode image data
        # shape => (batch_size, 49(=7x7), embedding_dim)
        img_encoded = imgs
        context_img, _ = self.attention_img(img_encoded, q_state)

        # apply attentions to each question sequence and image
        context_q, weights = self.attention_q(q_encoded, context_img)
        x = tf.concat([context_img, context_q], axis=-1)
        features = self.fc(x)
        return features, weights
