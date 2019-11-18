#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

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
        self.dense1 = tf.keras.layers.Dense(units)
        self.out_layer = tf.keras.layers.Dense(num_classes)

    def call(self, sequences):
        x = self.embedding(sequences)
        x = self.gru(x)
        x = self.dense1(x)

        # output shape = (batch_size, num_classes)
        x = self.out_layer(x)
        return x


class ClassificationModel(tf.keras.Model):
    def __init__(self, units, vocab_size, embedding_dim, num_classes):
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
        self.fc = tf.keras.layers.Dense(units, name='fc1')
        self.output_layer = tf.keras.layers.Dense(num_classes, name='output_layer')

    def call(self, imgs, qs):
        """Execute network and output the result
        Args:
            imgs: Tensor
                shape = (None, 49, 1024)
            qs: Tensor
                shape = (None, sequence_length)
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
        x = self.fc(x)
        x = self.output_layer(x)
        return x, weights
