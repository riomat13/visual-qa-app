#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# This is question answering model focusing on
# dataset whose category is 'yes/no' question
# such that it is binary classification
#
# This is written for running on NVIDIA Jetson so that
# it will be not for tensorflow 2.0 but for 1.14
# which is the latest version supported by NVIDIA

import os
import random
import time
import json
import warnings

# ignore tensorflow debug info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ignore numpy futre warnigns
warnings.filterwarnings('ignore')


import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2

# TODO: make compatible with tf2.0
if tf.__version__ != '1.14.0':
    raise Exception('Tensorflow version has to be 1.14.0')

from tensorflow.python.keras.backend import set_session

from keras_preprocessing import image

from main.models.common import Attention
from main.utils.preprocess import text_processor

if tf.__version__ < '2.0.0':
    from main.metrics import calculate_accuracy_np as calculate_accuracy
    from main.utils.loader import load_image_simple as load_image
else:
    from main.metrics import calculate_accuracy
    from main.utils.loader import load_image


# paramaters
DEBUG = True

if DEBUG:
    np.set_printoptions(precision=4)

# Train
vocab_size = 20000
data_size = 60000
seq_length = 15
ans_length = 5  # maximum answer length

embedding_dim = 256
units = 512

learning_rate = 0.001

batch_size = 128
epochs = 20
display_step = 100

# make easy to calculate accuracy and loss by average
step_per_val = 1
val_size = batch_size * step_per_val


def data_generator(dataset, batch_size):
    steps_per_epoch = (len(dataset)-1) // batch_size + 1

    for step in range(steps_per_epoch):
        start = step * batch_size
        batch = dataset[start:start+batch_size]
        qs, answers, imgs = data_process(batch)

        yield qs, answers, imgs


# Load data predprocessed
# Format:
# a list of dict:
#   keys = {'question', 'questionType', 'answer', 'answerType', 'image_path'}
def data_process(dataset):
    global processor
    global ans_processor

    qs = [d['question'] for d in dataset]
    qs = processor(qs)

    answers = ['<BOS> ' + d['answer'] + ' <EOS>' for d in dataset]
    answers = ans_processor(answers)

    imgs = np.array([np.load(d['image_path'], allow_pickle=True) for d in dataset])
    return qs, answers, imgs


class PreEncoder(tf.keras.Model):
    def __init__(self, units):
        super(PreEncoder, self).__init__()
        # questions
        self.embedding = tf.keras.layers.Embedding(vocab_size+1, embedding_dim)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        # images
        #self.dense_img = tf.keras.layers.Dense(embedding_dim, name='image_dense')
        self.attention_img = Attention(units)

    def call(self, qs, imgs):
        # questions
        qs = self.embedding(qs)
        q_encoded, q_state = self.gru(qs)

        # images
        # encode image data
        # shape => (batch_size, 49(=7x7), embedding_dim)
        img_encoded = imgs
        context_img, _ = self.attention_img(img_encoded, q_state)
        return q_encoded, context_img


class Encoder(tf.keras.Model):
    def __init__(self, units):
        super(Encoder, self).__init__()
        self.attention_q = Attention(units)

        self.fc_out = tf.keras.layers.Dense(units)

    def call(self, q_encoded, context_img, ):

        # apply attentions to each question sequence and image
        context_q, weights = self.attention_q(q_encoded, context_img)
        x = tf.concat([context_img, context_q], axis=-1)
        out = self.fc_out(x)
        return out, weights


class AnswerDecodingModel(tf.keras.Model):
    def __init__(self, units, vocab_size, embedding_layer):
        super(AnswerDecodingModel, self).__init__()

        # questions
        # reuse embedding layer since it is the same language
        self.embedding = embedding_layer
        self.attention = Attention(units)

        # classification('yes', 'no' or 'others')
        self.gru = tf.keras.layers.GRU(256,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        # normal dense
        self.fc1 = tf.keras.layers.Dense(1024, name='fc1')
        self.fc2 = tf.keras.layers.Dense(1024, name='fc2')

        self.fc = tf.keras.layers.Dense(1024, name='fc')
        self.output_layer = tf.keras.layers.Dense(vocab_size, name='output_layer')

    def call(self, x, qs, encoded, hidden):
        """Execute network and output the result
        Args:
            x: Tensor, shape = (None, 1)
                input word tensor
            encoded: Tensor, shape = (None, units)
            hidden: Tensor, shape = (None, units)
                hidden states from previous step
        """
        # encode input => (batch_size, 1, embedding_dim)
        x = tf.expand_dims(x, axis=1)
        x = self.embedding(x)

        #encoded = tf.expand_dims(encoded, axis=1)

        qs = self.embedding(qs)
        context, weihgts = self.attention(qs, hidden)
        context = tf.expand_dims(context, 1)

        # run GRU with input word, image/question encoded, question with attention
        x = tf.concat([x, context], axis=-1)
        x, state = self.gru(x)

        x = tf.concat([x, encoded, hidden], axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc(x)
        x = self.output_layer(x)
        return x, state


def main(train, val):
    global graph

    graph = tf.Graph()

    with graph.as_default():
        pre_encoder = PreEncoder(units)
        encoder = Encoder(units)
        model = AnswerDecodingModel(units, vocab_size, pre_encoder.embedding)

        QS = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, seq_length), name='encoded_questions')
        IMGS = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 49, 1024), name='imgs')
        INPUTS = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, ), name='inputs')
        LABELS = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, ans_length), name='labels')
        LOSS = tf.zeros([1,], tf.float32)

        STATE = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, embedding_dim), name='decoder_state')

        _pred = []

        prev = STATE
        q_encoded, context_img = pre_encoder(QS, IMGS)
        encoded, W = encoder(q_encoded, context_img)

        inputs = INPUTS

        for i in range(1, ans_length):
            PRED_, prev = model(inputs, QS, encoded, prev)
            _pred.append(PRED_)
            COST = tf.keras.losses.sparse_categorical_crossentropy(LABELS[:, i], PRED_,
                                                                   from_logits=True, axis=-1)
            LOSS += tf.reduce_mean(COST)

            # teacher forcing
            inputs = LABELS[:, i]


        PRED = tf.stack(_pred, axis=1)
        PRED = tf.argmax(PRED, axis=-1)
        OPT = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate) \
            .minimize(LOSS)

    tf.compat.v1.reset_default_graph()

    # begging of sentence to predict answer

    config_proto = tf.ConfigProto()
    off = rewriter_config_pb2.RewriterConfig.OFF
    config_proto.graph_options.rewrite_options.arithmetic_optimization = off

    with tf.compat.v1.Session(graph=graph, config=config_proto) as sess:

        sess.run(tf.compat.v1.global_variables_initializer())
        set_session(sess)

        for epoch in range(1, epochs+1):
            epoch_start = time.time()
            print('=====' * 10)
            print('  Epochs:', epoch)
            print('=====' * 10)

            batch_start = time.time()

            random.shuffle(train)

            for batch, (questions, labels, img_features) \
                    in enumerate(data_generator(train, batch_size=batch_size)):
                st = time.time()

                # TODO: currently mini batch size is vary due to missing files

                state = np.zeros((labels.shape[0], embedding_dim), dtype=np.float32)
                inputs = np.array([processor.word_index['<bos>']] * labels.shape[0])

                _, loss, pred, weights = sess.run([OPT, LOSS, PRED, W], feed_dict={QS: questions,
                                                                       INPUTS: inputs,
                                                                       LABELS: labels,
                                                                       STATE: state,
                                                                       IMGS: img_features})

                if batch % display_step == 0:
                    if DEBUG:
                        print('[DEBUG] Batch:', batch)
                        #print('[DEBUG] Average weights:')
                        #for layer in model.layers:
                        #    print('Layer:', model.name + ':' + layer.name)
                        #    print('  weights:')
                        #    print('    mean:', np.mean(layer.get_weights()[0]))
                        #    print('    std: ', np.std(layer.get_weights()[0]))
                        print('[DEBUG] Predicted Sentence:')
                        print(' Input:', questions[0])
                        print(' weights:', weights[0].reshape(-1))
                        print(labels[0])
                        print(' Label: {}'.format(
                            ' '.join(ans_processor.index_word[idx] for idx in labels[0] if idx > 0)))
                        # prediction does not have <bos>
                        print('  Pred: {}'.format(
                            '<bos> ' + ' '.join(ans_processor.index_word[idx] for idx in pred[0] if idx > 0)))

                    # general output
                    print('    Batch -', batch)
                    print('      Train:  Loss - {:.4f}  Time(calc) - {:.4f}s/batch  Time(total) - {:.4f}s/batch'
                            .format(loss[0], time.time()-st, time.time()-batch_start))

                batch_start = time.time()

            # after finished training in each epoch
            # evaluate model by validation dataset
            loss_val = 0
            acc_val = 0
            st_val = time.time()
            # TODO: need to build evaluate function without teacher forcing
            for q_val, l_val, i_val in data_generator(val, batch_size=batch_size):
                state = np.zeros((l_val.shape[0], embedding_dim), dtype=np.float32)
                inputs = np.array([processor.word_index['<bos>']] * l_val.shape[0])
                _loss_val = sess.run(LOSS, feed_dict={QS: q_val,
                                                      INPUTS: inputs,
                                                      LABELS: l_val,
                                                      STATE: state,
                                                      IMGS: i_val})
                loss_val += _loss_val[0]

            loss_val /= step_per_val

            end_val = time.time()

            print()
            print('      Validation: Loss - {:.4f}  Time - {:.4f}s'
                    .format(loss_val, end_val-st_val))
            print('  Total time per epoch: {:.4f}s'.format(time.time() - epoch_start))
            print()


if __name__ == '__main__':
    st = time.time()
    print('Setting up dataset')
    with open('./data/answer_what.json', 'r') as f:
        dataset = json.load(f)

    random.shuffle(dataset)

    train, val = dataset[:data_size], dataset[data_size: data_size+val_size]

    # use all words from training set processed primarily
    processor = text_processor(maxlen=seq_length, from_config=True)
    ans_processor = text_processor(maxlen=ans_length, from_config=True)

    print('Time to setup: {:.4f}s'.format(time.time() - st))

    main(train, val)
    print('Training completed')
    print('Total running time: {:.4f}s'.format(time.time() - st))
