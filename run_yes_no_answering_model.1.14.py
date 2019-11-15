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

# TODO: make compatible with tf2.0
if tf.__version__ != '1.14.0':
    raise Exception('Tensorflow version has to be 1.14.0')

from tensorflow.python.keras.backend import set_session

from keras_preprocessing import image

from main.models.common import Attention
from main.utils.preprocess import text_processor

if tf.__version__ < '2.0.0':
    from main.metrics import calculate_accuracy_np as calculate_accuracy
else:
    from main.metrics import calculate_accuracy


# paramaters
DEBUG = True

if DEBUG:
    np.set_printoptions(precision=4)

# # train
image_seq = 49
pad_max_len = 15  # max length from dataset is 22 and longer than 15 is < 0.1%
data_size = 120000
embedding_dim = 256
units = 256

# due to high variance
dropout_cls1 = 0.
dropout_cls2 = 0.
dropout_cls3 = 0.

learning_rate = 0.01

batch_size = min(64, data_size)  # for adjusting to testing
epochs = 50
display_step = 100

# make easy to calculate accuracy and loss by average
step_per_val = 100
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

    qs = [d['question'] for d in dataset]
    qs = processor(qs)

    answers = np.array([1 if d['answer'] == 'yes' else 0 if d['answer'] =='no' else 2 for d in dataset])
    imgs = np.array([np.load(d['image_path'], allow_pickle=True) for d in dataset])

    return qs, answers, imgs


def main(train, val):
    global graph

    # not necessary when run this as script
    # but this is needed if run this code iteratively such as on jupyter notebook
    tf.compat.v1.reset_default_graph()

    graph = tf.Graph()

    with graph.as_default():

        with tf.name_scope('cls'):
            LABELS = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, ), name='labels')

        with tf.name_scope('questions'):
            embedding = tf.keras.layers.Embedding(processor.vocab_size+1, embedding_dim)
            q_gru = tf.keras.layers.GRU(units,
                                        return_state=True,
                                        return_sequences=True,
                                        recurrent_initializer='glorot_uniform')
            attention_q = Attention(units)
            QS = tf.compat.v1.placeholder(dtype=tf.float32,
                                          shape=(None, pad_max_len),
                                          name='encoded_questions')

        with tf.name_scope('images'):
            attention_img = Attention(units)
            IMGS = tf.compat.v1.placeholder(dtype=tf.float32,
                                            shape=(None, image_seq, 1024),
                                            name='imgs')

        # images
        img_encoded = tf.keras.layers.Dense(embedding_dim)(IMGS)

        # use last state from question encoding for attention input
        # (batch_size, seq_length, embedding_dim)
        q_encoded = embedding(QS)
        q_outputs, q_state = q_gru(q_encoded)

        # image attention
        context2, _ = attention_img(img_encoded, q_state)

        # questions
        context1, WEIGHTS = attention_q(q_outputs, context2)

        # classification
        x = tf.concat([context1, context2], axis=-1)
        x = tf.keras.layers.Dense(1024)(x)
        x = tf.keras.layers.Dense(1024)(x)
        PRED = tf.keras.layers.Dense(3)(x)

        COST = tf.keras.losses.sparse_categorical_crossentropy(LABELS, PRED, from_logits=True, axis=-1)
        LOSS = tf.reduce_mean(COST)

        OPT = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(LOSS)

    with tf.compat.v1.Session(graph=graph) as sess:

        sess.run(tf.compat.v1.global_variables_initializer())
        set_session(sess)

        if DEBUG:
            trainables = sess.run(tf.compat.v1.trainable_variables())
            print('Total trainables:', len(trainables))


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

                _, loss, cost, pred, weights  = sess.run([OPT, LOSS, COST, PRED, WEIGHTS],
                                                feed_dict={QS: questions,
                                                           LABELS: labels,
                                                           IMGS: img_features})

                end_calc = time.time()

                if DEBUG:
                    if batch % display_step == 0:
                        print('[DEBUG] Batch: {}'.format(batch))
                        #print('[DEBUG] Average weights :'.format(batch))
                        #for layer in model.layers:
                        #    print('Layer:', model.name + ':' + layer.name)
                        #    print('Weights:')
                        #    print('  mean:', np.mean(layer.get_weights()[0]))
                        #    print('   std:', np.std(layer.get_weights()[0]))
                        #    print()
                        print('[DEBUG] Prediction:')
                        #print('   values:\n', pred.reshape(-1, 3))
                        print('   Pred:\n    ', np.argmax(pred, axis=-1))
                        print('  Label:\n    ', labels.reshape(-1,))
                        print('[DEBUG] Weights/Question:')
                        print(weights[0].reshape(-1))
                        print(*[processor.index_word[q] for q in questions[0] if q > 0])
                        print()

                acc = calculate_accuracy(pred, labels)

                if batch % display_step == 0:
                    print('    Batch -', batch)
                    print('      Train:  Loss - {:.4f}  Acc - {:.4f}  Time(calc) - {:.4f}s/batch  Time(total) - {:.4f}s/batch'
                            .format(loss, acc, end_calc-st, time.time()-batch_start))

                batch_start = time.time()

            # after finished training in each epoch
            # evaluate model by validation dataset
            loss_val = 0
            acc_val = 0
            st_val = time.time()
            #for q_val, l_val, i_val in data_generator(val, batch_size=batch_size):
            # TODO:test
            for q_val, l_val, i_val in data_generator(val, batch_size=batch_size):
                _loss_val, pred_val = sess.run([LOSS, PRED],
                                                feed_dict={QS: q_val,
                                                           LABELS: l_val,
                                                           IMGS: i_val})
                l_val = l_val.ravel()
                loss_val += _loss_val
                acc_val += calculate_accuracy(pred_val, l_val)

            loss_val /= step_per_val
            acc_val /= step_per_val

            end_val = time.time()

            print()
            print('      Validation(approx.): Loss - {:.4f}  Acc - {:.4f}  Time - {:.4f}s'
                    .format(loss_val, acc_val, end_val-st_val))
            print('  Total time per epoch: {:.4f}s'.format(time.time() - epoch_start))
            print()


if __name__ == '__main__':
    st = time.time()
    print('Setting up dataset')
    with open('./data/answer_yes_no.json', 'r') as f:
        dataset = json.load(f)

    random.shuffle(dataset)

    train, val = dataset[:data_size], dataset[data_size: data_size+val_size]

    # use only if words appeared in training set
    words = [d['question'] for d in train]

    processor = text_processor(words, maxlen=pad_max_len)

    assert processor(words).shape[1] == pad_max_len

    print('Time to setup: {:.4f}s'.format(time.time() - st))

    main(train, val)
    print('Total running time: {:.4f}s'.format(time.time() - st))
