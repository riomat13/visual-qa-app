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
from functools import partial
import json
import warnings

# ignore tensorflow debug info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ignore numpy futre warnigns
warnings.filterwarnings('ignore')


import numpy as np
import tensorflow as tf

from tensorflow.python.keras.backend import set_session

from keras_preprocessing import image

from main.settings import Config
from main.models import SequenceGeneratorModel, QuestionImageEncoder
from main.models.train import make_training_seq_model
from main.utils.preprocess import text_processor

if tf.__version__ < '2.0.0':
    from main.metrics import calculate_accuracy_np as calculate_accuracy
    from main.utils.loader import load_image_simple as load_image
else:
    from main.metrics import calculate_accuracy
    from main.utils.loader import load_image


# paramaters
DEBUG = False

# Train
vocab_size = 20000
data_size = 40000
seq_length = 15
ans_length = 5 + 2  # maximum answer length + '<bos>' and '<eos>'

embedding_dim = 256
units = 512

learning_rate = 0.001

batch_size = 128
epochs = 2
display_step = 1000

# make easy to calculate accuracy and loss by average
step_per_val = 30
val_size = batch_size * step_per_val


def data_generator(dataset, batch_size):
    steps_per_epoch = len(dataset) // batch_size

    for step in range(steps_per_epoch):
        start = step * batch_size
        batch = dataset[start:start+batch_size]
        qs, answers, imgs = data_process(batch)

        yield (qs, imgs), answers


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

    imgs = np.array([np.load(d['image_path'], allow_pickle=True)
                     for d in dataset])
    return qs, answers, imgs


def main(model_type, train, val, *, save=False):
    model_type = model_type.upper()

    if save:
        # threshold to save model weights
        min_loss = 50.0
        base_path = Config.MODELS.get(model_type)
        enc_weights_path = os.path.join(base_path, 'encoder', 'weights')
        gen_weights_path = os.path.join(base_path, 'gen', 'weights')

    encoder = QuestionImageEncoder(units, vocab_size, embedding_dim)
    model = SequenceGeneratorModel(units,
                                   vocab_size,
                                   embedding_dim,
                                   encoder.embedding)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_seq_step = make_training_seq_model(
        model,
        ans_length,
        optimizer,
        encoder_model=encoder,
        loss='sparse_categorical_crossentropy'
    )


    for epoch in range(1, epochs+1):
        epoch_start = time.time()
        print('=====' * 10)
        print('  Epochs:', epoch)
        print('=====' * 10)

        batch_start = time.time()

        random.shuffle(train)

        for batch, (inputs, labels) \
                in enumerate(data_generator(train, batch_size=batch_size)):
            st = time.time()

            # ============================
            #   Run model
            # ============================
            x = np.array([processor.word_index['<bos>']] * len(labels))
            loss, pred, attention_weights = train_seq_step(x, inputs, labels)

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
                    print(' Input:', inputs[0][0])
                    print(labels[0])
                    print(' Label: {}'.format(
                          ' '.join(ans_processor.index_word[idx]
                                   for idx in labels[0] if idx > 0)))
                    # prediction does not have <bos>
                    print('  Pred: {}'.format(
                        '<bos> ' + ' '.join(ans_processor.index_word[idx]
                                            for idx in pred[0] if idx > 0)))

                # general output
                curr_time = time.time()
                print('    Batch -', batch)
                print(f'      Train:  Loss - {loss:.4f}  '
                      f'Time(calc) - {curr_time-st:.4f}s/batch  '
                      f'Time(total) - {curr_time-batch_start:.4f}s/batch')

            batch_start = time.time()

        # after finished training in each epoch
        # evaluate model by validation dataset
        st_val = time.time()

        loss_val = 0
        predicts = []

        for in_val, l_val in data_generator(val, batch_size=batch_size):
            features, _ = encoder(*in_val)
            hidden = np.zeros((len(l_val), embedding_dim))
            batch_preds = []

            x = np.array([processor.word_index['<bos>']] * len(l_val))

            for i in range(1, ans_length):
                x, hidden, _ = model(x, in_val[0], features, hidden)
                cost = tf.keras.losses.sparse_categorical_crossentropy(
                    labels[:, i], x,
                    from_logits=True
                )
                x = tf.argmax(x, axis=-1)
                loss_val += tf.reduce_mean(cost)
                batch_preds.append(x)
            batch_preds = tf.stack(batch_preds, axis=1)
            predicts.append(batch_preds)

        loss_val /= len(predicts)
        end_val = time.time()

        print()
        print('      Validation: Loss - {:.4f}  Time - {:.4f}s'.format(
            loss_val, end_val-st_val))
        print('  Total time per epoch: {:.4f}s'.format(
            time.time() - epoch_start))
        print()

        if save and loss_val < min_loss:
            min_loss = loss_val
            print('Saving model weights')
            encoder.save_weights(enc_weights_path)
            model.save_weights(gen_weights_path)
            print('Saved!')


if __name__ == '__main__':
    from main.utils import make_parser
    parser = make_parser()

    args = parser.parse_args()

    DEBUG = args.debug

    if DEBUG:
        np.set_printoptions(precision=4)

    save = args.no_save
    model_type = args.model.lower()

    if model_type not in ('what', 'where', 'which', 'who', 'why', 'how', 'none'):
        raise ValueError(f'Invalid model type: {model_type}')

    st = time.time()
    print(f'Running - {model_type} model')
    print('Setting up dataset')
    with open(f'./data/answer_{model_type}.json', 'r') as f:
        dataset = json.load(f)

    print('Total loaded data size:', len(dataset))
    random.shuffle(dataset)

    train, val = dataset[:data_size], dataset[data_size: data_size+val_size]
    print('Data size: Train: {} Val: {}'.format(len(train), len(val)))

    # use all words from training set processed primarily
    processor = text_processor(maxlen=seq_length, from_config=True)
    ans_processor = text_processor(maxlen=ans_length, from_config=True)

    print('Time to setup: {:.4f}s'.format(time.time() - st))

    main(model_type, train, val, save=save)
    print('Training completed')
    print('Total running time: {:.4f}s'.format(time.time() - st))
