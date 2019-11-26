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
seq_length = 15
ans_length = 5 + 2  # maximum answer length + '<bos>' and '<eos>'

embedding_dim = 256
learning_rate = 0.001


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


def run(model_type, train, val, *,
        units=512,
        embedding_dim=embedding_dim,
        vocab_size=20000,
        learning_rate=learning_rate,
        sequence_length=ans_length,
        save=False):
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
        sequence_length,
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
            features, q_embedded = encoder(*in_val)
            hidden = np.zeros((len(l_val), embedding_dim))
            batch_preds = []

            x = np.array([processor.word_index['<bos>']] * len(l_val))

            for i in range(1, ans_length):
                x, hidden, _ = model(x, q_embedded, features, hidden)
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
    from main.utils.parse import make_parser
    parser = make_parser()
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help='model type to execute, for instnce, "what", "which"'
    )

    args = parser.parse_args()

    DEBUG = args.debug

    if DEBUG:
        np.set_printoptions(precision=4)

    save = args.no_save
    model_type = args.model.lower()

    batch_size = args.batch
    epochs = args.epoch
    units = args.units
    vocab_size = args.vocab_size

    display_step = args.display_step

    if model_type not in ('what', 'where', 'which', 'who', 'why', 'how', 'none'):
        raise ValueError(f'Invalid model type: {model_type}')

    st = time.time()
    print(f'Running - {model_type} model')
    print('Setting up dataset')
    with open(f'./data/answer_{model_type}.json', 'r') as f:
        dataset = json.load(f)

    print('Total loaded data size:', len(dataset))
    random.shuffle(dataset)

    data_size = args.data_size
    val_size = args.batch * args.val_step
    if data_size + val_size > len(dataset):
        raise ValueError('Indicated data size exceeds actual dataset size')

    train, val = dataset[:data_size], dataset[data_size: data_size+val_size]

    print()
    print('  Parameters')
    print('  ----------')
    print('  Data size:')
    print(f'      Train: {len(train):>7}')
    print(f'      Val:   {len(val):>7}')
    print()
    print(f'  Epoch:            {epochs}')
    print(f'  Batch Size:       {batch_size}')
    print(f'  Hidden unit size: {units}')
    print(f'  Vocabulary size:  {vocab_size}')
    print()

    # use all words from training set processed primarily
    processor = text_processor(maxlen=seq_length, from_config=True)
    ans_processor = text_processor(maxlen=ans_length, from_config=True)

    print('Time to setup: {:.4f}s'.format(time.time() - st))
    print()

    run(model_type, train, val,
        units=units,
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        learning_rate=learning_rate,
        sequence_length=ans_length,
        save=save)
    print('Training completed')
    print('Total running time: {:.4f}s'.format(time.time() - st))
