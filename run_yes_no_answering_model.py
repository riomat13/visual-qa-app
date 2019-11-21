#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# This is question answering model focusing on
# dataset whose category is 'yes/no' question
# such that it is binary classification

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

from main.settings import Config
from main.models import ClassificationModel
from main.models.train import make_training_cls_model
from main.utils.preprocess import text_processor

if tf.__version__ < '2.0.0':
    from main.metrics import calculate_accuracy_np as calculate_accuracy
else:
    from main.metrics import calculate_accuracy


# paramaters
DEBUG = False

load_config = False

# train
image_seq = 49
pad_max_len = 15  # max length from dataset is 22 and longer than 15 is < 0.1%
data_size = 150000
embedding_dim = 256
vocab_size = 20000
units = 256  # used for attention

learning_rate = 0.001

batch_size = min(64, data_size)  # for adjusting to testing
epochs = 30
display_step = 500

# make easy to calculate accuracy and loss by average
step_per_val = 100
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

    qs = [d['question'] for d in dataset]
    qs = processor(qs)

    answers = np.array([1 if d['answer'] == 'yes' else 0
                        if d['answer'] == 'no' else 2
                        for d in dataset], dtype=np.int32)
    imgs = np.array([np.load(d['image_path'], allow_pickle=True)
                     for d in dataset])

    return qs, answers, imgs


def main(train, val, *, save=False):
    max_acc = 0

    # set up training model
    model = ClassificationModel(units, pad_max_len, processor.vocab_size, embedding_dim, 3)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_cls_step = make_training_cls_model(model, optimizer,
                                             'sparse_categorical_crossentropy')

    for epoch in range(1, epochs+1):
        epoch_start = time.time()
        print('=====' * 10)
        print('  Epochs:', epoch)
        print('=====' * 10)

        batch_st = time.time()

        random.shuffle(train)

        for batch, (inputs, labels) \
                in enumerate(data_generator(train, batch_size=batch_size)):
            st = time.time()

            loss, acc = train_cls_step(inputs, labels)

            if batch % display_step == 0:
                if DEBUG:
                    pred, weights = model(*inputs)
                    print('[DEBUG] Batch: {}'.format(batch))
                    #print('[DEBUG] Average weights :'.format(batch))
                    #for layer in model.layers:
                    #    print('  Layer:', model.name + ':' + layer.name)
                    #    print('  Weights:')
                    #    print('    mean:', np.mean(layer.get_weights()[0]))
                    #    print('     std:', np.std(layer.get_weights()[0]))
                    #    print()
                    print('[DEBUG] Prediction:')
                    #print('   values:\n', pred.reshape(-1, 3))
                    print('   Pred:\n    ', np.argmax(pred, axis=-1))
                    print('  Label:\n    ', labels.reshape(-1,))
                    print('[DEBUG] Weights/Question:')
                    print(weights[0].numpy().reshape(-1))
                    print(*[processor.index_word[q] for q in questions[0] if q > 0])
                    print()

                end = time.time()
                batch_end = time.time()
                print('    Batch -', batch)
                print('      Train:  Loss - {:.4f}  Acc - {:.4f}  '
                      'Time(calc) - {:.4f}s/batch  '
                      'Time(total) - {:.4f}s/batch'.format(loss, acc,
                                                           end-st,
                                                           batch_end-batch_st)
                      )

            batch_st = time.time()

        loss_val = 0
        acc_val = 0
        count = 0

        val_st = time.time()

        # calculate validation data
        for in_val, l_val in data_generator(val, batch_size=batch_size):
            out_val = model(*in_val)
            if isinstance(out_val, tuple):
                out_val = out_val[0]
            cost = tf.keras.losses.sparse_categorical_crossentropy(
                l_val,
                out_val,
                from_logits=True
            )
            loss_val += tf.reduce_mean(cost)
            acc_val += calculate_accuracy(out_val, l_val)
            count += 1

        # calculate average
        loss_val /= count
        acc_val /= count

        val_end = time.time()

        print()
        print('      Validation(approx.): Loss - {:.4f}  Acc - {:.4f}  '
              'Time - {:.4f}s'.format(loss_val, acc_val, val_end-val_st))
        print('  Total time per epoch: {:.4f}s'.format(time.time() - epoch_start))
        print()

        # save when get the highest accuracy in validation
        score = acc_val - loss_val
        if save and acc_val > max_acc:
            max_acc = acc_val
            print('Saving model weights')
            model.save_weights(os.path.join(Config.MODELS.get('Y/N'), 'weights'))
            print('Saved!')


if __name__ == '__main__':
    from main.utils import make_parser
    parser = make_parser()

    args = parser.parse_args()

    DEBUG = args.debug

    if DEBUG:
        np.set_printoptions(precision=4)

    save = args.no_save

    st = time.time()
    print('Setting up dataset')
    with open('./data/answer_yes_no.json', 'r') as f:
        dataset = json.load(f)

    print('Total loaded data size:', len(dataset))

    random.shuffle(dataset)

    train, val = dataset[:data_size], dataset[data_size: data_size+val_size]
    print('Data size: Train: {} Val: {}'.format(len(train), len(val)))

    if args.no_config:
        # use only if words appeared in training set
        words = [d['question'] for d in train]
        processor = text_processor(words, maxlen=pad_max_len)
        assert processor(words).shape[1] == pad_max_len
    else:
        processor = text_processor(num_words=vocab_size, maxlen=pad_max_len, from_config=True)

    print('Time to setup: {:.4f}s'.format(time.time() - st))

    main(train, val, save=save)

    print('Training completed')
    print('Total running time: {:.4f}s'.format(time.time() - st))
