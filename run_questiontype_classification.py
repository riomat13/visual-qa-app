#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# This is to run model to predict question type by question sentence
# Question types are defined at:
#   https://github.com/GT-Vision-Lab/VQA/tree/master/QuestionTypes

import os
import time
import json

import numpy as np
import tensorflow as tf

from main.settings import ROOT_DIR
from main.models import QuestionTypeClassification
from main.utils.loader import VQA, fetch_question_types
from main.utils.preprocess import text_processor, one_hot_converter
from main.metrics import calculate_accuracy

# ignore tensorflow debug info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# dataset
data_size = 30000

# parameters
embedding_dim = 256
hidden_units = 64

batch_size = 32
epochs = 2

# initialize labels
classes = fetch_question_types()
q2id = {q: i for i, q in enumerate(classes)}
num_classes = len(classes)

# preprocess
processor = None


def data_generator(inputs, labels, batch_size=batch_size):
    steps_per_epoch = (len(inputs)-1) // batch_size + 1

    for step in range(steps_per_epoch):
        start = step * batch_size
        batch_inputs = inputs[start:start+batch_size]
        batch_labels = labels[start:start+batch_size]
        yield batch_inputs, batch_labels


def main(*, training=True, save_to=None, load_from=None, val=0.2):
    global data_size
    global num_classes
    global processor

    vqa = VQA()
    vqa.load_data(num_data=data_size)
    questions, question_types, _, _ = next(vqa.data_generator())
    labels = [
        q2id[q] if q in q2id else q2id['none of the above']
        for q in question_types]

    # set text processor
    if load_from is not None:
        # has to reuse past tokenizer for consistency
        with open('./.env/tokenizer_config.json', 'r') as f:
            cfg = f.read()

        processor = text_processor(cfg, from_json=True)
    
    # build processor based on training dataset
    # if processor is not reused
    if training:
        # preprocessing dataset
        # split train and test set
        train_size = int(data_size * (1 - val))

        # inputs
        inputs_train = questions[:train_size]
        inputs_val = questions[train_size:]

        # process inputs
        # if tokenizer is not loaded, create new one
        if processor is None:
            processor = text_processor(inputs_train)

    # iinitialize model
    model = QuestionTypeClassification(
        embedding_dim=embedding_dim,
        units=hidden_units,
        vocab_size=processor.vocab_size+1,  # need to add 1 due to Embedding implementation
        num_classes=num_classes
    )

    # set initial weights to the model
    if load_from is not None:
        model.load_weights(load_from)

    # TRAINING STEP
    if training:
        inputs_train = processor(inputs_train)
        inputs_val = processor(inputs_val)

        # labels
        labels = one_hot_converter(labels, C=num_classes)

        labels_train = labels[:train_size]
        labels_val = labels[train_size:]

        loss = 0
        optimizer = tf.keras.optimizers.Adam()


        @tf.function
        def training_step(inputs, labels, inputs_val, labels_val, num_classes):

            # TODO: add validation step
            with tf.GradientTape() as tape:
                out = model(inputs)
                loss = \
                    tf.keras.losses.categorical_crossentropy(labels, out)
                loss = tf.reduce_mean(loss)

            total_loss = (loss / int(labels.shape[1]))

            trainables = model.trainable_variables
            gradients = tape.gradient(loss, trainables)
            optimizer.apply_gradients(zip(gradients, trainables))

            # count correct predictions
            accuracy = calculate_accuracy(out, labels)

            # validation
            out_val = model(inputs_val)
            accuracy_val = calculate_accuracy(out_val, labels_val)

            return loss, total_loss, accuracy, accuracy_val

        # execute training
        for epoch in range(epochs):
            print('=====' * 10)
            print('    Epoch {}'.format(epoch+1))
            print('=====' * 10)

            dataset = data_generator(inputs_train, labels_train, batch_size)

            for batch, (ins, outs) in enumerate(dataset, 1):
                st = time.time()
                batch_loss, _, accuracy, accuracy_val = \
                    training_step(ins, outs, inputs_val, labels_val, num_classes)
                end = time.time()

                if batch % 100 == 0:
                    batch_loss = batch_loss.numpy()
                    print('  Batch:', batch)
                    # TODO: add accuracy
                    print('    Loss: {:.4f}  Accuracy(Train): {:.4f}  Accuracy(Val): {:.4f}  Time(batch): {:.4f}s'
                            .format(batch_loss, accuracy, accuracy_val, end-st))

        print('Saving models...')
        # save tokenizer info for resuse
        processor.to_json('./.env/tokenizer_config.json')
        model.save_weights(save_to)
        print('Saved!!')

        print()
        print('Training completed')

    else:
        # if not training mode test with all given data
        st = time.time()
        inputs = processor(questions)
        out = model(inputs)
        labels = tf.Variable(labels, dtype=tf.int32)
        accuracy = calculate_accuracy(out, labels)
        end = time.time()
        print('Evaluated score: Accuracy: {:.4f} Time: {:.4f}s'
                .format(accuracy, end-st))

    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
            description='Run question type classification model')
    parser.add_argument(
        '--no-train', default=False, action='store_true',
        help="not run training step (must set '-p', otherwise run training)"
    )
    parser.add_argument(
        '-s', '--save', type=str, default='weights',
        help='file name to save as checkpoint'
    )
    parser.add_argument(
        '-p', '--path', type=str, default=None,
        help='path to model data to load'
    )
    parser.add_argument(
        '-i', '--interactive', default=False, action='store_true',
        help='interactive mode. can pass sentence to predict from stdin.'
    )

    args = parser.parse_args()

    interactive = args.interactive
    load_from = args.path

    # if set no-train, not run training step
    training = True ^ args.no_train

    file_name = args.save
    save_to = f'{ROOT_DIR}/checkpoints/train/question_types/{file_name}'

    model = main(training=training, load_from=load_from, save_to=save_to)

    if interactive:
        print()
        print('-----' * 10)
        print('    Interactive mode')
        print('-----' * 10)
        sentence = input("  Input sentece(if quit, type 'q'): ").strip()

        while sentence != 'q':
            sentence = processor([sentence])

            # avoid to be broken by inputs with only unseen word nor empty
            # however this will appear DeprecatedWarning
            if not np.any(sentence):
                sentence = np.array([[0]])

            pred = model(sentence)
            pred = np.argmax(pred[0])
            print('  Predicted type => ', classes[pred])
            print()
            sentence = input("  Input sentece(if quit, type 'q'): ").strip()

    print()
    print('Closing...')
