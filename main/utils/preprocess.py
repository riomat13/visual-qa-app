#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json

import numpy as np

from keras_preprocessing.text import Tokenizer, tokenizer_from_json
from keras_preprocessing.sequence import pad_sequences

from main.settings import Config


def one_hot_converter(labels, C):
    """Convert 1-d array of labels to 2-d matrix consist of one-hot vectors of labels.
    Args:
        labels: list or numpy.ndarray
        C: int
            number of classes
    """
    data_size = len(labels)
    vector = np.zeros((data_size, C), dtype=np.int8)
    vector[np.arange(data_size), labels] = 1
    return vector


def text_processor(inputs='', num_words=None, maxlen=None, *, from_json=False, from_config=False):
    """Process text with Tokenizer and pad_sequence from keras.

    Padding by 0 and fix the lengths to the logest one.
    Unknown word will be conerted to '<unk>' and padded word will be
    '<pad>'. Each word index is 1 and 0 respectively.

    Args:
        inputs: str or a list of str
            this will be fit on tokenizer
        num_words: int
            the maximum number of words to keep,
            based on word frequency.
        maxlen: int
            if this is set to a certain number
            the output sequence length will be fixed to the given number
            by padding with '<pad>'.
            if the length is longer than the number, it will be truncated.
        from_json: boolean
            if this is set to `True`, created tokenizer object
            based on json string.
            if saved as json file, load it beforehand
        from_config: boolean
            if this is set to `True`, load file in location set in config
            `from_json` will be ignored if both set to be `True`

    Returns:
        tokenizer function

    Attributes:
        vocab_size: int
            current vocabulary size
            can be updated by update()
        index_word: dict
        word_index: dict
            dicts storing word index and word pairs

    Methods:
        update(sentences: list(str)):
            update tokenizer
        to_json(**kwargs):
            dump JSON string containing the tokenizer configuration

    Example:
        >>> dataset = ['He is a boy', 'You are a student', 'This is an example sentence']
        >>> processor = text_processor(dataset)
        >>> processed = processor(dataset)
        >>> print(processed)
        [[ 4  2  3  5  0]
         [ 6  7  3  8  0]
         [ 9  2 10 11 12]]

        # skip words if not exist in initial dataset
        >>> new_sents = ['This is a new sentence']
        >>> processed = processor(new_sents)
        >>> print(processed)
        [[ 9  2  3  1 12]]
        # word index 1 means out of vocabulary
        >>> print(processor.index_word[1])
        <unk>

        # you can get vocabulary size
        >>> processor.vocab_size
        13

        # and also it can be updated
        # the IDs can be vary due to Tokenizer implementation
        >>> processor.update(new_sents)
        # now 'new' is included in the vocabulary
        #
        # Note: once apply update, the word index may be changed
        # based on word frequency such as following example.
        >>> print(processor(new_sents))
        [[ 4  2  3 13  5]]
        >>> processor.vocab_size
        13
    """
    if from_config:
        with open(Config.MODELS['TOKENIZER'].get('path'), 'r') as f:
            config = f.read()
        tokenizer = tokenizer_from_json(config)
    elif from_json:
        tokenizer = tokenizer_from_json(inputs)
    else:
        if isinstance(inputs, str):
            inputs = [inputs]
        tokenizer = Tokenizer(num_words=num_words,
                              oov_token='<unk>',
                              filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~')
        tokenizer.fit_on_texts(inputs)

    def processor(sentences):
        """Convert sentences into padded sequences based on
        internal vocabulary.

        Args:
            sentences: str or a list of str
        """
        nonlocal tokenizer

        if isinstance(sentences, str):
            sentences = [sentences]

        tensor = tokenizer.texts_to_sequences(sentences)
        tensor = pad_sequences(tensor, maxlen=maxlen, padding='post')
        return tensor

    def _update():
        """Helper function to update internal vocabulary."""
        nonlocal processor
        nonlocal tokenizer

        tokenizer.word_index['<pad>'] = 0
        tokenizer.index_word[0] = '<pad>'

        if num_words is not None:
            tokenizer.num_words = num_words
            processor.vocab_size = num_words
        else:
            processor.vocab_size = len(tokenizer.word_index)
        processor.index_word = tokenizer.index_word
        processor.word_index = tokenizer.word_index

    def update(sentences):
        """Update internal vocabulary.

        Args:
            sentences: a list of texts(str)
        """
        nonlocal tokenizer
        nonlocal processor

        tokenizer.fit_on_texts(sentences)
        _update()

    def to_json(file_path=None, **kwargs):
        """Returns a JSON string containing the tokenizer configuration.
        This is equivalent to `to_json` method in Tokenizer object.

        Args:
            file_path: str or None
                if not None, save tokenizer config with JSON
                to the given file path
            **kwargs:
                Additional Keyword arguments to be passed
                to `json.dumps()`
        Returns:
            JSON string
        """
        nonlocal tokenizer

        json_cfg = tokenizer.to_json(**kwargs)

        # if set the path, save to file
        if file_path is not None:
            with open(file_path, 'w') as f:
                f.write(json_cfg)
        return json_cfg

    processor.update = update
    processor.to_json = to_json
    _update()

    return processor


def data_generator(dataset, batch_size, expand=False, process_func=None):
    """Generate data with batch.

    Args:
        dataset: list or tuple
            if process_func is given, each element will be processed
            with this
        batch_size: int
            batch size to generate in each iteration
        expand: boolean
            set this to True if each element in dataset has
            multiple data such as inputs and labels
            otherwise it may generate data with unexpected way
        process_func: function
            apply to batch dataset

    Return:
        process_func(sub batch)
    """
    # this can generate all dataset even if indivisible
    steps_per_batch = (len(dataset)-1) // batch_size + 1

    for step in range(steps_per_batch):
        start = step * batch_size
        batch = dataset[start:start+batch_size]

        if process_func is not None:
            batch = process_func(batch)

        if expand:
            yield list(zip(*batch))
        else:
            yield batch
