#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


def text_processor(sentences):
    """Process text with Tokenizer and pad_sequence from keras.
    Padding by 0 and fix the lengths to the logest one.

    Args:
        sentences: a list of str
            this will be fit on tokenizer

    Returns:
        tokenizer function

    Attributes:
        vocab_size: int
            current vocabulary size
            can be updated by update()

    Methods:
        update(sentences):
            update tokenizer

    Example:
        >>> dataset = ['He is a boy', 'You are a student', 'This is an example sentence']
        >>> processor = text_processor(dataset)
        >>> processed = processor(dataset)
        >>> print(processed)
        [[ 3  1  2  4  0]
         [ 5  6  2  7  0]
         [ 8  1  9 10 11]]

        # skip words if not exist in initial dataset
        >>> new_sents = ['This is a new sentence']
        >>> processed = processor(new_sents)
        >>> print(processed)
        [[8 1 2 11]]  # new is not contained in dataset

        # you can get vocabulary size
        >>> processor.vocab_size
        11

        # and also it can be updated
        # the IDs can be vary due to Tokenizer implementation
        >>> processor.update(new_sents)
        # now 'new' is included in the vocabulary
        >>> processor(new_sents)
        [[ 3  1  2 12  4]]
        >>> processor.vocab_size
        12
    """
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(sentences)

    def processor(sentences):
        """Convert sentences into padded sequences based on
        internal vocabulary.

        Args:
            sentences: a list of texts(str)
        """
        nonlocal tokenizer

        tensor = tokenizer.texts_to_sequences(sentences)
        tensor = pad_sequences(tensor, padding='post')
        return tensor

    def update(sentences):
        """Update internal vocabulary.

        Args:
            sentences: a list of texts(str)
        """
        nonlocal tokenizer
        nonlocal processor

        tokenizer.fit_on_texts(sentences)
        processor.vocab_size = len(tokenizer.word_index)

    processor.vocab_size = len(tokenizer.word_index)
    processor.update = update

    return processor
