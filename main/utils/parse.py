#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def make_parser():
    """Build parser for running scripts.

    Given arguments by this are:
        --no-config: build text tokenizer from scratch
        -p, --path:  path to model weights if reuse pretrained one
        --debug:     run with debug mode, this displays detail
        --no-save:   do not save weights automatically
    """

    import argparse

    parser = argparse.ArgumentParser(description='Run model to answer yes/no')
    parser.add_argument(
        '--no-config', default=False, action='store_true',
        help='build text tokenizer from scratch'
    )
    parser.add_argument(
        '-p', '--path', type=str, default=None,
        help='path to model weights if reuse pretrained one'
    )
    parser.add_argument(
        '--debug', default=False, action='store_true',
        help='run with debug mode'
    )

    parser.add_argument(
        '--display-step', type=int, default=100,
        help='steps to display training score (default: 100)'
    )

    parser.add_argument(
        '--no-save', default=True, action='store_false',
        help='do not save weights automatically'
    )

    parser.add_argument(
        '-d', '--data-size', type=int, default=30000,
        help='data size to train (default: 30k)'
    )

    parser.add_argument(
        '--epoch', type=int, default=5,
        help='number of epochs for training (default: 5)'
    )

    parser.add_argument(
        '--batch', type=int, default=128,
        help='mini batch size for training/validation (default: 128)'
    )

    parser.add_argument(
        '--val-step', type=int, default=100,
        help='steps for each validation (default: 100)'
    )

    parser.add_argument(
        '--units', type=int, default=512,
        help='base hidden unit size (default: 512)'
    )

    parser.add_argument(
        '--vocab-size', type=int, default=20000,
        help='vocabulary size (default: 20000)'
    )


    return parser
