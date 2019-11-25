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
        '-m', '--model', type=str, required=True,
        help='model type to execute, for instnce, "what", "which"'
    )
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
        '--no-save', default=True, action='store_false',
        help='do not save weights automatically'
    )

    return parser
