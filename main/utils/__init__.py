#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def make_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Run model to answer yes/no')
    parser.add_argument(
        '--no-config', default=False, action='store_true',
        help='build text tokenizer from scratch'
    )
    parser.add_argument(
        '-p', '--path', type=str, default=None,
        help='path to model weights if reuse pretrained'
    )
    parser.add_argument(
        '--debug', default=False, action='store_true',
        help='run with debug mode'
    )

    return parser
