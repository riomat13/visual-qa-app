#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from main.web.app import create_app

parser = argparse.ArgumentParser(description='Running web server')
parser.add_argument('-p', '--port', type=int, default=5000,
                    help='port')
parser.add_argument('-c', '--config', type=str, default='development',
                    help='set configuration to run web server')


def main(config):
    from main.web.app import create_app
    
    if config not in ('production', 'local', 'development', 'test'):
        config = 'developement'
    app = create_app(config)

    return app


if __name__ == '__main__':
    args = parser.parse_args()

    from main.settings import set_config
    config = args.config
    set_config(config)

    app = main(config)
    port = args.port
    app.run('127.0.0.1', port)
