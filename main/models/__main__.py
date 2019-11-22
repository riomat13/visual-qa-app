#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

if __name__ == '__main__':
    import argparse
    import asyncio

    from main.models.server import run_server

    parser = argparse.ArgumentParser(description='Serve model')
    parser.add_argument('-H', '--host', type=str, default='',
                        help='host machine to make connection')
    parser.add_argument('-p', '--port', type=int, default=0,
                        help='port number to connect')
    parser.add_argument('-c', '--config', type=str,
                        default='local',
                        help='Config type: production, '
                             'local, development, test')

    args = parser.parse_args()
    host = args.host
    port = args.port

    from main.settings import set_config
    config = args.config
    # TODO: add production
    if config not in ('local', 'development', 'test'):
        log.warning(f'Invalid config: {config} "local" config will be used')
        config = 'local'
    set_config(args.config)

    from main.settings import Config

    # if host and port is not set, use from config
    if not host:
        host = Config.MODEL_SERVER.get('host', 'localhost')
    if not port:
        port = Config.MODEL_SERVER.get('port', 12345)

    asyncio.run(run_server(host, port))
