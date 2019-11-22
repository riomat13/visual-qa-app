#!/usr/bin/env python3
# -*- coding: utf-8 -*-


if __name__ == '__main__':
    import argparse
    import asyncio

    from main.settings import Config
    from main.models.server import run_server

    parser = argparse.ArgumentParser(description='Serve model')
    parser.add_argument('-H', '--host', type=str,
                        default=Config.MODEL_SERVER
                            .get('host', 'localhost'),
                        help='host machine to make connection')
    parser.add_argument('-p', '--port', type=int,
                        default=Config.MODEL_SERVER
                            .get('port', 12345),
                        help='port number to connect')

    args = parser.parse_args()
    host = args.host
    port = args.port

    asyncio.run(run_server(host, port))
