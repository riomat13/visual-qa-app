#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio

from main.settings import Config


async def run_model(filepath: str, sentence: str):
    """Send request to predicto model and return
    the received result.

    Args:
        filepath: str
            path to an image data to predict
        sentence: str
    Return:
        str: predicted result

    Example:
        >>> asyncio.run(run_model('/path/to/img', 'what is this?')
        'some answer'
    """
    host = Config.MODEL_SERVER.get('host', 'localhost')
    port = Config.MODEL_SERVER.get('port', 12345)
    reader, writer = await asyncio.open_connection(host, port)

    msg = filepath + '\t' + sentence

    writer.write(msg.encode())

    pred = await reader.read(1024)
    pred = pred.decode()
    writer.close()

    return pred
