#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import asyncio
from contextlib import asynccontextmanager

from main.settings import Config


@asynccontextmanager
async def _get_connection(host=None, port=None):
    """Make connection and close the socket properly."""
    if host is None:
        host = Config.MODEL_SERVER.get('host', 'localhost')
    if port is None:
        port = Config.MODEL_SERVER.get('port', 12345)

    addr = (await asyncio.get_running_loop()
        .getaddrinfo(host, port, proto=socket.IPPROTO_TCP))[-1]

    if not addr:
        raise ValueError('Could not resolve address')

    reader, writer = await asyncio.open_connection(host, port)

    try:
        yield reader, writer
    finally:
        writer.close()
        # wait until properly closed the connection
        await asyncio.shield(writer.wait_closed())


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
    msg = filepath + '\t' + sentence

    async with _get_connection() as (reader, writer):
        writer.write(msg.encode())
        await writer.drain()
        pred = await reader.read(1024)

    pred, fig_id = pred.decode().strip().split('\t')

    if pred == '<e>':
        pred = 'Sorry, some problem occured'

    return pred, int(fig_id)
