#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import logging

from main.utils.loader import load_image_simple as load_image
#from main.models.questions import QType

log = logging.getLogger(__name__)

# build model for serving
#q_type_model = QType(serve=True)
q_type_model = lambda x: x
predict = lambda x: 'test result'
q_type_model.predict = predict


async def run_prediction(reader, writer):
    global q_type_model

    data = await reader.read(1024)
    filepath, sentence = data.decode().split('\t')

    img = load_image(filepath)

    # TODO: add prediction pipeline
    pred = q_type_model.predict(img)
    writer.write(pred.encode())
    await writer.drain()
    writer.close()


async def run_server(host, port):
    server = await asyncio.start_server(
        run_prediction, host, port)
    addr = server.sockets[0].getsockname()
    log.info(f'Serving on {addr!r}')

    async with server:
        await server.serve_forever()
