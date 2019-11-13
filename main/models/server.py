#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import logging

from main.utils.loader import load_image_simple as load_image
from main.orm.models.ml import ModelRequestLog
from main.models.infer import predict_question_type

log = logging.getLogger(__name__)


async def save_predict_history(filepath, sentence):
    # TODO: store results to DB
    pass


async def run_prediction(reader, writer):
    global q_type_model

    data = await reader.read(1024)
    filepath, sentence = data.decode().split('\t')

    # TODO: make active after build model
    #img = load_image(filepath)

    # TODO: add prediction pipeline
    pred = predict_question_type(sentence)
    writer.write(pred.encode())
    await writer.drain()
    writer.close()

    # save the result
    await save_predict_history(filepath, sentence)


async def run_server(host, port):
    server = await asyncio.start_server(
        run_prediction, host, port)
    addr = server.sockets[0].getsockname()
    log.info(f'Serving on {addr!r}')

    async with server:
        await server.serve_forever()
