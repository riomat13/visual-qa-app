#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import asyncio
import logging

from main.utils.preprocess import text_processor
from main.utils.logger import save_log
from main.orm.models.ml import PredictionScore, RequestLog
from main.models.infer import predict_question_type, PredictionModel

log = logging.getLogger(__name__)

_processor = None
predictor = PredictionModel.get_model()


async def save_predict_history(filepath, sentence, pred, log_model):
    # TODO: store results to DB
    filename = os.path.basename(filepath)
    score = PredictionScore(filename=filename,
                            question=sentence,
                            prediction=pred,
                            log=log_model)
    score.save()


@save_log(RequestLog)
async def run_prediction(reader, writer):
    global predictor
    global _processor

    data = await reader.read(1024)
    filepath, sentence = data.decode().split('\t')
    log.debug(sentence)

    try:
        if not os.path.isfile(filepath):
            log.warning('Could not find image')
            pred = 'Could not find image.'
        else:
            pred, _ = predictor.predict(sentence, filepath)
    except Exception as e:
        log.error(e)
        # send error code
        writer.write(b'<e>')
        raise
    else:
        pred = pred.encode()
        writer.write(pred)
    finally:
        await writer.drain()
        writer.close()

    filename = os.path.basename(filepath)
    log_model = RequestLog.query().filter_by(filename=filename).first()
    if log_model is not None:
        # save the result
        await save_predict_history(filepath,
                                   sentence,
                                   pred,
                                   log_model)
    else:
        log.error('Could not store predict history')


async def run_server(host, port):
    global _processor

    _processor = text_processor(num_words=15, from_config=True)

    server = await asyncio.start_server(
        run_prediction, host, port)
    addr = server.sockets[0].getsockname()
    log.info(f'Serving on {addr!r}')

    async with server:
        await server.serve_forever()
