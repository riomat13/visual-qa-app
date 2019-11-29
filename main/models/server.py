#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import asyncio
import logging

from main.utils.preprocess import text_processor
from main.utils.logger import save_log
from main.orm.models.ml import PredictionScore, RequestLog
from main.orm.models.data import Image, Question
from main.models.infer import predict_question_type, PredictionModel

log = logging.getLogger(__name__)

_processor = None
predictor = PredictionModel.get_model()


async def save_predict_history(filepath, sentence, pred, log_model):
    # TODO: store results to DB
    raise NotImplementedError


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
            pred, _, qtype_id = predictor.predict(sentence, filepath)
    except Exception as e:
        log.error(e)
        # send error code
        writer.write(b'<e>')
        log.error('Could not store predict history')
        kwargs = {
            'log_text': str(e),
            'log_type': 'error'
        }
        raise
    else:
        writer.write(pred.encode())
        kwargs = {
            'question_type_id': qtype_id,
            'log_text': 'success',
            'log_type': 'success'
        }

    finally:
        # save the result
        filename = os.path.basename(filepath)
        img = Image(filename=filename)
        img.save()
        q = Question(question=sentence)
        q.save()
        log_model = RequestLog(image_id=img.id,
                               question_id=q.id,
                               **kwargs)
        log_model.save()
        if kwargs.get('log_type') == 'success':
            pred_log = PredictionScore(prediction=pred,
                                       log_id=log_model.id)
            pred_log.save()
        await writer.drain()
        writer.close()


async def run_server(host, port):
    global _processor

    _processor = text_processor(num_words=15, from_config=True)

    server = await asyncio.start_server(
        run_prediction, host, port)
    addr = server.sockets[0].getsockname()
    log.info(f'Serving on {addr!r}')

    async with server:
        await server.serve_forever()
