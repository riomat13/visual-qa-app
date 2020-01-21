#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import asyncio
import logging

from main.utils.preprocess import text_processor
from main.utils.logger import save_log
#from main.utils.figures import generate_heatmap, save_figure
from main.models.ml import PredictionScore, RequestLog
from main.models.data import Image, Question
from main.ml.infer import predict_question_type, PredictionModel

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
    log.info(f'Run prediction: {sentence}')

    fig_id = None

    try:
        if not os.path.isfile(filepath):
            log.warning('Could not find image')
            pred = 'Could not find image.\t0'
            raise FileNotFoundError(f'Could not find image: {filepath}')
        else:
            pred, w, qtype_id = predictor.predict(sentence, filepath)
            # TODO: add attention weight for Closed-Question
            if qtype_id < 1:
                fig_id = 0
            else:
                # generate attention weighted heatmap
                sentence_ = sentence.split()
                pred_ = pred.split()
                # weights = w[0, :len(pred_)]
                # f, a = generate_heatmap(weights, sentence_, pred_)
                # fig_id = save_figure(f)
                fig_id = 0
            pred_out = pred + '\t' + str(fig_id)

    except Exception as e:
        log.error(e)
        # send error code
        pred_out = '<e>\t0'
        log.error('Could not store predict history')
        kwargs = {
            'log_text': str(e),
            'log_type': 'error'
        }
        raise
    else:
        kwargs = {
            'question_type_id': qtype_id,
            'log_text': 'success',
            'log_type': 'success'
        }

    finally:
        # save the result
        filename = os.path.basename(filepath)
        img = Image.query().filter_by(filename=filename).first()
        pred_id = None
        q = Question(question=sentence)
        q.save()

        if kwargs.get('log_type') == 'success':
            pred_log = PredictionScore(prediction=pred)
            pred_log.save()
            pred_id = pred_log.id

        log_model = RequestLog(image_id=img.id,
                               question_id=q.id,
                               fig_id=fig_id,
                               score_id=pred_id,
                               **kwargs)
        log_model.save()

        writer.write(pred_out.encode())
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
