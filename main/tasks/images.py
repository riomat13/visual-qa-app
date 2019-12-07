#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from . import celery
from main.orm.models.data import Image
from main.utils.images import update_row_image

log = logging.getLogger(__name__)


@celery.task()
def image_process_task(ids=[]):
    """Process original image and return processed image ids.
    
    Args:
        ids: list or tuple of integer
            Image ids to apply the task
            if not provided, apply to all unprocessed data
    """
    res = []

    # TODO: remove files after process
    if not ids:
        for img_model in Image.query().filter_by(processed=False).all():
            update_row_image(img_model, remove=False)
            res.append(img_model.id)
    else:
        for id_ in ids:
            img_model = Image.get(id_)
            if img_model.processed:
                log.warning(f'ID: {id_} is already processed')
                continue
            update_row_image(img_model, remove=False)
            res.append(img_model.id)
    return res
