#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging

import numpy as np
from PIL import Image
from flask import current_app

from main.settings import Config
from main.utils.loader import load_image_simple

log = logging.getLogger(__name__)


def update_row_image(img_model, send=False, upload=False):
    img_file = img_model.filename
    path = os.path.join(Config.UPLOAD_DIR, img_file)
    
    # resize image to (224, 224, 3)
    img = load_image_simple(path)

    # update data in database
    img_model.update()
    new_path = os.path.join(Config.UPLOAD_DIR, img_model.filename)
    save_image(img, new_path)

    if upload:
        # upload original data to outside of app
        upload_image(path)

    if send:
        # send original data via email
        send_image(path)

    # delete old data from app
    delete_image(path)


def save_image(img_arr, path):
    """save processed array to file.
    
    Args:
        img_arr: numpy array
            represents image to save
        path: str
            path to save the image
    """
    im = Image.fromarray(img_arr.astype(np.uint8))
    im.save(path)
    log.info(f'Image file saved: {path}')


def send_image(path):
    # TODO: send by email
    pass


def upload_image(path):
    # TODO: implement own upload function
    pass


def delete_image(path):
    try:
        os.remove(path)
        log.info(f'Image file deleted: {path}')
    except FileNotFoundError as e:
        log.error(str(e))
