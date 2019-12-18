#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import logging

import numpy as np
from PIL import Image
from flask import current_app

from main.settings import Config
from main.utils.loader import load_image_simple

log = logging.getLogger(__name__)


def update_row_image(img_model, remove=False, *, send=False, upload=False):
    img_file = img_model.filename
    path = os.path.join(Config.UPLOAD_DIR, img_file)
    
    # resize image to (224, 224, 3)
    img = load_image_simple(path, normalize=False)

    # update data in database
    img_model.update()
    new_path = os.path.join(Config.UPLOAD_DIR, img_model.filename)
    save_image(img, new_path)

    if upload:
        # upload original data to outside of app
        upload_image(path)

    if send:
        # send original data via email
        # (only store in directory for sending)
        keep = not remove
        register_to_send_image(path, keep=keep)

    elif remove:
        # delete old data from app without send data
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


def register_to_send_image(path, keep=False, by='scp'):
    """Send image.
    This function is only to save a file to directory and
    other scheduled task do the job.

    Args:
        path: str
            path to the image to send
        keep: boolean
            if True, keep the file
        by: str
            the way to send data, e.g., 'scp'
            (currently only for scp, may add email)
    """
    # TODO: add email
    if by not in ('scp',):
        raise ValueError('Can send image via scp only')

    filename = os.path.basename(path)
    if keep:
        shutil.copy(path, f'data/tmp/images/{filename}')
    else:
        shutil.move(path, f'data/tmp/images/{filename}')


def upload_image(path):
    # TODO: implement own upload function
    pass


def delete_image(path):
    try:
        os.remove(path)
        log.info(f'Image file deleted: {path}')
    except FileNotFoundError as e:
        log.error(str(e))
