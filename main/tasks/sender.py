#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os.path
import contextlib

import paramiko

from main.settings import Config
from main.utils.logger import save_log
from main.models.base import AppLog

log = logging.getLogger(__name__)


def _sftp_connection_builder(server, username=None, password=None):
    username = username or Config.REMOTE_USERNAME
    password = password or Config.REMOTE_PASSWORD
    client = paramiko.SSHClient()
    # TODO: update host key
    client.set_missing_host_key_policy(paramiko.client.AutoAddPolicy())

    @contextlib.contextmanager
    def make_sftp_connection():
        nonlocal username
        nonlocal password
        nonlocal client
        try:
            client.connect(server, username=username, password=password)
            sftp = client.open_sftp()
            yield sftp
        finally:
            sftp.close()
            client.close()

    return make_sftp_connection


make_sftp_connection = _sftp_connection_builder(
    server=Config.REMOTE_SERVER,
    username=Config.REMOTE_USERNAME,
    password=Config.REMOTE_PASSWORD
)


@save_log(AppLog)
def send_data(filename, data_type):
    """Send data via SSH
    Data type has to be chosen from (`question`, `image`, `answer`, `log`)

    Args:
        filepath: str
            path to target data to send
        data_type: str
            choose from (`question`, `image`, `answer`, `log`)
    """
    data_type = data_type.lower()
    if data_type not in ('question', 'image', 'answer', 'log'):
        raise ValueError(f'Invalid data type: {data_type}')

    # make plural
    data_type = data_type + 's'

    filepath = os.path.join(Config.SEND_DATA_DIR, data_type, filename)

    with make_sftp_connection() as sftp:
        sftp.put(filepath, os.path.join(Config.REMOTE_DIR, data_type))


def send_dataset(filepaths, data_type):
    """Send data via SSH
    Data type has to be chosen from (`question`, `image`, `answer`, `log`)

    Args:
        filepath: str
            path to target data to send
        data_type: str
            choose from (`question`, `image`, `answer`)
    """
    with make_sftp_connection() as sftp:
        for filepath in filepaths:
            try:
                sftp.put(filepath, os.path.join(Config.REMOTE_DIR, data_type))
                log.info(f'Sent file: {os.path.basename(filepath)}')
            except Exception as e:
                log_ = AppLog(log_type='error',
                             log_class=e.__class__.__name__,
                             log_text=str(e))
                log_.save()
                log.error(str(e))
