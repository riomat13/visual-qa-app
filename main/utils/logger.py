#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import warnings
from functools import wraps

from main.settings import Config
from main.mixins.models import ModelLogMixin, BaseMixin


def save_log(model):

    # check if model has required interface by inheritance
    if not issubclass(model, ModelLogMixin) and \
            not issubclass(model, BaseMixin):
        raise TypeError('Invalid type to log model. '
                        'Make sure the model is inherit from ModelLogMixin and BaseMixin.')

    def decorator(func):
        """Logging result to DB."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings(record=True) as w:
                try:
                    res = func(*args, **kwargs)
                    log = model(log_type='success',
                                log_text='success')
                    log.save()
                    return res
                except Exception as e:
                    log = model(log_type='error',
                                log_class=e.__class__.__name__,
                                log_text=str(e))
                    log.save()

                finally:
                    # whether success or not, save warnings
                    for warning in w:
                        text = '{}:{} {}'.format(warning.filename,
                                                 warning.lineno,
                                                 warning.message)
                        log = model(log_type='warning',
                                    log_class=warning.category.__name__,
                                    log_text=text)
                        log.save()
        return wrapper
    return decorator
