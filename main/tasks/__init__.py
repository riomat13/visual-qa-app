#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from celery import Celery

from main.web.app import create_app
from main.orm.models.data import Image


def make_celery(app):
    """Build Celery instance with app context.
    
    Args:
        app: flask app
    
    Return:
        Celery object
    """
    # add app context to celery task
    celery = Celery(
        app.name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


# TODO: replace app type with production
app = create_app('test')
celery = make_celery(app)

from .images import image_process_task


if __name__ == '__main__':
    celery.start()
