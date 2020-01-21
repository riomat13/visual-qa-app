#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from celery import Celery
from celery.schedules import crontab

from main.web.app import create_app
from main.models.data import Image


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

    # assume data size is small
    celery.conf.beat_schedule = {
        'daily-process-image': {
            'task': 'main.tasks.images.image_process_task',
            'schedule': crontab(hour='*/24'),
        }
    }


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
