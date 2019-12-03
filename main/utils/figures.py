#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os.path

import numpy as np
import matplotlib.pyplot as plt

from main.settings import Config
from main.utils.loader import load_image_simple
from main.orm.models.ml import RequestLog
from main.orm.models.data import WeightFigure

log = logging.getLogger(__name__)


def save_figure(fig, log=None):
    """Save figure to file and store the info to DB.
    Args:
        fig: matplotlib.figure.Figure
            figure object to save
        log: sqlalchemy DeclativeMeta (optional)
            correspond RequestLog model
    Return:
        None
    """
    base = Config.FIG_DIR

    # automatically generate new file name
    w = WeightFigure(log=log)
    w.save()

    filename = w.filename

    path = os.path.join(base, filename)
    fig.savefig(path)


def generate_heatmap(arr, inputs, outputs):
    """Generate Heatmap from numpy.ndarray.

    x-axis will be question.
    y-axis will be answer.

    Args:
        arr: 2d numpy.ndarray
        inputs: list of str
            question sentence
        outputs: list of str
            answer sentence

    Returns:
        fig: matplotlib.figure.Figure
        axes: matplotlib.axes._subplots.AxesSubplot
    """
    out_, in_ = arr.shape

    fig, ax = plt.subplots()
    im = ax.imshow(arr)

    ax.set_xticks(np.arange(in_))
    ax.set_yticks(np.arange(out_))

    ax.set_xticklabels(inputs)
    ax.set_yticklabels(outputs)

    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    for i in range(out_):
        for j in range(in_):
            text = ax.text(j, i, format(arr[i, j], '.2f'),
                           ha='center', va='center', color='w')

    fig.tight_layout()
    return fig, ax
