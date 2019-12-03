#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch

from main.utils.figures import save_figure


class BuildHeatMapTest(unittest.TestCase):

    @patch('main.utils.figures.WeightFigure')
    @patch('matplotlib.figure.Figure')
    def test_save_figure(self, Figure, mock_weights):
        mock_save = mock_weights.return_value.save
        mock_save.return_value = None
        mock_weights.return_value.filename = 'test'

        fig = Figure()
        save_figure(fig)
        mock_save.assert_called_once()
