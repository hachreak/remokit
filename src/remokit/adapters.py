# -*- coding: utf-8 -*-
#
# This file is part of remokit.
# Copyright 2018 Leonardo Rossi <leonardo.rossi@studenti.unipr.it>.
#
# pysenslog is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# pysenslog is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with pysenslog.  If not, see <http://www.gnu.org/licenses/>.

"""Matrix adapters."""

import cv2
import numpy as np
import dlib


def rgb_to_bn(features):
    """Adapt rgb image to input for the CNN as b/n image."""
    return np.array(
        [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in features]
    )


def matrix_to_bn(features):
    """Adapt matrix to input for the CNN as b/n image."""
    (img_x, img_y) = features[0].shape
    features = features.reshape(features.shape[0], img_x, img_y, 1)
    return features


def normalize(features):
    """Normalize features."""
    features = features.astype('float32')
    features /= 255
    return features


def resize(img_x, img_y):
    """Resize images."""
    def f(features):
        return np.array([
            dlib.resize_image(feature, img_x, img_y) for feature in features
        ])
    return f
