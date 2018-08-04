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


def rgb_to_bn(matrix):
    """Adapt rgb image to input for the CNN as b/n image."""
    return cv2.cvtColor(matrix, cv2.COLOR_RGB2GRAY)


def matrix_to_bn(batch_x):
    """Adapt matrix to input for the CNN as b/n image."""
    (img_x, img_y) = batch_x[0].shape
    return batch_x.reshape(batch_x.shape[0], img_x, img_y, 1)


def astype(name):
    """Convert matrix to this new type."""
    def f(matrix):
        return matrix.astype(name)
    return f


def normalize(max_):
    """Normalize matrix."""
    def f(matrix):
        matrix = matrix.astype('float32')
        matrix /= max_
        return matrix
    return f


def resize(img_x, img_y):
    """Resize image."""
    def f(img):
        # get proportional width x height
        wx, wy = _better_proportion(img_x, img_y, img.shape[0], img.shape[1])
        # resize proportionally and add black padding
        return padding(img_x, img_y, dlib.resize_image(img, wx, wy))
    return f


def _better_proportion(wx, wy, ix, iy):
    y = _get_proportion(wx, ix, iy)
    if y <= wy:
        return wx, y
    return _get_proportion(wy, iy, ix), wy


def _get_proportion(wx, ix, iy):
    return int(wx * iy / ix)


def padding(img_x, img_y, img):
    """Add padding."""
    shape = list(img.shape)
    delta_rows = img_y - img.shape[1]
    if delta_rows > 0:
        shape[1] = delta_rows
        pad_rows = np.zeros(shape)
        img = np.hstack((img, pad_rows))
        return img
    else:
        delta_cols = img_x - img.shape[0]
        shape[0] = delta_cols
        pad_cols = np.zeros(shape)
        img = np.vstack((img, pad_cols))
        return img


class extract_labels(object):
    """Extract processed labels."""

    def __init__(self):
        self.labels = []

    def __call__(self, batch, *args, **kwargs):
        (x, y) = batch
        self.labels.extend(y)
        return batch


def apply_distortion(datagen):
    """Apply a random distortion to the input."""
    def f(batch):
        (x, y) = batch
        x = np.array([
            datagen.apply_transform(
                img, datagen.get_random_transform(img.shape)
            ) for img in x
        ])
        return x, y
    return f
