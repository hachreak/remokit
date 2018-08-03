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


def rgb_to_bn(batch):
    """Adapt rgb image to input for the CNN as b/n image."""
    (x, y) = batch
    x = np.array(
        [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in x]
    )
    return x, y


def matrix_to_bn(batch):
    """Adapt matrix to input for the CNN as b/n image."""
    (x, y) = batch
    (img_x, img_y) = x[0].shape
    x = x.reshape(x.shape[0], img_x, img_y, 1)
    return x, y


def normalize(batch):
    """Normalize batch."""
    (x, y) = batch
    x = x.astype('float32')
    x /= 255
    return x, y


def resize(img_x, img_y):
    """Resize images."""
    def f(batch):
        (x, y) = batch
        images = []
        for img in x:
            # get proportional width x height
            wx, wy = _better_proportion(
                img_x, img_y, img.shape[0], img.shape[1]
            )
            # resize proportionally and add black padding
            images.append(
                padding(img_x, img_y, dlib.resize_image(img, wx, wy))
            )
        return np.array(images), y
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
    delta_rows = img_y - img.shape[1]
    if delta_rows > 0:
        pad_rows = np.zeros((img.shape[0], delta_rows))
        img = np.hstack((img, pad_rows))
        return img
    else:
        delta_cols = img_x - img.shape[0]
        pad_cols = np.zeros((delta_cols, img.shape[1]))
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
