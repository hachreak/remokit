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

"""Preprocessing: extract face from images."""

import os
import dlib
from remokit import dataset as ds, adapters, datasets
from remokit.preprocessing import features


def prepare_batch(directory, dataset, config):
    """Extract faces from the dataset."""
    stream = dataset.get_files(directory)

    stream = ds.stream(ds.apply_to_x(adapters.astype('uint8')), stream)
    stream = ds.stream(ds.apply_to_x(features.get_face()), stream)
    stream = ds.stream(
        ds.apply_to_x(adapters.resize(**config['big_image_size'])), stream
    )

    return stream


def save(batches, config):
    """Save preprocessed images."""
    indices = {v: 0 for v in ds._category.keys()}
    index = 0
    for (X, y) in batches:
        if y == 'neutral':
            index += 1
        indices[y] += 1
        filename = '{0}_{1}_{2}.jpg'.format(y, index, indices[y])
        destination = os.path.join(config['destination'], filename)
        dlib.save_image(X.astype('uint8'), destination)
        print(destination)


def get_label(filename):
    return os.path.split(filename)[1].split('_')[0]
