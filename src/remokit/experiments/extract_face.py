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
from remokit import dataset as ds, adapters, utils, detect
from remokit.preprocessing import features


def merge(config_list):
    """Merge different datasets in a single preprocessed dataset."""
    indices = None
    for config in config_list:
        stream = prepare_batch(config)
        indices = save(stream, config, indices)
    return indices


def prepare_batch(config):
    """Extract faces from the dataset."""
    gl = utils.load_fun(config['get_label'])

    stream = utils.load_fun(config['get_files'])(config['directory'])
    stream = ds.stream(ds.add_label(gl), stream)
    stream = ds.stream(ds.apply_to_x(detect.load_img), stream)
    stream = ds.stream(ds.apply_to_x(adapters.astype('uint8')), stream)

    if config['has_faces']:
        stream = ds.stream(ds.apply_to_x(features.get_face()), stream)

    stream = ds.stream(ds.apply_to_x(adapters.rgb_to_bn), stream)
    stream = ds.stream(
        ds.apply_to_x(adapters.resize(**config['image_size'])), stream
    )
    stream = ds.stream(ds.apply_to_x(adapters.astype('uint8')), stream)

    return stream


def save(batches, config, indices=None):
    """Save preprocessed images."""
    indices = indices or {v: 0 for v in ds._category.keys() + ['index']}
    print(indices)
    for X, y in batches:
        if y == 'neutral':
            indices['index'] += 1
        indices[y] += 1
        filename = '{0:08}_{1}_{2}.jpg'.format(indices['index'], y, indices[y])
        destination = os.path.join(config['destination'], filename)
        dlib.save_image(X, destination)
        print(destination)
    return indices


def get_data(files_stream):
    """Get a streaming of label/image to process."""
    for filename in files_stream:
        yield detect.load_img(filename), get_label(filename)


def get_label(filename):
    return os.path.split(filename)[1].split('_')[1]
