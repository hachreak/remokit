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

from copy import deepcopy

import os
import dlib

from remokit.utils import load_fun, recreate_directory
from remokit import dataset as ds, detect


def preprocess(config):
    """Preprocess images."""
    # create stage directory
    recreate_directory(config['directory'])

    # get personalized batch_size
    batch_size = config['batch_size']
    if 'preprocess_batch_size' in config:
        batch_size = config['preprocess_batch_size']

    # collect configurations
    config_list = []
    for prep in config['preprocess']:
        c = deepcopy(prep)
        c['destination'] = config['directory']
        c['full_image_size'] = deepcopy(config['full_image_size'])
        c['image_size'] = deepcopy(config['image_size'])
        c['batch_size'] = batch_size
        config_list.append(c)

    # merge datasets
    indices = None
    for config in config_list:
        prepare_batch = load_fun(config['prepare_batch'])
        save = load_fun(config['save'])
        stream = prepare_batch(config)
        indices = save(stream, config, indices)
    print(indices)


def save(batches, config, indices=None):
    """Save preprocessed images."""
    indices = indices or {v: 0 for v in ds._category.keys() + ['index']}
    print(indices)
    for Xbatch, ybatch in batches:
        for X, y in zip(Xbatch, ybatch):
            if y == 'neutral':
                indices['index'] += 1
            indices[y] += 1
            filename = '{0:08}_{1}_{2}.jpg'.format(
                indices['index'], y, indices[y]
            )
            destination = os.path.join(config['destination'], filename)
            dlib.save_image(X, destination)
            print(destination)
    return indices


def get_files(directory, *args, **kwargs):
    """Get image/label files."""
    return ds.get_files(directory)


def get_data(files_stream):
    """Get a streaming of label/image to process."""
    for filename in files_stream:
        yield detect.load_img(filename), get_label(filename)


def get_label(filename):
    return os.path.split(filename)[1].split('_')[1]
