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

"""RGB features CNN

The input is a list of RGB images to process.

Every batch of images will be submitted to the following pipeline:
    - RGB to B/N matrix
    - Resize images to a matrix width x height (see config.json file)
    - extract features
    - Normalize values to [0, 1]

Every batch of images will be training a NN (see config.json file).
"""

from __future__ import absolute_import

import numpy as np
from remokit import dataset, adapters
from remokit.utils import load_fun


def prepare_batch(filenames, config, epochs):
    """Prepare a batch."""
    shape = 68 * 2
    steps_per_epoch = len(filenames) // config['batch_size']
    norm_max = max(config['image_size'].values())

    filenames = dataset.epochs(filenames, epochs=epochs)

    get_label = load_fun(config['get_label'])
    stream = dataset.get_data(filenames, get_label, loader=np.load)

    get_labels = adapters.extract_labels()

    batches = dataset.stream_batch(stream, config['batch_size'])

    adapters_list = [
        dataset.apply_to_y(dataset.foreach(dataset.categorical)),
        get_labels,
        dataset.apply_to_x(dataset.foreach(dataset.flatten)),
        dataset.apply_to_x(adapters.normalize(norm_max)),
    ]

    batches = dataset.batch_adapt(batches, adapters_list)

    return batches, steps_per_epoch, shape, config['epochs'], get_labels
