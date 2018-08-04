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

"""RGB CNN

The input is a list of RGB images to process.

Every batch of images will be submitted to the following pipeline:
    - RGB to B/N matrix
    - Resize images to a matrix width x height (see config.json file)
    - Convert B/N matrix to B/N image (with 1 channel)
    - Normalize values to [0, 1]

Every batch of images will be training a CNN (see config.json file).
"""

from __future__ import absolute_import

from remokit import dataset, adapters
from remokit.utils import load_fun
from keras.preprocessing.image import ImageDataGenerator


def prepare_batch(filenames, config, epochs):
    """Prepare a batch."""
    shape = config['image_size']['img_x'], config['image_size']['img_y'], 1
    steps_per_epoch = len(filenames) // config['batch_size']

    filenames = dataset.epochs(filenames, epochs=epochs)

    stream = load_fun(config['get_data'])(filenames)

    get_labels = adapters.extract_labels()

    batches = dataset.stream_batch(stream, config['batch_size'])

    adapters_list = [
        dataset.apply_to_y(dataset.foreach(dataset.categorical)),
        get_labels,
        dataset.apply_to_x(dataset.foreach(adapters.rgb_to_bn)),
        dataset.apply_to_x(dataset.foreach(
            adapters.resize(**config['image_size'])
        )),
        dataset.apply_to_x(adapters.matrix_to_bn)
    ]

    if 'distortions' in config:
        adapters_list.append(
            adapters.apply_distortion(
                ImageDataGenerator(**config['distortions'])
            )
        )

    adapters_list.append(dataset.apply_to_x(adapters.normalize(255))),

    batches = dataset.batch_adapt(batches, adapters_list)

    return batches, steps_per_epoch, shape, config['epochs'], get_labels
