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

"""RGB Face CNN."""

from __future__ import absolute_import

from remokit import dataset, adapters, utils
from remokit.preprocessing import features


def prepare_batch(filenames, config, epochs, *args, **kwargs):
    """Prepare a batch."""
    shape = config['image_size']['img_x'], config['image_size']['img_y'], 1

    filenames = dataset.epochs(filenames, epochs=epochs)

    get_label = utils.load_fun(config['get_label'])
    stream = dataset.get_data(filenames, get_label)

    batches = dataset.stream_batch(stream, config['batch_size'])

    adapters_list = [
        dataset.apply_to_y(dataset.foreach(dataset.categorical)),
        dataset.apply_to_x(dataset.foreach(adapters.rgb_to_bn)),
        dataset.apply_to_x(dataset.foreach(
            features.get_face(config['image_size'])
        )),
        dataset.apply_to_x(dataset.foreach(adapters.astype('uint8'))),
        dataset.apply_to_x(adapters.matrix_to_bn),
        dataset.apply_to_x(adapters.normalize(255)),
    ]

    batches = dataset.batch_adapt(batches, adapters_list)

    return batches, shape
