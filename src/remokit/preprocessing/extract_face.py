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

from remokit import dataset as ds, adapters, utils, detect
from remokit.preprocessing import features


def prepare_batch(config):
    """Extract faces from the dataset."""
    gl = utils.load_fun(config['get_label'])

    stream = utils.load_fun(config['get_files'])(**config)
    stream = ds.stream(ds.add_label(gl), stream)
    stream = ds.stream(ds.apply_to_x(detect.load_img), stream)
    stream = ds.stream(
        ds.apply_to_x(adapters.resize(**config['full_image_size'])), stream
    )

    batches = ds.stream_batch(stream, config['batch_size'])

    adapters_list = [
        ds.apply_to_x(ds.foreach(adapters.astype('uint8')))
    ]

    if config['has_faces']:
        adapters_list.append(
            ds.apply_to_x(ds.foreach(features.get_face()))
        )

    adapters_list.extend([
        ds.apply_to_x(ds.foreach(adapters.resize(**config['image_size']))),
        ds.apply_to_x(ds.foreach(adapters.astype('uint8')))
    ])

    batches = ds.batch_adapt(batches, adapters_list)

    return batches
