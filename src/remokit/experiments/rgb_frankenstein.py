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

"""RGB CNN frankenstein

The input is a list of RGB images to process.

Every batch of images will be training a CNN (see config.json file).
"""

from __future__ import absolute_import

import os
from copy import deepcopy
from remokit import dataset, adapters, utils, models


def get_names_only(filenames):
    return [os.path.basename(name) for name in filenames]


def attach_basepath(basepath, names):
    return [os.path.join(basepath, name) for name in names]


def _prepare_submodels(filenames, config, epochs):
    """Prepare submodels."""
    filenames = get_names_only(filenames)
    merge_strategy = config.get('merge_strategy', 'flatten')
    output_shape = 0

    batches_list = []
    for subconf in config['submodels']:
        get_data = utils.load_fun(subconf['get_data'])
        # get filenames for the submodel
        subtrain = deepcopy(filenames)
        subtrain = attach_basepath(
            subconf['directory'], subtrain
        )
        subtrain = dataset.epochs(subtrain, epochs=epochs)
        # convert in a stream of images
        stream = get_data(subtrain)
        # load keras submodel
        submodel = models.load_model(subconf['model'])
        # with/without the classification layer
        if subconf.get('only_conv', False):
            submodel = models.get_conv_layers(submodel)
        # compute the output shape
        (_, shape) = submodel.output_shape
        #  if merge_strategy == 'flatten':
        output_shape += shape
        #  else:
        #      output_shape = shape

        # input shape
        (_, img_x, img_y, _) = submodel.input_shape

        # build prediction batches
        batches = dataset.stream_batch(stream, config['batch_size'])
        batches = dataset.batch_adapt(batches, [
            dataset.apply_to_y(dataset.foreach(dataset.categorical)),
            dataset.apply_to_x(dataset.foreach(adapters.rgb_to_bn)),
            dataset.apply_to_x(dataset.foreach(adapters.resize(img_x, img_y))),
            dataset.apply_to_x(adapters.matrix_to_bn),
            dataset.apply_to_x(adapters.normalize(255)),
            dataset.apply_to_x(dataset.to_predict(submodel)),
        ])

        batches_list.append(batches)

    if merge_strategy == 'flatten':
        todo = dataset.flatten
    #  elif merge_strategy == 'mean':
    #      todo = lambda x: np.mean(x, axis=0)
    #  else:
    #      todo = lambda x: np.max(x, axis=0)

    return dataset.merge_batches(
        batches_list, adapters=[
            dataset.apply_to_x(dataset.foreach(todo))
        ]
    ), output_shape


def prepare_batch(filenames, config, epochs):
    """Prepare a batch."""
    filenames = list(dataset.epochs(filenames, epochs=epochs))

    batches, output_shape = _prepare_submodels(
        filenames, config, epochs
    )

    return batches, output_shape


def get_files(submodels, directory, *args, **kwargs):
    """Return list of files availables for all submodels."""
    filenames = []
    for subconf in submodels:
        filenames.extend(
            get_names_only(utils.load_fun(subconf['get_files'])(
                subconf['directory'])
            )
        )

    filenames = list(set(filenames))
    exists = [filenames]
    for subconf in submodels:
        exists.append([
            os.path.exists(f)
            for f in attach_basepath(subconf['directory'], filenames)
        ])

    for i, f in enumerate(exists[0]):
        if all([exists[j][i] for j in range(1, len(exists))]):
            yield os.path.join(directory, f)
