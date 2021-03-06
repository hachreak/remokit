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
from remokit import dataset, utils, models


def get_names_only(filenames):
    return [os.path.basename(name) for name in filenames]


def attach_basepath(basepath, names):
    return [os.path.join(basepath, name) for name in names]


def attach_filetype(filetype, names):
    res = []
    for n in names:
        name, _ = os.path.splitext(n)
        res.append(name + filetype)
    return res


def _prepare_submodels(filenames, config, epochs, type_):
    """Prepare submodels."""
    filenames = get_names_only(filenames)
    output_shape = 0

    batches_list = []
    for subconf in config['submodels']:
        # get filenames for the submodel
        subtrain = deepcopy(filenames)
        subtrain = attach_basepath(
            subconf['directory'], subtrain
        )
        if 'files_types' in subconf:
            subtrain = attach_filetype(subconf['files_types'][0], subtrain)
        subtrain = dataset.epochs(subtrain, epochs=epochs)
        # load keras submodel
        submodel = models.load_model(subconf['model'])

        # compute the output shape
        (_, shape) = submodel.output_shape
        output_shape += shape

        # input shape if is not explicitly set
        if 'image_size' not in subconf:
            # input shape
            (_, img_x, img_y, _) = submodel.input_shape
            subconf['image_size'] = {'img_x': img_x, 'img_y': img_y}

        # copy global configuration
        subconf['batch_size'] = config['batch_size']

        # check if defined a personalized prediction function
        to_predict = dataset.to_predict
        if 'to_predict' in subconf:
            to_predict = utils.load_fun(subconf['to_predict'])(type_=type_)

        # build prediction batches
        prepare = utils.load_fun(subconf['prepare_batch'])
        batches, _shape = prepare(subtrain, subconf, epochs)
        batches = dataset.batch_adapt(batches, [
            dataset.apply_to_x(to_predict(model=submodel)),
        ])

        batches_list.append(batches)

    todo = dataset.flatten

    return dataset.merge_batches(
        batches_list, adapters=[
            dataset.apply_to_x(dataset.foreach(todo))
        ]
    ), output_shape


def prepare_batch(filenames, config, epochs, type_):
    """Prepare a batch."""
    filenames = list(dataset.epochs(filenames, epochs=epochs))

    batches, output_shape = _prepare_submodels(
        filenames, config, epochs, type_
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
