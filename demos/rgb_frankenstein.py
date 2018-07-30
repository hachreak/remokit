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

"""Train CNN model01 with colored image dataset using frankenstein models."""

from __future__ import absolute_import

import os
import json
import sys
from copy import deepcopy
from remokit import dataset, adapters
from remokit.models import get_conv_layers
from remokit.datasets import get_filenames
from remokit.utils import load_fun
from remokit.train import compile_, run
from keras.models import load_model


def get_names_only(filenames):
    return [os.path.basename(name) for name in filenames]


def attach_basepath(basepath, names):
    return [os.path.join(basepath, name) for name in names]


def get_training_filenames(config):
    directory = config['directory']
    model_index = 0
    get_label = load_fun(config['submodels'][model_index]['get_label'])
    index = config['kfold']['index']
    k = config['kfold']['k']
    batch_size = config['train']['batch_size']

    validating, training = get_filenames(
        index, k, directory, get_label, batch_size
    )

    return get_names_only(training)


def prepare_batch(config, filenames):
    batch_size = config['train']['batch_size']
    epochs = config['train']['epochs']

    batches_list = []
    output_shape = 0
    for submodel in config['submodels']:
        get_data = load_fun(submodel['get_data'])

        subtrain = deepcopy(filenames)
        subtrain = attach_basepath(
            submodel['directory'], subtrain
        )

        subtrain = dataset.epochs(subtrain, epochs=epochs)

        stream = get_data(subtrain)
        stream = dataset.categorical(stream)

        submodel = load_model(submodel['model'])
        conv = get_conv_layers(submodel)

        # output shape
        (_, shape) = conv.output_shape
        output_shape += shape
        # input shape
        (_, img_x, img_y, _) = conv.input_shape

        #  get_labels = adapters.extract_labels()

        batches = dataset.stream_batch(stream, batch_size)
        batches = dataset.batch_adapt(batches, [
            #  get_labels,
            adapters.rgb_to_bn,
            adapters.resize(img_x, img_y),
            adapters.matrix_to_bn,
            adapters.normalize,
            dataset.apply_to_x(dataset.to_predict(conv)),
        ])

        batches_list.append(batches)

    return dataset.merge_batches(batches_list), output_shape


# Load config

if len(sys.argv) < 2:
    msg = "Usage: {0} config.json"
    print(msg.format(sys.argv[0]))
    sys.exit(1)

# Start

config_file = sys.argv[1]
with open(config_file) as data_file:
    config = json.load(data_file)

filenames = get_training_filenames(config)

batch_size = config['train']['batch_size']
steps_per_epoch = len(filenames) // batch_size

batches, output_shape = prepare_batch(config, filenames)

num_classes = len(dataset._category)
epochs = config['train']['epochs']

# get CNN model
get_model = load_fun(config['model'])
model = get_model(output_shape, num_classes)
model = compile_(model)

run(model, batches, steps_per_epoch, epochs)

model.save(config['result'])
print("hello")
