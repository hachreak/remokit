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

"""Experiment 01."""

from __future__ import absolute_import

#  import os
import json
import sys
#  import numpy as np
#  from copy import deepcopy
from remokit import dataset, adapters
from remokit.datasets import get_tvt_filenames
from remokit.utils import load_fun, set_reproducibility
from remokit.train import compile_, run
#  from keras.models import load_model
#  from sklearn.metrics import classification_report, confusion_matrix, \
#          accuracy_score


def prepare_batch(filenames, config):
    """Prepare a batch."""
    shape = config['image_size']['img_x'], config['image_size']['img_y'], 1
    steps_per_epoch = len(filenames) // config['batch_size']

    filenames = dataset.epochs(filenames, epochs=config['epochs'])

    stream = load_fun(config['get_data'])(filenames)
    stream = dataset.categorical(stream)

    get_labels = adapters.extract_labels()

    batches = dataset.stream_batch(stream, config['batch_size'])
    batches = dataset.batch_adapt(batches, [
        get_labels,
        adapters.rgb_to_bn,
        adapters.resize(**config['image_size']),
        adapters.matrix_to_bn,
        adapters.normalize
    ])

    return batches, steps_per_epoch, shape, config['epochs']


if len(sys.argv) < 2:
    msg = "Usage: {0} config.json [train|predict]"
    print(msg.format(sys.argv[0]))
    sys.exit(1)

# check if train or predict
is_training = sys.argv[2] if len(sys.argv) > 2 else 'train'
is_training = is_training == 'train'

# load config file
config_file = sys.argv[1]
with open(config_file) as data_file:
    config = json.load(data_file)

# init randomness
set_reproducibility(config['seed'])

# config
num_classes = len(dataset._category)

test, validating, training = get_tvt_filenames(
    config['kfold']['test'], config['kfold']['validation'],
    config['kfold']['k'], config['directory'],
    load_fun(config['get_label']), config['batch_size']
)

if is_training:
    batches, steps_per_epoch, shape, epochs = prepare_batch(training, config)
    validation_data, validation_steps, _, _ = prepare_batch(validating, config)
    model = load_fun(config['model'])(shape, num_classes)
    model = compile_(model)

    run(model, batches, steps_per_epoch, epochs,
        validation_data=validation_data, validation_steps=validation_steps)

    model.save(config['result'])

import ipdb; ipdb.set_trace()
print("ok")
