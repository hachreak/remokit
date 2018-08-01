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
from remokit.datasets import get_tvt_filenames
from remokit.utils import load_fun, set_reproducibility
from remokit.train import compile_, run
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, \
    accuracy_score


def prepare_batch(filenames, config, epochs):
    """Prepare a batch."""
    shape = config['image_size']['img_x'], config['image_size']['img_y'], 1
    steps_per_epoch = len(filenames) // config['batch_size']

    filenames = dataset.epochs(filenames, epochs=epochs)

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

    return batches, steps_per_epoch, shape, config['epochs'], get_labels


def train(training, validating, config):
    batches, steps_per_epoch, shape, epochs, _ = prepare_batch(
        training, config, config['epochs']
    )
    validation_data, validation_steps, _, _, _ = prepare_batch(
        validating, config, config['epochs']
    )
    # FIXME is always calling next() one time more
    validation_steps -= 1

    num_classes = len(dataset._category)
    model = load_fun(config['model'])(shape, num_classes)
    model = compile_(model)

    run(model, batches, steps_per_epoch, epochs,
        validation_data=validation_data, validation_steps=validation_steps)

    model.save(config['result'])


def predict(testing, config):
    batches, steps_per_epoch, shape, epochs, get_labels = prepare_batch(
        testing, config, 1
    )

    model = load_model(config['result'])

    y_pred = model.predict_generator(batches, steps=steps_per_epoch)

    y_val = dataset.list_apply(dataset.categorical2category, get_labels.labels)
    y_pred = dataset.list_apply(dataset.categorical2category, y_pred)

    ordered_labels = dataset.ordered_categories()

    print('Confusion Matrix')

    matrix = confusion_matrix(y_val, y_pred)
    for i, row in enumerate(matrix):
        to_print = ''.join(['{:4}'.format(item) for item in row])
        print("{0:<15} {1}".format(ordered_labels[i], to_print))

    report = classification_report(y_val, y_pred, target_names=ordered_labels)
    print(report)

    print("Accuracy")
    print(accuracy_score(y_val, y_pred))


def experiment(config):
    """Run experiment."""
    # init randomness
    set_reproducibility(config['seed'])

    testing, validating, training = get_tvt_filenames(
        config['kfold']['test'], config['kfold']['validation'],
        config['kfold']['k'], config['directory'],
        load_fun(config['get_label']), config['batch_size']
    )

    if config['is_training']:
        train(training, validating, config)
    else:
        predict(testing, config)
