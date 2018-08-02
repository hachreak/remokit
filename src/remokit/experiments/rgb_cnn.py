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

import numpy as np
from remokit import dataset, adapters
from remokit.utils import load_fun
from remokit.train import compile_, run
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, \
    accuracy_score, precision_recall_fscore_support


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

    if 'result' in config:
        model.save(config['result'])

    return model


def predict(testing, config, **kwargs):
    batches, steps_per_epoch, shape, epochs, get_labels = prepare_batch(
        testing, config, 1
    )

    if 'result' in config:
        model = load_model(config['result'])
    else:
        model = kwargs['model']

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

    # get average metrics
    (prec, recall, fscore, _) = precision_recall_fscore_support(
        y_val, y_pred, average='weighted'
    )
    report = {
        'average': {
            'precision': prec,
            'recall': recall,
            'f1-score': fscore,
        }
    }
    # get per class metrics
    metrics = np.array(
        precision_recall_fscore_support(y_val, y_pred)
    ).transpose()
    report['classes'] = {}
    for i, m in enumerate(metrics):
        report['classes'][dataset.category2label(i)] = {
            'precision': m[0],
            'recall': m[1],
            'f1-score': m[2],
            'support': m[3],
        }

    print("Accuracy")
    accuracy = accuracy_score(y_val, y_pred)
    print(accuracy)

    return matrix, report, accuracy


def evaluate(testing, config, **kwargs):
    """Evaluate."""
    batches, steps_per_epoch, shape, epochs, get_labels = prepare_batch(
        testing, config, 1
    )

    if 'result' in config:
        model = load_model(config['result'])
    else:
        model = kwargs['model']

    metrics = model.evaluate_generator(batches, steps=steps_per_epoch)

    result = {}
    for (i, j) in zip(model.metrics_names, metrics):
        result[i] = j
    return result