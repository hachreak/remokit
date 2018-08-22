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

"""Experiment utils."""

import os
import numpy as np
from remokit.train import compile_, run, default
from remokit import dataset
from remokit.utils import load_fun, set_seed
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, \
    accuracy_score, precision_recall_fscore_support
from remokit.datasets import get_tvt_filenames


def train(training, validating, config, prepare_batch, **kwargs):
    batches, shape, _ = prepare_batch(
        training, config, config['epochs']
    )
    steps_per_epoch = len(training) // config['batch_size']

    validation_data, _, _ = prepare_batch(
        validating, config, config['epochs']
    )
    validation_steps = len(validating) // config['batch_size']

    # FIXME is always calling next() one time more
    validation_steps -= 1

    training = config.get('training', default())
    num_classes = len(dataset._category)
    model = load_fun(config['model'])(shape, num_classes)
    model = compile_(model, config=training)

    run(
        model, batches, steps_per_epoch, config['epochs'],
        validation_data=validation_data, validation_steps=validation_steps,
        config=training, **kwargs
    )

    if 'result' in config:
        model.save(config['result'])

    return model


def predict(testing, config, prepare_batch, **kwargs):
    batches, shape, get_labels = prepare_batch(
        testing, config, 1
    )
    steps_per_epoch = len(testing) // config['batch_size']

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


def evaluate(testing, config, prepare_batch, **kwargs):
    """Evaluate."""
    batches, shape, get_labels = prepare_batch(
        testing, config, 1
    )
    steps_per_epoch = len(testing) // config['batch_size']

    if 'result' in config:
        model = load_model(config['result'])
    else:
        model = kwargs['model']

    metrics = model.evaluate_generator(batches, steps=steps_per_epoch)

    result = {}
    for (i, j) in zip(model.metrics_names, metrics):
        result[i] = j
    return result


class save_best(object):
    """Save best model."""

    def __init__(self, config):
        self._config = config
        self._last_accuracy = 0
        if 'best_model' in self._config:
            destination, _ = os.path.split(self._config['best_model'])
            if not os.path.isdir(destination):
                os.makedirs(destination)

    def __call__(self, metrics, model):
        """Save if better."""
        if metrics['acc'] > self._last_accuracy:
            self._last_accuracy = metrics['acc']
            if 'best_model' in self._config:
                model.save(self._config['best_model'])


def run_experiment(test_index, validation_index, config):
    """Run a single experiment."""
    print("seed {0}".format(config['seed']))
    print("kfold: test [{0}]  validation [{1}]".format(
        test_index, validation_index
    ))

    set_seed(config['seed'])

    prepare_batch = load_fun(config['prepare_batch'])

    if 'get_files' in config:
        filenames = load_fun(config['get_files'])(**config)
    else:
        filenames = dataset.get_files(config['directory'],
                                      types=config.get('files_types'))
    # split filenames in groups
    testing, validating, training = get_tvt_filenames(
        test_index, validation_index,
        config['kfolds'], filenames,
        load_fun(config['get_label']), config['batch_size']
    )
    # run training
    model = train(
        training, validating, config, prepare_batch, verbose=config['verbose']
    )
    # get metrics
    m = evaluate(testing, config, prepare_batch, model=model)

    m['history'] = model.history.history

    m['kfolds'] = {
        'k': config['kfolds'],
        'testing': test_index,
        'validation': validation_index
    }

    # get more metrics running predict
    matrix, report, accuracy = predict(
        testing, config, prepare_batch, model=model
    )

    m['confusion_matrix'] = matrix
    m['report'] = report

    m['seed'] = config['seed']

    m['files'] = {
        'testing': testing,
        'validating': validating
    }
    return m, model
