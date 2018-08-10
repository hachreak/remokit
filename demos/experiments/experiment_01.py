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

"""Experiment 01.

RGB images as input of a CNN with a classifier for the emotions.
"""

from __future__ import absolute_import

from copy import deepcopy
from sys import argv
from remokit.experiments import train, evaluate, predict
from remokit.datasets import get_tvt_filenames
from remokit.dataset import permute_index_kfold
from remokit.utils import load_fun, set_seed, clean_session, load_config
from remokit.metrics import save_metrics
from remokit.experiments.extract_face import merge


def run_experiment(test_index, validation_index, config):
    """Run a single experiment."""
    print("seed {0}".format(config['seed']))
    print("kfold: test [{0}]  validation [{1}]".format(
        test_index, validation_index
    ))

    set_seed(config['seed'])

    prepare_batch = load_fun(config['prepare_batch'])

    # split filenames in groups
    testing, validating, training = get_tvt_filenames(
        test_index, validation_index,
        config['kfolds'], config['directory'],
        load_fun(config['get_label']), config['batch_size']
    )
    # run training
    model = train(
        training, validating, config, prepare_batch, verbose=config['verbose']
    )
    # get metrics
    m = evaluate(testing, config, prepare_batch, model=model)

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

    return m


def run_all(config):
    k = config['kfolds']
    metrics = []
    for myseed in range(0, config['repeat_seeds']):
        config['seed'] = myseed
        for test_index, validation_index in permute_index_kfold(k):
            m = run_experiment(test_index, validation_index, config)
            metrics.append(m)
            save_metrics(metrics, config['metrics'])
            clean_session()


def preprocess(config):
    """Preprocess images."""
    # collect configurations
    config_list = []
    for prep in config['preprocess']:
        c = deepcopy(prep)
        c['destination'] = config['directory']
        c['image_size'] = deepcopy(config['image_size'])
        config_list.append(c)
    # merge datasets
    indices = merge(config_list)
    print(indices)


def main(args):
    config = load_config(args[2])

    if args[1] == 'preprocess':
        preprocess(config)
    else:
        run_all(config)


main(deepcopy(argv))
