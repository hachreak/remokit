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

import sys
from copy import deepcopy
from remokit.experiments import train, evaluate, predict
from remokit.experiments.rgb_cnn import prepare_batch
from remokit.datasets import get_tvt_filenames
from remokit.dataset import permute_index_kfold
from remokit.utils import load_fun, set_seed, clean_session, load_config
from remokit.metrics import save_metrics


def run_experiment(test_index, validation_index, config):
    """Run a single experiment."""
    config['kfold']['test'] = test_index
    config['kfold']['validation'] = validation_index

    print("seed {0}".format(config['seed']))
    set_seed(config['seed'])

    testing, validating, training = get_tvt_filenames(
        config['kfold']['test'], config['kfold']['validation'],
        config['kfold']['k'], config['directory'],
        load_fun(config['get_label']), config['batch_size']
    )

    model = train(
        training, validating, config, prepare_batch, verbose=config['verbose']
    )

    # get metrics
    m = evaluate(testing, config, prepare_batch, model=model)

    m['kfold'] = deepcopy(config['kfold'])

    matrix, report, accuracy = predict(
        testing, config, prepare_batch, model=model
    )
    m['confusion_matrix'] = matrix
    m['report'] = report

    m['seed'] = config['seed']

    return m


def run_all(config):
    k = config['kfold']['k']
    metrics = []
    for myseed in range(0, config['repeat_seeds']):
        config['seed'] = myseed
        for test_index, validation_index in permute_index_kfold(k):
            m = run_experiment(test_index, validation_index, config)
            metrics.append(m)
            save_metrics(metrics, config['metrics'])
            clean_session()


run_all(load_config(sys.argv[1]))
