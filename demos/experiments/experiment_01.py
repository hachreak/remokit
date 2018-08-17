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

import os
import shutil
from copy import deepcopy
from sys import argv
from remokit.dataset import permute_index_kfold
from remokit.utils import clean_session, load_config
from remokit.metrics import save_metrics
from remokit.experiments.extract_face import merge

from remokit.experiments import run_experiment, save_best


def run_all(config):
    k = config['kfolds']
    metrics = []
    save_best_model = save_best(config)
    for myseed in range(0, config['repeat_seeds']):
        config['seed'] = myseed
        for test_index, validation_index in permute_index_kfold(k):
            m, model = run_experiment(test_index, validation_index, config)
            metrics.append(m)
            save_metrics(m, config['metrics'])
            save_best_model(m, model)
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
        # create stage directory
        shutil.rmtree(c['destination'], ignore_errors=True)
        os.makedirs(c['destination'])

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
