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

"""Experiment 03."""

from __future__ import absolute_import

import os
from sys import argv, exit
from copy import deepcopy
from remokit.utils import load_config, clean_session
from remokit.dataset import permute_index_kfold
from remokit.experiments import run_experiment
from remokit.preprocessing import preprocess
from remokit.metrics import append_metrics


def load_all(config_files):
    """Load config files."""
    result = []
    for config_file in config_files:
        config = load_config(config_file)
        config['name'] = config_file
        result.append(config)
    return result


def preprocess_all(configs):
    """Preprocess all."""
    for config in configs:
        preprocess(config)


def copy_conf(main_config, configs):
    """Copy global configurations."""
    for i in range(0, len(configs)):
        configs[i]['kfolds'] = main_config['kfolds']
        configs[i]['batch_size'] = main_config['batch_size']
        configs[i]['epochs'] = main_config['epochs']
        configs[i]['verbose'] = main_config['verbose']
    return configs


def run_all(main_config, configs):
    """Run all."""
    for t, v, s in permute(main_config):
        # run, save best/metrics for each submodel
        for config in configs:
            print("Run submodel from {0}".format(config['name']))
            config['seed'] = s
            m, model = run_experiment(t, v, config)
            model.save(config['best_model'])
            append_metrics(m, config['metrics'])
        # run, save best/metrics for main config
        print("Run model from {0}".format(main_config['name']))
        main_config['seed'] = s
        m, model = run_experiment(t, v, main_config)
        model.save(main_config['best_model'])
        append_metrics(m, main_config['metrics'])
        clean_session()


def permute(main_config):
    k = main_config['kfolds']
    for myseed in range(0, main_config['repeat_seeds']):
        for test_index, validation_index in permute_index_kfold(k):
            yield test_index, validation_index, myseed


def get_config_filenames(filename, main_config):
    """Get configuration filenames for submodels."""
    directory, _ = os.path.split(filename)
    return [os.path.join(directory, s['config'])
            for s in main_config['submodels']]


def main(args):
    if len(args) < 2:
        menu = "Usage: {0} [main_config]"
        print(menu.format(args[0]))
        exit(1)

    filename = args[1]
    main_config = load_config(filename)
    main_config['name'] = filename
    configs = copy_conf(
        main_config, load_all(get_config_filenames(filename, main_config))
    )
    preprocess_all(configs)
    run_all(main_config, configs)


main(deepcopy(argv))
