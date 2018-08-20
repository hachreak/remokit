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

from sys import argv, exit
from copy import deepcopy
from remokit.utils import load_config, clean_session
from remokit.dataset import permute_index_kfold
from remokit.experiments import run_experiment, save_best
from remokit.preprocessing.extract_face import save, preprocess
from remokit.metrics import append_metrics


def load_all(config_files):
    """Load config files."""
    return [load_config(config_file) for config_file in config_files]


def preprocess_all(configs):
    """Preprocess all."""
    for config in configs:
        preprocess(save, config)


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
        for config in configs:
            config['seed'] = s
            save_best_model = save_best(config)
            # run, save best/metrics
            m, model = run_experiment(t, v, config)
            save_best_model(m, model)
            append_metrics(m, config['metrics'])
        # run, save best/metrics for main config
        main_config['seed'] = s
        m, model = run_experiment(t, v, main_config)
        save_best(main_config)(m, model)
        append_metrics(m, main_config['metrics'])
        clean_session()


def permute(main_config):
    k = main_config['kfolds']
    for myseed in range(0, main_config['repeat_seeds']):
        for test_index, validation_index in permute_index_kfold(k):
            yield test_index, validation_index, myseed


def main(args):
    if len(args) < 2:
        menu = "Usage: {0} [main_config] config_file1 config_file2 ... \n"
        print(menu.format(args[0]))
        exit(1)

    main_config = load_config(args[1])
    configs = copy_conf(main_config, load_all(args[2:]))
    preprocess_all(configs)
    run_all(main_config, configs)


main(deepcopy(argv))