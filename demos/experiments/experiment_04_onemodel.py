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

"""Experiment 04 - run a single model and save the result."""

from __future__ import absolute_import

from sys import argv, exit
from copy import deepcopy
from remokit.utils import load_config

from remokit.experiments import run_experiment, save_best
from remokit.preprocessing import preprocess
from remokit.metrics import save_metrics


def run(test_index, validation_index, myseed, config):
    config['seed'] = myseed
    save_best_model = save_best(config)
    m, model = run_experiment(test_index, validation_index, config)
    save_best_model(m, model)
    save_metrics([m], config['metrics'])


def main(args):
    if len(args) < 2:
        menu = ("Usage: {0} [preprocess] config_file \n"
                "       {0] run config_file test_index validation_index seed ")
        print(menu.format(args[0]))
        exit(1)

    config = load_config(args[2])
    if args[1] == 'preprocess':
        preprocess(config)
    else:
        test_index = int(args[3])
        validation_index = int(args[4])
        myseed = int(args[5])
        run(test_index, validation_index, myseed, config)


main(deepcopy(argv))
