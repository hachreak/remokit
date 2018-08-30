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

"""Experiment 06."""

from __future__ import absolute_import

from sys import argv, exit
from copy import deepcopy
from keras.models import load_model

from remokit.utils import load_config, load_fun, set_seed
from remokit.experiments import run_experiment, save_best, files_split, \
        evaluate
from remokit.preprocessing import preprocess
from remokit.metrics import save_metrics, plot_saliency


def train(test_index, validation_index, myseed, config):
    config['seed'] = myseed
    save_best_model = save_best(config)
    m, model = run_experiment(test_index, validation_index, config)
    save_best_model(m, model)
    save_metrics([m], config['metrics'])


def saliency(test_index, validation_index, myseed, config, setname, index):
    set_seed(myseed)
    prepare_batch = load_fun(config['prepare_batch'])
    setnames = {
        'testing': 0,
        'validating': 1,
        'training': 2,
    }
    with files_split(test_index, validation_index, config) as files:
        filenames = files[setnames[setname]][index:index+1]
        model = load_model(config['best_model'])
        config['batch_size'] = 1

        # print evaluation
        metrics = evaluate(filenames, config, prepare_batch, model)
        # plot saliency
        batches, _ = prepare_batch(filenames, config, 1)
        plt = next(plot_saliency(model, batches))
        plt.suptitle("Accuracy: {0}".format(metrics['acc']))
        plt.show()
        #  plt.show()
        #  plt.gcf().clear()


def main(args):
    if len(args) < 2:
        menu = (
            "Usage: {0} preprocess [config_file] \n"
            "       {0] train [config_file] [test_index] [val_index] [seed]\n"
            "       {0} saliency [config_file] [test_index] [val_index] "
            "[seed] [testing|validating|training] [index]"
        )
        print(menu.format(args[0]))
        exit(1)

    config = load_config(args[2])
    if args[1] == 'preprocess':
        preprocess(config)
    elif args[1] == 'train':
        test_index = int(args[3])
        validation_index = int(args[4])
        myseed = int(args[5])
        train(test_index, validation_index, myseed, config)
    else:
        test_index = int(args[3])
        validation_index = int(args[4])
        myseed = int(args[5])
        setname = args[6]
        index = int(args[7])
        saliency(test_index, validation_index, myseed, config, setname, index)


main(deepcopy(argv))
