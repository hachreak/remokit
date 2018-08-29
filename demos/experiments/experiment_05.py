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

"""Experiment 05 - Check CK+ sequence."""

from __future__ import absolute_import

import sys
from sys import argv, exit
from copy import deepcopy
from keras.models import load_model

from remokit.utils import load_config, load_fun, set_seed
from remokit.datasets import ckp, get_tvt_filenames
from remokit.experiments import run_experiment, save_best
from remokit.preprocessing import preprocess
from remokit.metrics import plot_prediction as plot, save_metrics


def predict(testing, config, prepare_batch, model):
    """Run predict and return raw result."""
    # build input batch stream
    batches, shape = prepare_batch(
        testing, config, 1
    )
    steps_per_epoch = len(testing) // config['batch_size']

    return model.predict_generator(batches, steps=steps_per_epoch)


def run(myseed, config, test_index, validation_index, i):
    [ckconf] = [p for p in config['preprocess'] if 'ck+' in p['directory']]
    set_seed(myseed)

    filenames = load_fun(ckconf['get_files'])(**ckconf)

    # get all testing files
    testing, _, _ = get_tvt_filenames(
        test_index, validation_index,
        config['kfolds'], filenames,
        load_fun(ckconf['get_label']), config['batch_size']
    )
    # print description
    print("Sequence from: {0}".format(testing[i]))
    print("Emotion: {0}".format(ckp.get_label(testing[i])))
    # select one and get the entire sequence
    testing = sorted(ckp.get_sequence(testing[i]))

    # load configuration
    model = load_model(config['best_model'])
    ckconf['image_size'] = deepcopy(config['image_size'])
    ckconf['batch_size'] = config['batch_size']
    prepare_batch = load_fun('remokit.experiments.rgb_face_cnn.prepare_batch')

    # plot prediction
    y_pred = predict(testing, ckconf, prepare_batch, model)
    plot(y_pred).show()


def experiment(test_index, validation_index, myseed, config):
    config['seed'] = myseed
    save_best_model = save_best(config)
    m, model = run_experiment(test_index, validation_index, config)
    save_best_model(m, model)
    save_metrics([m], config['metrics'])


def main(args):
    if len(args) < 2:
        menu = ("Usage: {0} preprocess [config_file] \n"
                "       {0} train [config_file] [test_index] [val_index] "
                "[seed]\n"
                "       {0} predict [config_file] [test_index] [val_index] "
                "[seed] [i>0]")
        print(menu.format(args[0]))
        exit(1)

    config = load_config(args[2])
    if args[1] == 'preprocess':
        preprocess(config)
    elif args[1] == 'train':
        test_index = int(args[3])
        validation_index = int(args[4])
        myseed = int(args[5])
        experiment(test_index, validation_index, myseed, config)
    else:
        myseed = int(args[3])
        test_index = int(args[4])
        validation_index = int(args[5])
        i = int(args[6])
        if i == 0:
            sys.exit(1)
        run(myseed, config, test_index, validation_index, i)


main(deepcopy(argv))
