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

import os

from sys import argv, exit
from copy import deepcopy
from keras.models import load_model

from remokit.utils import load_config, load_fun, set_seed, recreate_directory
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


def get_ckp_conf(config):
    """Get CK+ configuration."""
    [ckconf] = [p for p in config['preprocess'] if 'ck+' in p['directory']]
    return ckconf


def get_testing(config, test_index, validation_index, myseed):
    """Get testing files."""
    ckconf = get_ckp_conf(config)
    set_seed(myseed)

    filenames = load_fun(ckconf['get_files'])(**ckconf)

    # get all testing files
    testing, _, _ = get_tvt_filenames(
        test_index, validation_index,
        config['kfolds'], filenames,
        load_fun(ckconf['get_label']), config['batch_size']
    )
    return testing


def predict_sequence(config, testing, model=None):
    ckconf = get_ckp_conf(config)

    # load configuration
    model = model or load_model(config['best_model'])
    ckconf['image_size'] = deepcopy(config['image_size'])
    ckconf['batch_size'] = config['batch_size']
    prepare_batch = load_fun('remokit.experiments.rgb_face_cnn.prepare_batch')

    # plot prediction
    return predict(testing, ckconf, prepare_batch, model)


def plot_title(testing):
    """Get plot tilte."""
    _, filepath = ckp._get_relative_path(testing[-1])
    label = ckp.get_label(testing[-1])
    return "Sequence from: {0}\nEmotion: {1}".format(filepath, label)


def run_all(myseed, config, test_index, validation_index):
    model = load_model(config['best_model'])
    testing = get_testing(config, test_index, validation_index, myseed)
    dest = config['predictions']
    recreate_directory(dest)
    for t in testing:
        seq = sorted(ckp.get_sequence(t))
        y_pred = predict_sequence(config, seq, model=model)
        basename = os.path.basename(t)
        name = "{0}_{1}_{2}_{3}.png".format(
            basename, test_index, validation_index, myseed
        )
        fullname = os.path.join(dest, name)
        plt = plot(plot_title(seq), y_pred)
        plt.savefig(fullname)
        plt.close(fullname)
        plt.gcf().clear()
        print("Plotted {0}".format(name))


def run(myseed, config, test_index, validation_index, i):
    testing = get_testing(config, test_index, validation_index, myseed)
    # print description
    print("Sequence from: {0}".format(testing[i]))
    print("Emotion: {0}".format(ckp.get_label(testing[i])))
    # select one and get the entire sequence
    testing = sorted(ckp.get_sequence(testing[i]))
    # predict and show
    y_pred = predict_sequence(config, testing)
    plot(plot_title(testing), y_pred).show()


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
                "[seed] [i>=0] \n"
                "       {0} predict [config_file] [test_index] [val_index] "
                "[seed]")
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
        if len(args) < 7:
            run_all(myseed, config, test_index, validation_index)
        else:
            i = int(args[6])
            run(myseed, config, test_index, validation_index, i)


main(deepcopy(argv))
