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

"""Experiment 01."""

from __future__ import absolute_import

import sys
import json
import numpy as np
from copy import deepcopy
from remokit.experiments.rgb_cnn import train, evaluate, predict
from remokit.datasets import get_tvt_filenames
from remokit.dataset import permute_index_kfold
from remokit.utils import load_fun, set_reproducibility


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


config_file = sys.argv[1]
with open(config_file) as data_file:
    config = json.load(data_file)

k = config['kfold']['k']
metrics = []
for test_index, validation_index in permute_index_kfold(k):
    config['kfold']['test'] = test_index
    config['kfold']['validation'] = validation_index

    set_reproducibility(config['seed'])

    testing, validating, training = get_tvt_filenames(
        config['kfold']['test'], config['kfold']['validation'],
        config['kfold']['k'], config['directory'],
        load_fun(config['get_label']), config['batch_size']
    )

    model = train(training, validating, config, verbose=config['verbose'])

    m = evaluate(testing, config, model=model)
    m['kfold'] = deepcopy(config['kfold'])

    matrix, report, accuracy = predict(testing, config, model=model)
    m['confusion_matrix'] = matrix
    m['report'] = report

    m['seed'] = config['seed']

    metrics.append(m)

    # save metrics
    with open(config['metrics'].format(config['seed']), 'w') as outfile:
        json.dump(metrics, outfile, cls=NumpyEncoder)
