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

from remokit.experiments.rgb_cnn import experiment
from remokit.dataset import permute_index_kfold


config = {
    "get_label": "remokit.datasets.kdef.get_label",
    "get_data": "remokit.datasets.kdef.get_data",
    "kfold": {
        "test": 0,
        "validation": 1,
        "k": 10
    },
    "image_size": {
      "img_x": 100,
      "img_y": 100
    },
    "directory": "data/KDEF-straight_cut/eyes-mouth/eyes",
    "model": "remokit.models.model01.get_model",
    "batch_size": 28,
    "epochs": 15,
    "seed": 0,
    "result": "data/experiment_{0}_{1}.h5",
    "is_training": True
}

k = config['kfold']['k']
result = config['result']
for test_index, validation_index in permute_index_kfold(k):
    config['kfold']['test'] = test_index
    config['kfold']['validation'] = validation_index
    config['result'] = result.format(test_index, validation_index)

    experiment(config)
