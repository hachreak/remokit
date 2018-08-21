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

import os
import shutil
from copy import deepcopy

from remokit.utils import load_fun
#  from remokit.preprocessing.extract_face import prepare_batch


def preprocess(config):
    """Preprocess images."""
    # create stage directory
    shutil.rmtree(config['directory'], ignore_errors=True)
    os.makedirs(config['directory'])

    # collect configurations
    config_list = []
    for prep in config['preprocess']:
        c = deepcopy(prep)
        c['destination'] = config['directory']
        c['image_size'] = deepcopy(config['image_size'])
        config_list.append(c)

    # merge datasets
    indices = None
    for config in config_list:
        prepare_batch = load_fun(config['prepare_batch'])
        save = load_fun(config['save'])
        stream = prepare_batch(config)
        indices = save(stream, config, indices)
    print(indices)
