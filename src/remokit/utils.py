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

"""Utils."""

import os
import importlib


def load_fun(name):
    """Load a function from name."""
    module, fun_name = name.rsplit('.', 1)
    mod = importlib.import_module(module)
    return getattr(mod, fun_name)


def set_seed(myseed):
    """Globally set seed."""
    from random import seed
    from numpy.random import seed as npseed
    from tensorflow import set_random_seed as tfseed

    seed(myseed)
    npseed(int(myseed))
    tfseed(int(myseed))
    os.environ['PYTHONHASHSEED'] = str(myseed)


def set_reproducibility():
    """Set system as reproducible as much as possible.

    See keras#2280
    """
    import tensorflow as tf
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1)

    from keras import backend as K
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
