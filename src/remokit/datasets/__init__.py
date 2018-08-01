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

import random

from .. import dataset


def get_filenames(index, k, directory, get_label, batch_size):
    """Get a list of test/training files splittes with kfold."""
    filenames = dataset.get_files(directory)
    filenames = list(filenames)
    random.shuffle(filenames)

    v, t = dataset.get_tt_sets(dataset.kfold_split(
        filenames, get_label=get_label, k=k
    ), index)
    return _fill(v, batch_size), _fill(t, batch_size)


def get_tvt_filenames(index_t, index_v, k, directory, get_label, batch_size):
    """Get a list of test/validating/training files splittes with kfold."""
    filenames = dataset.get_files(directory)
    filenames = list(filenames)
    random.shuffle(filenames)

    ts, v, tr = dataset.get_tvt_sets(dataset.kfold_split(
        filenames, get_label=get_label, k=k
    ), index_t, index_v)
    return _fill(ts, batch_size), _fill(v, batch_size), _fill(tr, batch_size)


def _fill(filenames, batch_size):
    rest = len(filenames) % batch_size
    if rest > 0:
        rest = batch_size - rest
        index = random.randrange(len(filenames) - rest)
        end = index + rest
        filenames.extend(filenames[index:end])
    return filenames
