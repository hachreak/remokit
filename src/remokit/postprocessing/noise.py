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

"""Inject noise."""

import numpy as np
from random import randint, uniform

from .. import dataset


def to_random_predict(model):
    """Return a random predictor."""
    print("Load random predictor...")

    def probability(todo):
        min_ = 0
        max_ = 0.3
        if todo == 1:
            min_ = 0.7
            max_ = 1
        return uniform(min_, max_)

    def f(batch, *args, **kwargs):
        mapping = [
            dataset.category2categorical(randint(0, model.output_shape[1] - 1))
            for i in range(0, len(batch))
        ]
        result = []
        for m in mapping:
            result.append([probability(v) for v in m])
        return np.array(result)
    return f


def predict_to_random_predict(type_):
    """Random predictor only on predict and evaluate."""
    if type_ in ['predict', 'evaluate']:
        return to_random_predict
    return dataset.to_predict
