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

from random import randint

from .. import dataset


def to_random_predict(model):
    """Return a random predictor."""
    print("Load random predictor...")

    def f(batch, *args, **kwargs):
        return [
            dataset.category2categorical(randint(0, model.output_shape[1] - 1))
            for i in range(0, len(batch))
        ]
    return f


def predict_to_random_predict(model, type_):
    """Random predictor only on predict and evaluate."""
    if type_ in ['predict', 'evaluate']:
        return to_random_predict(model)
    return dataset.to_predict
