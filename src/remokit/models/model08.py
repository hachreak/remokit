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

"""Model 08 definition."""

from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Input

from .inception import get_model as inception


def get_model(input_shape, num_classes):
    """Get a model with 2 inception layers."""
    input_img = Input(shape=input_shape)

    inc1 = inception(input_img)
    inc2 = inception(inc1)
    flat = Flatten()(inc2)

    dense1 = Dense(512, activation='relu')(flat)
    drop1 = Dropout(0.5)(dense1)
    dense2 = Dense(num_classes, activation='softmax')(drop1)

    return Model(inputs=input_img, output=dense2)
