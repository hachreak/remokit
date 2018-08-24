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

from keras import regularizers
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Input, AveragePooling2D, \
    Conv2D, MaxPooling2D

from .inception import get_model as inception


def get_model(input_shape, num_classes):
    """Get a model with 2 inception layers."""
    input_img = Input(shape=input_shape)

    c1 = Conv2D(
        16, kernel_size=(7, 7), strides=(2, 2), activation='relu'
    )(input_img)
    m1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(c1)

    inc1 = inception(m1, 8)
    inc2 = inception(inc1, 4)

    pool = AveragePooling2D((5, 5), strides=(1, 1), padding='same')(inc2)
    flat = Flatten()(pool)

    dense1 = Dense(
        1024, activation='relu',
        kernel_regularizer=regularizers.l2(0.01)
    )(flat)
    drop1 = Dropout(0.4)(dense1)
    dense2 = Dense(
        num_classes, activation='softmax',
        kernel_regularizer=regularizers.l2(0.01)
    )(drop1)

    return Model(inputs=input_img, output=dense2)
