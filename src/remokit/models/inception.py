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

"""Inception definition."""

from keras.layers import Conv2D, MaxPooling2D, concatenate, AveragePooling2D


def get_model(input_img, size):
    """Get an inception layer."""
    t1 = Conv2D(size, (1, 1), padding='same', activation='relu')(input_img)
    t1 = Conv2D(size, (3, 3), padding='same', activation='relu')(t1)

    t2 = Conv2D(size, (1, 1), padding='same', activation='relu')(input_img)
    t2 = Conv2D(size, (5, 5), padding='same', activation='relu')(t2)

    t3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
    t3 = Conv2D(size, (1, 1), padding='same', activation='relu')(t3)

    conc = concatenate([t1, t2, t3], axis=3)
    pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(conc)

    return pool
