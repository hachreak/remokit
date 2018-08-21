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

"""Model 07 definition."""

from keras import Sequential
from keras.layers import Dense, Dropout


def get_model(input_shape, num_classes):
    model = Sequential()

    nn = 784
    model.add(Dense(nn, activation='relu', input_dim=input_shape))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))

    return model
