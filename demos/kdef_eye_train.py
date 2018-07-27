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

"""Train CNN model01 with KDEF dataset eye only."""

from __future__ import absolute_import

from remokit import dataset, adapters
from remokit.datasets import get_filenames
from remokit.datasets.kdef import get_data, get_label
from remokit.models.model01 import get_model
from remokit.train import run, compile_


directory = "data/KDEF-straight_cut/eyes-mouth/eyes"
index = 0
k = 10
batch_size = 5
epochs = 20
num_classes = len(dataset._category)
shape = 100, 100, 1
img_x, img_y, _ = shape

validating, training = get_filenames(index, k, directory, get_label)

steps_per_epoch = len(training) // batch_size

training = dataset.epochs(training, epochs=epochs)

stream = get_data(training)
stream = dataset.categorical(stream)

batches = dataset.stream_batch(stream, batch_size)
batches = dataset.batch_adapt(batches, [
    adapters.rgb_to_bn,
    adapters.resize(img_x, img_y),
    adapters.matrix_to_bn,
    adapters.normalize
])

# get CNN model
model = get_model(shape, num_classes)
model = compile_(model)

run(model, batches, steps_per_epoch, epochs)

model.save("data/kdef_eye_train_{0}.h5".format(index))
