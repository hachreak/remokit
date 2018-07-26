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

"""Train CNN model01 with KDEF dataset."""

from __future__ import absolute_import

from remokit.dataset import _category
from remokit.models.model01 import get_model
from remokit.train import run, compile_

from kdef_pipeline import training


num_classes = len(_category)
index = 0
batch_size = 10
epochs = 10

(shape, epochs, steps_per_epoch, batches) = training(
    index, batch_size, epochs
)

# get CNN model
model = get_model(shape, num_classes)
model = compile_(model)

run(model, batches, steps_per_epoch, epochs)

model.save("data/kdef_train_{0}.h5".format(index))
