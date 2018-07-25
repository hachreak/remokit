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

from remokit import dataset
from remokit.datasets.kdef import get_data, get_label
from remokit.dataset import _category, build_batches, kfold_count
from remokit.models.model01 import get_model
from remokit.train import run, compile_

shape_predictor = "data/shape_predictor_68_face_landmarks.dat"
directory = "data/KDEF-straight"

img_x, img_y, img_channels = 100, 100, 1
num_classes = len(_category)
epochs = 5
batch_size = 5
k = 10
index = 0

files = dataset.get_files(directory)
files = dataset.first(files, 20)
files = list(files)

files = list(files)
_, files = dataset.kfold_split(
    files, get_label=get_label, k=k, index=index
)

steps_per_epoch = kfold_count(files, k) * (k - 1) // batch_size

batches = build_batches(files, get_data, get_label, k, index, batch_size,
                        epochs, img_x, img_y, shape_predictor)

# get CNN model
model = get_model((img_x, img_y, img_channels), num_classes)
model = compile_(model)

run(model, batches, steps_per_epoch, epochs)

model.save("data/kdef_train_{0}.h5".format(index))
