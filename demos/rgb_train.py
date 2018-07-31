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

"""Train CNN model01 with colored image dataset."""

from __future__ import absolute_import

import sys
import numpy as np
from datetime import datetime
from remokit import dataset, adapters
from remokit.datasets import get_filenames
from remokit.datasets.kdef import get_data, get_label
from remokit.train import run, compile_
from remokit.utils import load_fun, set_reproducibility


if len(sys.argv) < 9:
    msg = ("Usage: {0} "
           "[directory] [img_x] [img_y] [index] [k] [batch_size] [epochs] "
           "[model.h5] [seed]")
    print(msg.format(sys.argv[0]))
    sys.exit(1)

directory = sys.argv[1]
img_x = int(sys.argv[2])
img_y = int(sys.argv[3])
index = int(sys.argv[4])
k = int(sys.argv[5])
batch_size = int(sys.argv[6])
epochs = int(sys.argv[7])
model_file = sys.argv[8]
get_model = load_fun(sys.argv[9])
seed_value = sys.argv[10] if len(sys.argv) > 10 else datetime.now().microsecond

# Start training

set_reproducibility(seed_value)

num_classes = len(dataset._category)
shape = img_x, img_y, 1

validating, training = get_filenames(
    index, k, directory, get_label, batch_size
)

steps_per_epoch = len(training) // batch_size

training = dataset.epochs(training, epochs=epochs)

stream = get_data(training)
stream = dataset.categorical(stream)

get_labels = adapters.extract_labels()

batches = dataset.stream_batch(stream, batch_size)
batches = dataset.batch_adapt(batches, [
    get_labels,
    adapters.rgb_to_bn,
    adapters.resize(img_x, img_y),
    adapters.matrix_to_bn,
    adapters.normalize
])

# get CNN model
model = get_model(shape, num_classes)
model = compile_(model)

run(model, batches, steps_per_epoch, epochs)

model.save(model_file)

print("Training")

y_val = dataset.list_apply(dataset.categorical2category, get_labels.labels)
counts = np.bincount(y_val)
cats = np.nonzero(counts)[0]
for cat, count in zip(cats, counts[cats]):
    print("{0:<15}: {1} elements".format(dataset.category2label(cat), count))
