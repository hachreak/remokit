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

"""Predict CNN model01 with colored image dataset."""

from __future__ import absolute_import

import sys
from keras.models import load_model
from remokit import dataset, adapters
from remokit.datasets import get_filenames
from remokit.datasets.kdef import get_data, get_label
from sklearn.metrics import classification_report, confusion_matrix, \
        accuracy_score


if len(sys.argv) < 8:
    msg = ("Usage: {0} "
           "[directory] [img_x] [img_y] [index] [k] [batch_size] "
           "[model.h5]")
    print(msg.format(sys.argv[0]))
    sys.exit(1)

directory = sys.argv[1]
img_x = int(sys.argv[2])
img_y = int(sys.argv[3])
index = int(sys.argv[4])
k = int(sys.argv[5])
batch_size = int(sys.argv[6])
model_file = sys.argv[7]

# Start predict

shape = img_x, img_y, 1
img_x, img_y, _ = shape

validating, training = get_filenames(index, k, directory, get_label)

steps_per_epoch = len(validating) // batch_size

stream = get_data(validating)
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
model = load_model(model_file)

y_pred = model.predict_generator(batches, steps=steps_per_epoch)

y_val = dataset.list_apply(dataset.categorical2category, get_labels.labels)
y_pred = dataset.list_apply(dataset.categorical2category, y_pred)

ordered_labels = dataset.ordered_categories()

print('Confusion Matrix')

matrix = confusion_matrix(y_val, y_pred)
for i, row in enumerate(matrix):
    to_print = ''.join(['{:4}'.format(item) for item in row])
    print("{0:<15} {1}".format(ordered_labels[i], to_print))

report = classification_report(y_val, y_pred, target_names=ordered_labels)
print(report)

print("Accuracy")
print(accuracy_score(y_val, y_pred))
