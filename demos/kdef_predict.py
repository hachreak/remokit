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

"""Predict using CNN model01 with KDEF dataset."""

from __future__ import absolute_import

from keras.models import load_model
from remokit import dataset
#  from remokit.dataset import _category
#  from remokit.models.model01 import get_model
#  from remokit.train import run, compile_
from sklearn.metrics import classification_report, confusion_matrix, \
        accuracy_score

from kdef_pipeline import validating


index = 0
batch_size = 10

(batches, y_val) = validating(index, batch_size)

steps = len(y_val) // batch_size

# get CNN model
model = load_model("data/kdef_train_{0}.h5".format(index))

y_pred = model.predict_generator(batches, steps=steps)

print('Confusion Matrix')

y_pred = [dataset.categorical2category(v) for v in y_pred]
y_val = y_val[:len(y_pred)]
y_val = [dataset._category[v] for v in y_val]

matrix = confusion_matrix(y_val, y_pred)
print(matrix)

report = classification_report(
    y_val, y_pred, target_names=dataset.ordered_categories())
print(report)

print("Accuracy")
print(accuracy_score(y_val, y_pred))
