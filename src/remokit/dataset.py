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

"""Get dataset."""

import os
import cv2
import numpy as np
from keras.utils import to_categorical


_category = {
    'neutral': 0,
    'angry': 1,
    'disgust': 2,
    'afraid': 3,
    'happy': 4,
    'sad': 5,
    'surprised': 6,
}


def batch_adapt(batches, adapters):
    """Adapt a streaming batch."""
    #  import ipdb; ipdb.set_trace()
    for x, y in batches:
        for adapter in adapters:
            x = adapter(x)
        yield x, y


def categorical(stream):
    """Convert label to categorical."""
    for img, label in stream:
        cat = to_categorical(label2category(label), len(_category))
        yield img, cat


def ordered_categories():
    """Get the ordered version of categories."""
    cats = {v: k for k, v in _category.iteritems()}
    return [cats[v] for v in range(0, 7)]


def epochs(filenames, epochs=1):
    """Repeat filenames epochs times."""
    for _ in range(0, epochs):
        for name in filenames:
            yield name


def kfold_split(filenames, get_label, k=10, index=0, shuffle=True):
    """Split filenames in validation and training partitions."""
    per_category = files_per_category(filenames, get_label)
    validation = []
    testing = []
    for names in per_category.values():
        slices = np.array_split(names, k)
        validation.extend(slices.pop(index))
        for s in slices:
            testing.extend(s)
    if shuffle:
        np.random.shuffle(testing)
    return validation, testing


def files_per_category(filenames, get_label):
    """Return filenames per categories."""
    # init dict
    result = {}
    for cat in _category.keys():
        result[cat] = []
    # split files
    for filename in filenames:
        result[get_label(filename)].append(filename)
    return result


def count(directory, types=None):
    """Count how many files."""
    return len(list(get_files(directory=directory, types=types)))


def get_files(directory, types=None):
    """Get list of images reading recursively."""
    types = types or ['.jpg']
    for root, dirnames, files in os.walk(directory):
        for name in files:
            _, ext = os.path.splitext(name)
            if ext.lower() in types:
                yield os.path.join(root, name)


def first(stream, size):
    """Get only first `size` from the stream."""
    index = 0
    for value in stream:
        yield value
        index += 1
        if index >= size:
            raise StopIteration


def batch(block, size):
    """Split in batches the input."""
    for block in np.array_split(block, size):
        yield block


def stream_batch(stream, size):
    """Create batch on the fly."""
    while True:
        batch = []
        try:
            x = []
            y = []
            for i in range(size):
                x_value, y_value = next(stream)
                x.append(x_value)
                y.append(y_value)
            yield np.array(x), np.array(y)
        except StopIteration:
            if not batch:
                raise StopIteration
            yield batch


def label2category(label):
    """Convert label to category."""
    return _category[label]


def categorical2category(category):
    """Convert category to label."""
    return np.argmax(category)
