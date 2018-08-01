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


def apply_adapters(batch, adapters):
    """Apply adapters to the batch."""
    for adapter in adapters:
        batch = adapter(batch)
    return batch


def batch_adapt(batches, adapters):
    """Adapt a streaming batch."""
    for batch in batches:
        yield apply_adapters(batch, adapters)


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


def permute_index_kfold(k):
    """Make permutation of testing/validation indices."""
    for i in range(0, k):
        test_index = i
        validation_index = (i + 1) % k
        yield test_index, validation_index


def kfold_split(filenames, get_label, k=10, shuffle=True):
    """Split filenames in validation and training partitions."""
    per_category = files_per_category(filenames, get_label)
    boxes = [[] for _ in range(0, k)]
    for names in per_category.values():
        slices = np.array_split(names, k)
        for i, s in enumerate(slices):
            boxes[i].extend(s)
    if shuffle:
        for b in boxes:
            np.random.shuffle(b)
    return boxes


def get_tvt_sets(boxes, index_test, index_validation):
    """Split in test, validation, trainin sets."""
    test = boxes.pop(index_test)
    if index_validation > index_test:
        index_validation -= 1
    validation = boxes.pop(index_validation)
    training = list_flatten(boxes)
    return test, validation, training


def get_tt_sets(boxes, index_test):
    """Split in test, trainin sets."""
    test = boxes.pop(index_test)
    training = list_flatten(boxes)
    return test, training


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


def apply_to_x(fun):
    """Apply function to X."""
    def f(batch):
        (Xblock, y) = batch
        return (fun(Xblock), y)
    return f


def list_flatten(batch):
    """Flatten a list."""
    result = []
    for b in batch:
        result.extend(b)
    return result


def flatten(batch):
    """Flatten a list of list."""
    if isinstance(batch, list):
        batch = np.array(batch)
    return batch.flatten()


def foreach(fun):
    """Call for each."""
    def f(list_):
        result = []
        for el in list_:
            result.append(fun(el))
        return result
    return f


def to_np_array(x):
    """Transform in numpy array."""
    return np.array(x)


def merge_batches(batches_list, adapters):
    """Join multiple batches."""
    while True:
        X_list = None
        y_list = None
        for b in batches_list:
            (X, y) = next(b)
            if X_list is None:
                X_list = [[] for _ in range(0, X.shape[0])]
            for i, singlex in enumerate(X):
                X_list[i].append(singlex)
            y_list = y
        batch = (X_list, y_list)
        adapters.append(apply_to_x(to_np_array))
        yield apply_adapters(batch, adapters=adapters)


def label2category(label):
    """Convert label to category."""
    return _category[label]


def category2label(category):
    """Category as number converted to label string."""
    rev_cat = {v: k for k, v in _category.items()}
    return rev_cat[category]


def categorical2category(category):
    """Convert category to label."""
    return np.argmax(category)


def list_apply(fun, args):
    """Apply function to each argument of the list."""
    return [fun(arg) for arg in args]


def to_predict(model):
    """Call predict as fun."""
    def f(batch):
        return model.predict(batch)
    return f
