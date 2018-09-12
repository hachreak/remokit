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

"""Dataset FER 2013

Dataset source: https://www.kaggle.com/c/
challenges-in-representation-learning-facial-expression-recognition-challenge

Best result in the competition: 0.71161
"""

import numpy as np
import os
import dlib
import csv

from .. import detect, dataset

_label = {
    'NE': 'neutral',
    'AN': 'angry',
    'DI': 'disgust',
    'AF': 'afraid',
    'HA': 'happy',
    'SA': 'sad',
    'SU': 'surprised',
}

_csv_label = {
    '0': 'AN',
    '1': 'DI',
    '2': 'AF',
    '3': 'HA',
    '4': 'SA',
    '5': 'SU',
    '6': 'NE',
}


def get_preprocessed_files(directory, *args, **kwargs):
    """Preprocess cvs and then get file list."""
    csv_filename = list(dataset.get_files(directory, types=['.csv']))[0]
    destination = _build_destination_directory(csv_filename)
    if not os.path.exists(destination):
        os.makedirs(destination)
        # extract images
        preprocess_csv(csv_filename, destination)
    # return images list
    return get_files(destination)


def get_files(directory, *args, **kwargs):
    """Get image/label files."""
    return dataset.get_files(directory, types=['.jpg'])


def get_data(files_stream):
    """Get a streaming of label/image to process."""
    for filename in files_stream:
        yield detect.load_img(filename), get_label(filename)


def get_label(filename):
    """Convert a 2 chars label to a full name label."""
    keys = os.path.basename(filename).split('.')
    return _label[keys[1]]


def preprocess_csv(filename, destination):
    """Extract images/annotations from the csv file.

    Before use it, you should extract images with this function and than
    you are ready to use it.
    """
    with open(filename) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        # get headers
        next(reader)
        # get content
        for index, line in enumerate(reader):
            label = _csv_label[line[0]]
            img = _load_img(line[1])
            name = "{0}.{1}.jpg".format(index, label)
            fullname = os.path.join(destination, name)
            print("Save {0}".format(fullname))
            dlib.save_image(img, fullname)


def _load_img(column):
    """Load image from csv column."""
    return np.array(
        [int(n) for n in column.split(' ')]
    ).astype('uint8').reshape(48, 48)


def _build_destination_directory(csv_filename):
    """Build destination directory for images contained inside the csv."""
    # create destination name
    basedir, _ = os.path.split(csv_filename)
    destination = os.path.join(basedir, 'images')

    return destination
