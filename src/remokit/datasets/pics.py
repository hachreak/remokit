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

"""Dataset PICS Pain

Dataset source: http://pics.stir.ac.uk/2D_face_sets.htm
Name: Pain expression subset
"""

import os
from .. import detect, dataset
from . import ExpressionNotFound

_label = {
    'n': 'neutral',
    'a': 'angry',
    'd': 'disgust',
    'f': 'afraid',
    'h': 'happy',
    'sa': 'sad',
    's': 'surprised',
}


def get_files(directory, *args, **kwargs):
    """Get image/label files."""
    for filename in dataset.get_files(directory, types=['.jpg']):
        try:
            get_label(filename)
            yield filename
        except ExpressionNotFound:
            # skip if can't find a valid expression
            pass


def get_data(files_stream):
    """Get a streaming of label/image to process."""
    for filename in files_stream:
        yield detect.load_img(filename), get_label(filename)


def get_label(filename):
    """Convert a 2 chars label to a full name label."""
    f = os.path.basename(filename)
    #  v1 = f[1]
    start = 2
    try:
        int(f[2])
        start = 3
    except ValueError:
        pass
    end = start + 1
    try:
        int(f[end])
    except ValueError:
        end += 1
    try:
        return _label[f[start:end]]
    except KeyError:
        raise ExpressionNotFound()
