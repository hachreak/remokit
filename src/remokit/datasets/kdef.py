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

"""Dataset KDEF

Dataset source: www.emotionlab.se/resources/kdef
"""

import os
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


def get_files(directory, *args, **kwargs):
    """Get image/label files."""
    return dataset.get_files(directory)


def get_data(files_stream):
    """Get a streaming of label/image to process."""
    for filename in files_stream:
        yield detect.load_img(filename), get_label(filename)


def get_label(filename):
    """Convert a 2 chars label to a full name label."""
    key = os.path.basename(filename)[4:6]
    return _label[key]
