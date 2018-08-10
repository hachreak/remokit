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

"""Dataset CK+

Dataset source: http://www.consortium.ri.cmu.edu/ckagree/
"""

import os
from .. import dataset
from . import ExpressionNotFound


_label = {
    0: 'neutral',
    1: 'angry',
    3: 'disgust',
    4: 'afraid',
    5: 'happy',
    6: 'sad',
    7: 'surprised',
}


def get_files(directory):
    """Get image/label files."""
    directory = os.path.abspath(directory)
    # for each label file, extract img files
    label_files = _get_label_files(directory)
    for label_file in label_files:
        try:
            # check if is correctly labelled
            _read_label(label_file)
            # get img file paths
            img_path = _from_label_file_to_img_dir(label_file)
            imgs = sorted(dataset.get_files(img_path, types=['.png']))
            slice_width = 2 if len(imgs) > 4 else 1
            yield os.path.join(img_path, imgs[0])
            # then the emotion faces
            for imgname in imgs[-slice_width:]:
                abs_name = os.path.join(img_path, imgname)
                yield abs_name
        except ExpressionNotFound:
            # skip contempt expression
            print("skip label {0}".format(label_file))


def get_label(filename):
    """From image filename get label."""
    if filename.endswith('001.png'):
        return _label[0]

    rest, relpath = _get_relative_path(filename)
    directory, _ = os.path.split(rest)
    label_dir = _get_label_dir(directory)

    lpath = os.path.join(label_dir, relpath)
    lname = next(dataset.get_files(lpath, types=['.txt']))

    fullname = os.path.join(lpath, lname)
    return _read_label(fullname)


def _read_label(label_file):
    """Read fron file the label."""
    with open(label_file) as lfile:
        index = int(float(next(lfile)))

    if index == 2:
        raise ExpressionNotFound()

    return _label[index]


def _get_label_dir(directory):
    return os.path.join(directory, 'Emotion')


def _get_img_dir(directory):
    return os.path.join(directory, 'cohn-kanade-images')


def _get_label_files(directory):
    """Get label files."""
    return dataset.get_files(_get_label_dir(directory), types=['.txt'])


def _from_label_file_to_img_dir(filename):
    """Get img directory from label filename."""
    rest, relpath = _get_relative_path(filename)
    directory, _ = os.path.split(rest)
    img_dir = _get_img_dir(directory)
    return os.path.join(img_dir, relpath)


def _get_relative_path(filepath):
    rest, dir1 = os.path.split(filepath)
    rest, dir2 = os.path.split(rest)
    rest, dir3 = os.path.split(rest)
    return rest, os.path.join(dir3, dir2)
