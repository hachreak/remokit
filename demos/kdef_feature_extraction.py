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

"""Demo of feature extraction."""

from remokit import dataset, adapters
from remokit.datasets.kdef import get_data
from remokit.preprocessing import features
import dlib

directory = "data/KDEF-straight"
shape_predictor = "data/shape_predictor_68_face_landmarks.dat"
img_x, img_y = 100, 100

filenames = dataset.get_files(directory)
filenames = list(filenames)

stream = get_data(filenames)

batches = dataset.stream_batch(stream, 1)

batches = dataset.batch_adapt(batches, [
    dataset.apply_to_y(dataset.foreach(dataset.categorical)),
    dataset.apply_to_x(dataset.foreach(
        adapters.resize(100, 100)
    )),
    dataset.apply_to_x(dataset.foreach(adapters.astype('uint8'))),
    dataset.apply_to_x(dataset.foreach(
        features.extract(shape_predictor)
    )),
    dataset.apply_to_x(dataset.foreach(features.expand2image(100, 100)))

])

x, y = next(batches)

img = x[0]

win = dlib.image_window()
win.set_image(img)
dlib.hit_enter_to_continue()
