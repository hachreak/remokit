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

from remokit import dataset
from remokit.datasets.kdef import get_data
from remokit.preprocessing import features
import dlib

directory = "data/KDEF-straight"
shape_predictor = "data/shape_predictor_68_face_landmarks.dat"
img_x, img_y = 100, 100

filenames = dataset.get_files(directory)
#  filenames = dataset.first(filenames, 21)
filenames = list(filenames)

stream = get_data(filenames)
stream = dataset.categorical(stream)
stream = features.extract(shape_predictor, (img_x, img_y))(stream)

label, img = next(stream)

win = dlib.image_window()
win.set_image(img)
dlib.hit_enter_to_continue()
