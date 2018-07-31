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

"""Model utilities."""

from keras.models import Model, load_model as _load_model
from keras import backend as K


def submodel(model, last_layer):
    """Get a submodel."""
    last_layer = model.get_layer(last_layer)
    return Model(inputs=model.input, outputs=last_layer.output)


def subfun(model, last_layer):
    """Get a submodel as a callable function."""
    last_layer = model.get_layer(last_layer)
    return K.function([model.layers[0].input], [last_layer.output])


def get_conv_layers(model):
    """Remove classification layer in the end of the CNN."""
    flatten = [l.name for l in model.layers if l.name.startswith('flatten')]
    #  return model
    sub = submodel(model, flatten[-1])
    # FIXME see keras#6462
    sub._make_predict_function()
    return sub


def load_model(file_h5):
    """Load model from file."""
    model = _load_model(file_h5)
    # FIXME see keras#6462
    model._make_predict_function()
    return model
