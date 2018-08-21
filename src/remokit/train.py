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

"""Train the model."""

from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


def default():
    """Default configuration."""
    return {
      "early_stop": {
        "mode": "auto",
        "monitor": "val_loss",
        "min_delta": 0,
        "patience": 10
      },
      "reduce_lr": {
        "monitor": "val_loss",
        "factor": 0.5,
        "patience": 5,
        "min_lr": 0.0001
      },
      "optimizer": {
        "lr": 0.001
      }
    }


def compile_(model, config):
    """Compile model before train."""
    # compile model
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(**config['optimizer']),
                  metrics=['accuracy'])
    return model


def run(model, batches, steps_per_epoch, epochs, config, validation_data=None,
        validation_steps=None, history=None, **kwargs):
    """Run training."""
    early_stop = EarlyStopping(verbose=1, **config['early_stop'])
    reduce_lr = ReduceLROnPlateau(verbose=1, **config['reduce_lr'])

    model.fit_generator(
        generator=batches, max_queue_size=1,
        steps_per_epoch=steps_per_epoch, epochs=epochs,
        callbacks=[reduce_lr, early_stop], shuffle=True,
        validation_data=validation_data, validation_steps=validation_steps,
        **kwargs
    )
    return model
