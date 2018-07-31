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
from keras.callbacks import Callback


class AccuracyHistory(Callback):

    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


def compile_(model):
    """Compile model before train."""
    # compile model
    model.compile(loss=categorical_crossentropy, optimizer=Adam(),
                  metrics=['accuracy'])
    return model


def run(model, batches, steps_per_epoch, epochs, history=None):
    """Run training."""
    history = history or AccuracyHistory()
    model.fit_generator(
        generator=batches, max_queue_size=1, verbose=1,
        steps_per_epoch=steps_per_epoch, epochs=epochs,
        callbacks=[history], shuffle=False
    )
    return model
