
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
        callbacks=[history]

    )
    return model
