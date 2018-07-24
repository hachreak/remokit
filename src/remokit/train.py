
"""Train the model."""

from keras.callbacks import Callback


class AccuracyHistory(Callback):

    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


def run(model, train, batch_size=None, epochs=None, history=None):
    """Run training."""
    batch_size = batch_size or 128
    epochs = epochs or 10
    history = history or AccuracyHistory()

    x_train, y_train = train

    img_x, img_y, img_channels = x_train[0].shape
    (num_classes,) = y_train[0].shape
    # run training
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(x_train, y_train),
              callbacks=[history])
    return model
