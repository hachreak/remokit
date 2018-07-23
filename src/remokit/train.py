
"""Train the model."""

from keras.callbacks import Callback


class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


def run(get_model, train, test, batch_size=None, epochs=None, history=None):
    """Run training."""
    batch_size = batch_size or 128
    epochs = epochs or 10
    history = history or AccuracyHistory()

    x_train, y_train = train
    x_test, y_test = test

    img_x, img_y, img_channels = x_train[0].shape
    (num_classes,) = y_train[0].shape
    # get CNN model
    model = get_model((img_x, img_y, img_channels), num_classes)
    # run training
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=test, callbacks=[history])

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss: ", score[0])
    print("Test accuracy: ", score[1])
    return model
