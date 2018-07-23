
"""Model definition."""

from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam


def get_model(img_x, img_y, img_channels, num_classes):
    input_shape = (img_x, img_y, img_channels)
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu',
              input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=categorical_crossentropy, optimizer=Adam(),
                  metrics=['accuracy'])

    return model
