
"""Kdef demo pipeline."""

import numpy as np
from remokit import dataset
from remokit.datasets.kdef import get_data, get_label
from remokit.preprocessing import features
from remokit.datasets import kdef


k = 10
img_x, img_y, img_channels = 40, 40, 1
shape_predictor = "data/shape_predictor_68_face_landmarks.dat"


def _get_filenames(index):
    directory = "data/KDEF-straight"

    filenames = dataset.get_files(directory)
    #  filenames = dataset.first(filenames, 21)
    filenames = list(filenames)

    v, t = dataset.kfold_split(
        filenames, get_label=get_label, k=k, index=index
    )
    np.random.shuffle(t)
    return v, t


def _get_batches(filenames, img_x, img_y, shape_predictor, batch_size):
    stream = get_data(filenames)
    stream = dataset.categorical(stream)
    stream = features.extract(shape_predictor, (img_x, img_y))(stream)

    batches = dataset.stream_batch(stream, batch_size)
    batches = features.batch_adapt(batches)

    return batches


def validating(index, batch_size):
    """Build pipeline to validate model."""
    validating, _ = _get_filenames(index)

    filenames = validating

    y_val = [kdef.get_label(f) for f in filenames]

    batches = _get_batches(filenames, img_x, img_y, shape_predictor,
                           batch_size)

    return (batches, y_val)


def training(index, batch_size, epochs):
    """Build pipeline to train model."""
    _, training = _get_filenames(index)
    filenames = training

    steps_per_epoch = len(filenames) // batch_size

    filenames = dataset.epochs(filenames, epochs=epochs)

    batches = _get_batches(filenames, img_x, img_y, shape_predictor,
                           batch_size)

    return (
        (img_x, img_y, img_channels),
        epochs,
        steps_per_epoch,
        batches
    )
