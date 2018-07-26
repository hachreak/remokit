
"""Kdef demo pipeline."""

from remokit import dataset
from remokit.datasets.kdef import get_data, get_label
from remokit.preprocessing import features
from remokit.datasets import kdef


def _get_filenames(index, k):
    directory = "data/KDEF-straight"

    filenames = dataset.get_files(directory)
    #  filenames = dataset.first(filenames, 21)
    filenames = list(filenames)

    return dataset.kfold_split(
        filenames, get_label=get_label, k=k, index=index
    )


def _get_batches(filenames, img_x, img_y, shape_predictor, batch_size):
    stream = get_data(filenames)
    stream = dataset.categorical(stream)
    stream = features.extract(shape_predictor, (img_x, img_y))(stream)

    batches = dataset.stream_batch(stream, batch_size)
    batches = features.batch_adapt(batches)

    return batches


def validating(index, k, batch_size):
    """Build pipeline to validate model."""
    shape_predictor = "data/shape_predictor_68_face_landmarks.dat"

    img_x, img_y = 100, 100

    validating, _ = _get_filenames(index, k)

    filenames = validating

    y_val = [kdef.get_label(f) for f in filenames]

    batches = _get_batches(filenames, img_x, img_y, shape_predictor,
                           batch_size)

    return (
        batches,
        y_val
    )


def training(index, k, batch_size, epochs):
    """Build pipeline to train model."""
    shape_predictor = "data/shape_predictor_68_face_landmarks.dat"

    img_x, img_y, img_channels = 100, 100, 1

    _, training = _get_filenames(index, k)
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
