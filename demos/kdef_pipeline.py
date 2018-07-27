
"""Kdef demo pipeline."""

from remokit import dataset, adapters
from remokit.datasets.kdef import get_data, get_label
from remokit.preprocessing import features
from remokit.datasets import kdef
from remokit.datasets import get_filenames


directory = "data/KDEF-straight"
k = 10
img_x, img_y, img_channels = 40, 40, 1
shape_predictor = "data/shape_predictor_68_face_landmarks.dat"


def _get_batches(filenames, img_x, img_y, shape_predictor, batch_size):
    stream = get_data(filenames)
    stream = dataset.categorical(stream)
    stream = features.extract(shape_predictor, (img_x, img_y))(stream)

    batches = dataset.stream_batch(stream, batch_size)
    batches = dataset.batch_adapt(batches, [
        adapters.matrix_to_bn,
        adapters.normalize
    ])

    return batches


def validating(index, batch_size):
    """Build pipeline to validate model."""
    validating, _ = get_filenames(index, k, directory, get_label)

    filenames = validating

    y_val = [kdef.get_label(f) for f in filenames]

    batches = _get_batches(filenames, img_x, img_y, shape_predictor,
                           batch_size)

    return (batches, y_val)


def training(index, batch_size, epochs):
    """Build pipeline to train model."""
    _, training = get_filenames(index, k, directory, get_label)
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
