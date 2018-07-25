
"""Get dataset."""

import dlib
import os
import numpy as np
from remokit import detect
from keras.utils import to_categorical


_category = {
    'neutral': 0,
    'angry': 1,
    'disgust': 2,
    'afraid': 3,
    'happy': 4,
    'sad': 5,
    'surprised': 6,
}


def files_per_category(filenames, get_label):
    """Return filenames per categories."""
    # init dict
    result = {}
    for cat in _category.keys():
        result[cat] = []
    # split files
    for filename in filenames:
        result[get_label(filename)].append(filename)
    return result


def count(directory, types=None):
    """Count how many files."""
    return len(list(get_files(directory=directory, types=types)))


def get_files(directory, types=None):
    """Get list of images reading recursively."""
    types = types or ['.jpg']
    for root, dirnames, files in os.walk(directory):
        for name in files:
            _, ext = os.path.splitext(name)
            if ext.lower() in types:
                yield os.path.join(root, name)


def first(stream, size):
    """Get only first `size` from the stream."""
    index = 0
    for value in stream:
        yield value
        index += 1
        if index >= size:
            raise StopIteration


def batch(block, size):
    """Split in batches the input."""
    for block in np.array_split(block, size):
        yield block


def stream_batch(stream, size):
    """Create batch on the fly."""
    while True:
        batch = []
        try:
            for i in range(size):
                value = next(stream)
                batch.append(value)
            yield batch
        except StopIteration:
            if not batch:
                raise StopIteration
            yield batch


def stream_training(stream, detector, predictor, img_x=None, img_y=None):
    """Get new x_train/y_train values as streaming."""
    img_x = img_x or 100
    img_y = img_y or 100
    for label, img in stream:
        dets = detect.detect(detector, img)
        shapes = detect.shapes(img, predictor, dets)
        for shape in shapes:
            feature = detect.expand2img(detect.shape2matrix(shape))
            yield (
                to_categorical(label2category(label), len(_category)),
                dlib.resize_image(feature, img_x, img_y)
            )


def _batch_train(stream, size=None):
    """Transform a single x_train/y_train stream in a batch stream."""
    for batch in stream_batch(stream, size):
        x_train = []
        y_train = []
        for label, img in batch:
            x_train.append(img)
            y_train.append(label)
        x_train = feature2input(np.array(x_train))
        yield x_train, np.array(y_train)


def batch_training(stream, detector, predictor, size, img_x=None, img_y=None):
    """Get a batch streaming training of x_test/y_test."""
    return _batch_train(
        stream_training(stream, detector, predictor, img_x, img_y), size
    )


def feature2input(features):
    """Transform a image feature in a input for the CNN."""
    (img_x, img_y) = features[0].shape
    features = features.reshape(features.shape[0], img_x, img_y, 1)
    features = features.astype('float32')
    features /= 255
    return features


def label2category(label):
    """Convert label to category."""
    return _category[label]


def category2label(category):
    """Convert category to label."""
    return np.argmax(category)
