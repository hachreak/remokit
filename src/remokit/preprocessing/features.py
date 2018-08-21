
"""Feature extraction."""

import os
import numpy as np

from .. import detect, dataset as ds, adapters, utils


def get_face():
    """Extract face only."""
    detector = detect.get_detector()

    def f(img):
        dets = detect.detect(detector, img)
        for det in dets.values():
            det = det['detected']
            return img[det.top():det.bottom(), det.left():det.right()]
    return f


def extract(shape_predictor):
    """Extract features in a face image stream."""
    detector = detect.get_detector()
    predictor = detect.get_predictor(shape_predictor)

    def f(img):
        dets = detect.detect(detector, img)
        shapes = detect.shapes(img, predictor, dets)
        for shape in shapes:
            matrix = detect.shape2matrix(shape)
            return matrix

    return f


def expand2image(img_x, img_y):
    """Expand feature to a image."""
    def f(matrix):
        img = detect.expand2img(matrix)
        return img

    return f


def prepare_batch(config):
    """Extract faces from the dataset."""
    gl = utils.load_fun(config['get_label'])

    stream = utils.load_fun(config['get_files'])(**config)
    stream = ds.stream(ds.add_label(gl), stream)
    stream = ds.stream(ds.apply_to_x(detect.load_img), stream)
    stream = ds.stream(
        ds.apply_to_x(adapters.resize(**config['image_size'])), stream
    )
    stream = ds.stream(ds.apply_to_x(adapters.astype('uint8')), stream)

    if config['has_faces']:
        stream = ds.stream(ds.apply_to_x(get_face()), stream)
    if config.get('only_features', False):
        stream = ds.stream(ds.apply_to_x(extract(
            config['shape_predictor']
        )), stream)
    stream = ds.stream(ds.apply_to_x(adapters.astype('uint8')), stream)

    return stream


def save(batches, config, indices=None):
    """Save preprocessed features."""
    indices = indices or {v: 0 for v in ds._category.keys() + ['index']}
    print(indices)
    for X, y in batches:
        if y == 'neutral':
            indices['index'] += 1
        indices[y] += 1
        filename = '{0:08}_{1}_{2}'.format(indices['index'], y, indices[y])
        destination = os.path.join(config['destination'], filename)
        np.save(destination, X)
        print(destination)
    return indices
