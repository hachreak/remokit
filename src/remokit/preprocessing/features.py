
"""Feature extraction."""

import os
import cv2
import numpy as np

from .. import detect, dataset as ds, adapters, utils


def get_face(img_size=None):
    """Extract face only."""
    detector = detect.get_detector()
    adapter = None
    if img_size:
        adapter = adapters.resize(**img_size)

    def f(img):
        dets = detect.detect(detector, img)
        for det in dets.values():
            det = det['detected']
            img = img[det.top():det.bottom(), det.left():det.right()]
            if adapter:
                img = adapter(img)
            return img
    return f


def extract_shape(shape_predictor):
    """Extract features in a face image stream."""
    detector = detect.get_detector()
    predictor = detect.get_predictor(shape_predictor)

    def f(img):
        dets = detect.detect(detector, img)
        shapes = detect.shapes(img, predictor, dets)
        return shapes[0]
    return f


def expand2image(img_x, img_y):
    """Expand feature to a image."""
    def f(matrix):
        return detect.expand2img(matrix)
    return f


def extract_part(name, extract_shape):
    """Extract a part of the image (e.g. the eyes)."""
    rules = {
        'eyes': [37, 48, 0.1, 0.25],
        'mouth': [49, 68, 0.07, 0.07],
        'nose': [28, 36, 0.07, 0.07],
    }

    def f(img):
        # extract shape
        shape = extract_shape(img)
        matrix = np.array([
            [part.y, part.x] for part in shape.parts()
        ]).astype(np.int32)
        # get only interested shape
        bound = matrix[rules[name][0]: rules[name][1]]
        # extract face part with a small margin
        (x, y, w, h) = cv2.boundingRect(bound)
        margin_x = int(x * rules[name][2])
        margin_y = int(y * rules[name][3])
        return img[x - margin_x:x + w + margin_x,
                   y - margin_y:y + h + margin_y]
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
        stream = ds.stream(ds.apply_to_x(extract_shape(
            config['shape_predictor']
        )), stream)
        stream = ds.stream(ds.apply_to_x(detect.shape2matrix), stream)
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


def get_label(filename):
    return os.path.split(filename)[1].split('_')[1]


def get_files(directory, *args, **kwargs):
    """Get image/label files."""
    return ds.get_files(directory)
