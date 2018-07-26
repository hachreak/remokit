
"""Feature extraction."""

import dlib
from .. import detect


def extract(shape_predictor, img_size):
    """Extract features in a face image stream."""
    detector = detect.get_detector()
    predictor = detect.get_predictor(shape_predictor)

    img_x, img_y = img_size
    img_x = img_x or 100
    img_y = img_y or 100

    def f(stream):
        # FIXME invert img, label
        for cat, img in stream:
            dets = detect.detect(detector, img)
            shapes = detect.shapes(img, predictor, dets)
            for shape in shapes:
                feature = detect.expand2img(detect.shape2matrix(shape))
                img = dlib.resize_image(feature, img_x, img_y)
                yield cat, img

    return f


def adapt(features):
    """Adapt features to input for the CNN."""
    (img_x, img_y) = features[0].shape
    features = features.reshape(features.shape[0], img_x, img_y, 1)
    features = features.astype('float32')
    features /= 255
    return features


def batch_adapt(batches):
    """Adapt a streaming batch."""
    #  import ipdb; ipdb.set_trace()
    for x, y in batches:
        x = adapt(x)
        yield x, y
