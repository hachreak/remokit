
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
        for img, cat in stream:
            width = 100
            height = img.shape[1] * 100 / img.shape[0]
            img = dlib.resize_image(img, width, height)
            dets = detect.detect(detector, img)
            shapes = detect.shapes(img, predictor, dets)
            for shape in shapes:
                feature = detect.expand2img(detect.shape2matrix(shape))
                img = dlib.resize_image(feature, img_x, img_y)
                yield img, cat

    return f
