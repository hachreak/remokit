
"""Feature extraction."""

from .. import detect


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
